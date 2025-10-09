"""Command-line interface for the PDF cleaner tool."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Optional, Sequence

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .config import Config, load_config
from .passwords import PasswordProvider, load_password_file
from .processor import PDFCleaner, ProcessingEvent, RunStatistics

FILE_COLUMN_WIDTH = 48
BAR_WIDTH = 20
COUNT_COLUMN_WIDTH = 7
ELAPSED_COLUMN_WIDTH = 8
PAGES_COLUMN_WIDTH = 11
STATE_COLUMN_WIDTH = 12



def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Clean PDFs and mirror directory structures using PyMuPDF."
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Path to a PDF or directory containing PDFs to clean.",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a configuration file (default discovery order described in README).",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="path=value",
        help="Override configuration values (e.g. --set clean.sanitize_metadata=false).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Override the configured output directory (io.output_dir).",
    )
    parser.add_argument(
        "--keep-metadata",
        dest="keep_metadata",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Retain original metadata instead of stripping it (toggles clean.sanitize_metadata).",
    )
    parser.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip PDFs when the target file already exists (toggles io.overwrite_existing).",
    )
    parser.add_argument(
        "--run-log",
        dest="run_log",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable writing NDJSON run summaries (io.run_log.enabled).",
    )
    parser.add_argument(
        "--run-log-path",
        dest="run_log_path",
        type=Path,
        help="Override the NDJSON run log destination (io.run_log.path).",
    )
    parser.add_argument(
        "--run-log-quiet",
        dest="run_log_quiet",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Suppress confirmation messages after writing a run log (io.run_log.quiet).",
    )
    parser.add_argument(
        "--password",
        type=str,
        help="Default password applied to encrypted PDFs when no specific match is found.",
    )
    parser.add_argument(
        "--password-file",
        type=Path,
        help="File containing PDF-specific passwords (format: name=password).",
    )
    parser.add_argument(
        "--password-hints",
        type=str,
        help="Filename searched within each directory for PDF passwords (passwords.hint_filename).",
    )
    parser.add_argument(
        "--no-password-hints",
        dest="disable_password_hints",
        action="store_true",
        help="Disable per-directory password hint lookups.",
    )
    parser.add_argument(
        "--log-level",
        choices=["critical", "error", "warning", "info", "debug"],
        default="info",
        help="Logging verbosity (default: info).",
    )
    return parser


def _bool_to_scalar(value: bool) -> str:
    return "true" if value else "false"


def _collect_overrides(args: argparse.Namespace) -> list[str]:
    overrides: list[str] = list(args.overrides or [])
    if args.output_dir:
        overrides.append(f"io.output_dir={args.output_dir}")
    if args.keep_metadata is not None:
        # keep_metadata True -> sanitize_metadata False
        overrides.append(f"clean.sanitize_metadata={_bool_to_scalar(not args.keep_metadata)}")
    if args.skip_existing is not None:
        overrides.append(f"io.overwrite_existing={_bool_to_scalar(not args.skip_existing)}")
    if args.password is not None:
        overrides.append(f"passwords.default={args.password}")
    if args.password_file is not None:
        overrides.append(f"passwords.password_file={args.password_file}")
    if args.disable_password_hints:
        overrides.append("passwords.hint_filename=")
    elif args.password_hints is not None:
        overrides.append(f"passwords.hint_filename={args.password_hints}")
    if args.run_log is not None:
        overrides.append(f"io.run_log.enabled={_bool_to_scalar(args.run_log)}")
    if args.run_log_path is not None:
        overrides.append(f"io.run_log.path={args.run_log_path}")
    if args.run_log_quiet is not None:
        overrides.append(f"io.run_log.quiet={_bool_to_scalar(args.run_log_quiet)}")
    return overrides


def _resolve_process_count(value: Any) -> int:
    """Convert configured process count into a concrete worker total."""
    if isinstance(value, str):
        return max(1, os.cpu_count() or 1)
    return max(1, int(value))


def _configure_logging(log_level: str) -> None:
    """Initialise root logging configuration."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def _iter_paths_in_order(source: Path) -> Iterable[Path]:
    """Yield files under ``source`` following the same ordering as processing."""
    source = source.resolve()
    if source.is_file():
        yield source
        return
    for child in sorted(source.iterdir()):
        if child.is_dir():
            yield from _iter_paths_in_order(child)
        else:
            yield child


def _discover_pdf_files(source: Path) -> list[Path]:
    """Return the list of PDF files scheduled for processing."""
    return [path for path in _iter_paths_in_order(source) if path.suffix.lower() == ".pdf"]


def _format_summary(stats: RunStatistics) -> list[str]:
    """Prepare summary lines for terminal output."""
    lines = [
        f"PDFs processed: {stats.processed} (changed: {stats.changed})",
        f"PDFs skipped: {stats.skipped}",
        f"PDFs unlocked: {stats.unlocked}",
    ]
    if stats.failed:
        lines.append(f"PDFs failed: {stats.failed}")
    if stats.copied:
        lines.append(f"Assets copied: {stats.copied}")
    if stats.total_pages:
        lines.append(f"Pages cleaned: {stats.total_pages}")
    if stats.total_watermarks_removed:
        lines.append(f"Watermarks removed: {stats.total_watermarks_removed}")
    if stats.total_hidden_text_removed:
        lines.append(f"Hidden text removed: {stats.total_hidden_text_removed}")
    if stats.total_image_metadata_cleared:
        lines.append(f"Images metadata cleared: {stats.total_image_metadata_cleared}")
    if stats.documents_metadata_cleared:
        lines.append(f"Documents metadata cleared: {stats.documents_metadata_cleared}")
    lines.append(f"Elapsed: {stats.total_elapsed:.2f}s")
    return lines


def _truncate_middle(text: str, width: int) -> str:
    """Return text truncated in the middle so the tail (usually filename) stays visible."""
    if len(text) <= width:
        return text
    return f"…{text[-(width - 1):]}"


def _relative_path_for_display(path: Path, base: Path, *, prefer_name: bool) -> str:
    """Return a human friendly display path relative to the provided base."""
    if prefer_name:
        return path.name
    try:
        return str(path.relative_to(base))
    except ValueError:
        return path.name


def _format_duration(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_eta(elapsed: float, completed: int, total: Optional[int]) -> str:
    if not total or completed == 0:
        return "-:--:--"
    remaining = total - completed
    if remaining <= 0:
        return "00:00:00"
    average = elapsed / completed
    eta = average * remaining
    return _format_duration(eta)


def _format_step_time(seconds: Optional[float]) -> str:
    if seconds is None:
        return "  --.--s"
    return f"{seconds:7.2f}s"


def _format_pages(count: Optional[int]) -> str:
    if count is None:
        return "   -- pages"
    return f"{count:5d} pages"


def _format_state_label(event: ProcessingEvent) -> str:
    """Return a fixed-width label describing the current event state."""
    detail = event.message or ""
    if event.kind == "processed" and event.result is not None:
        detail = "changed" if event.result.changed else "no-change"
        base = detail
    elif detail:
        base = f"{event.kind}-{detail}"
    else:
        base = event.kind
    base = base.replace("_", "-")
    return base[:STATE_COLUMN_WIDTH].ljust(STATE_COLUMN_WIDTH)


def _build_bar(completed: int, total: Optional[int], width: int = BAR_WIDTH) -> str:
    if not total or total <= 0:
        return "-" * width
    ratio = max(0.0, min(1.0, completed / total))
    filled = int(ratio * width)
    return "=" * filled + "-" * (width - filled)


def _format_count(completed: int, total: Optional[int]) -> str:
    if not total or total <= 0:
        return "--/--"
    digits = len(str(total))
    return f"{completed:>{digits}}/{total}"


def _label_for_event(event: ProcessingEvent) -> str:
    mapping = {
        "processed": "Cleaning PDF...",
        "skipped": "Skipping PDF...",
        "failed": "Failed PDF...",
        "copied": "Copying asset...",
    }
    return mapping.get(event.kind, "Working...")


def _stats_to_dict(stats: RunStatistics) -> dict[str, Any]:
    """Convert RunStatistics to a JSON-friendly dictionary."""
    return {
        "processed": stats.processed,
        "skipped": stats.skipped,
        "copied": stats.copied,
        "failed": stats.failed,
        "unlocked": stats.unlocked,
        "changed": stats.changed,
        "total_pages": stats.total_pages,
        "total_elapsed": stats.total_elapsed,
        "total_watermarks_removed": stats.total_watermarks_removed,
        "total_hidden_text_removed": stats.total_hidden_text_removed,
        "total_image_metadata_cleared": stats.total_image_metadata_cleared,
        "documents_metadata_cleared": stats.documents_metadata_cleared,
    }


def _write_run_log(
    console: Console,
    config: Any,
    cleaner: PDFCleaner,
    run_started_at: datetime,
    run_completed_at: datetime,
    stats: RunStatistics,
    files: list[dict[str, Any]],
    processes_used: int,
    batch_size: int,
    source: Path,
) -> None:
    """Persist run metadata to an NDJSON log when enabled."""
    if not config.io.run_log.enabled:
        return

    raw_path = config.io.run_log.path
    if raw_path is None:
        log_path = cleaner.output_dir / "run_log.ndjson"
    else:
        log_path = raw_path

    log_path = Path(log_path).expanduser()
    if not log_path.is_absolute():
        log_path = Path.cwd() / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    duration = (run_completed_at - run_started_at).total_seconds()
    stats_payload = _stats_to_dict(stats)
    record: dict[str, Any] = {
        "version": 1,
        "started_at": run_started_at.isoformat(),
        "completed_at": run_completed_at.isoformat(),
        "duration": round(duration, 6),
        "source": str(source.resolve()),
        "output_dir": str(cleaner.output_dir.resolve()),
        "stats": stats_payload,
        "files": files,
        "config": {
            "overwrite_existing": config.io.overwrite_existing,
            "skip_unchanged": config.io.skip_unchanged,
            "clean": {
                "strip_document_metadata": config.clean.strip_document_metadata,
                "strip_image_metadata": config.clean.strip_image_metadata,
                "remove_hidden_text": config.clean.remove_hidden_text,
                "hidden_text_alpha_threshold": config.clean.hidden_text_alpha_threshold,
                "extract_text": config.clean.extract_text,
            },
            "watermarks": {
                "enabled": config.watermarks.enabled,
                "clip_bottom_mm": config.watermarks.clip_bottom_mm,
                "stop_after_first": config.watermarks.stop_after_first,
                "max_pages": config.watermarks.max_pages,
            },
            "save": {
                "linearize": config.save.linearize,
                "garbage": config.save.garbage,
                "deflate": config.save.deflate,
            },
            "performance": {
                "configured_processes": config.performance.processes,
                "resolved_processes": processes_used,
                "batch_size": batch_size,
            },
            "run_log": {
                "path": str(config.io.run_log.path) if config.io.run_log.path else None,
                "enabled": bool(config.io.run_log.enabled),
                "quiet": bool(config.io.run_log.quiet),
            },
        },
    }
    if stats.failures:
        record["failures"] = [
            {"source": str(path), "reason": reason}
            for path, reason in stats.failures
        ]
    quiet = bool(config.io.run_log.quiet)
    try:
        with log_path.open("a", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False)
            handle.write("\n")
    except OSError as exc:
        console.print(f"[bold red]Failed to write run log:[/bold red] {exc}", highlight=False)
    else:
        if not quiet:
            console.print(f"[dim]Run log appended to {log_path}[/dim]", highlight=False)


def _print_legend(console: Console, total: Optional[int]) -> None:
    """Display a legend explaining each column in the progress output."""
    count_example = "count" if not total else f"0/{total}"
    header = (
        f"Format: spin | action            | progress{' ' * (BAR_WIDTH - 8)}| "
        f"{count_example.rjust(COUNT_COLUMN_WIDTH)} | elapsed  | eta      | step   | pages       | file"
    )
    detail = (
        "        life | current step      | = done, - remaining     | "
        f"{'done/total'.rjust(COUNT_COLUMN_WIDTH)} | HH:MM:SS | HH:MM:SS | seconds | pages/assets | relative path"
    )
    console.print(header, highlight=False)
    console.print(detail, style="dim", highlight=False)


def _build_password_provider(config) -> PasswordProvider:
    mapping = {}
    if config.passwords.password_file:
        mapping.update(load_password_file(config.passwords.password_file))
    return PasswordProvider(
        default=config.passwords.default,
        global_mapping=mapping or None,
        hint_filename=config.passwords.hint_filename,
    )


def _worker_clean_task(args: tuple[dict[str, Any], str, str]) -> tuple[str, Any]:
    """
    Execute a single PDF cleaning task in a worker process.

    Returns:
        ("ok", DocumentProcessingResult) on success or ("error", (path, message)) on failure.
    """
    config_dict, source_str, output_str = args
    pdf_path = Path(source_str)
    output_path = Path(output_str)
    try:
        config = Config.model_validate(config_dict)
        config.performance.processes = 1
        config.performance.batch_size = 1
        cleaner = PDFCleaner.from_config(config, password_provider=_build_password_provider(config))
        result = cleaner.clean_document(pdf_path, output_path=output_path)
        return ("ok", result)
    except Exception as exc:  # pragma: no cover - executed in worker
        return ("error", (source_str, repr(exc)))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Execute the CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)

    if not args.source.exists():
        parser.error(f"Source path not found: {args.source}")

    console = Console()

    overrides = _collect_overrides(args)
    config = load_config(args.config, overrides)

    password_provider = _build_password_provider(config)
    cleaner = PDFCleaner.from_config(config, password_provider=password_provider)

    processes = _resolve_process_count(config.performance.processes)
    batch_size = max(1, config.performance.batch_size)
    pdf_candidates = _discover_pdf_files(args.source)
    total_pdfs = len(pdf_candidates)
    use_parallel = processes > 1 and total_pdfs > 1

    source_root = args.source.resolve()
    prefer_name = source_root.is_file()
    display_base = source_root if source_root.is_dir() else source_root.parent

    run_started_at = datetime.now(timezone.utc)
    run_completed_at: Optional[datetime] = None
    stats = RunStatistics()
    file_records: list[dict[str, Any]] = []

    console.print(
        f"[bold cyan]Cleaning[/bold cyan] {args.source} -> {cleaner.output_dir.resolve()}",
        highlight=False,
    )
    _print_legend(console, total_pdfs)

    spinner_cycle = itertools.cycle(["|", "/", "-", "\\"])
    start_time = perf_counter()
    completed = 0
    task_total = total_pdfs if total_pdfs else None

    with console.status("Preparing run...", spinner="dots") as status:
        progress_columns = (
            SpinnerColumn(),
            TextColumn("{task.description:<16}", justify="left"),
            BarColumn(bar_width=20),
            TextColumn("{task.completed}/{task.total}" if task_total else "{task.completed}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn(f"| {{task.fields[file]:<{FILE_COLUMN_WIDTH}}}", justify="left"),
            TextColumn(f"| {{task.fields[elapsed]:>{ELAPSED_COLUMN_WIDTH}}}", justify="right"),
            TextColumn(f"| {{task.fields[pages]:>{PAGES_COLUMN_WIDTH}}}", justify="right"),
            TextColumn(f"| {{task.fields[state]:<{STATE_COLUMN_WIDTH}}}", justify="left"),
        )

        with Progress(*progress_columns, console=console, transient=True) as progress:
            task_id = progress.add_task(
                "Cleaning PDFs",
                total=task_total,
                file="",
                elapsed="",
                pages="",
                state="",
            )

            def handle_event(event: ProcessingEvent) -> None:
                nonlocal completed
                if event.kind in {"processed", "skipped", "failed"}:
                    completed += 1

                spinner = next(spinner_cycle)
                label = _label_for_event(event)
                bar = _build_bar(completed, task_total)
                count_str = _format_count(completed, task_total)
                elapsed_total = perf_counter() - start_time
                elapsed_str = _format_duration(elapsed_total)
                eta_str = _format_eta(elapsed_total, completed, task_total)

                result = event.result
                step_time = result.elapsed if result else None
                pages_value = result.page_count if result else None
                pages_str = _format_pages(pages_value) if event.kind != "copied" else "   asset   "
                filename = _truncate_middle(
                    _relative_path_for_display(event.source, display_base, prefer_name=prefer_name),
                    FILE_COLUMN_WIDTH,
                )
                state_label = _format_state_label(event)

                progress.update(
                    task_id,
                    fields={
                        "file": filename,
                        "elapsed": _format_step_time(step_time),
                        "pages": pages_str,
                        "state": state_label,
                    },
                )
                if event.kind in {"processed", "skipped", "failed"}:
                    progress.advance(task_id, 1)

                status.update(f"{spinner} {label} {filename}")
                console.print(
                    f"{spinner} {label:<17} | {bar} | {count_str:>{COUNT_COLUMN_WIDTH}} | {elapsed_str} | "
                    f"{eta_str} | {_format_step_time(step_time)} | {pages_str:<11} | {filename}",
                    highlight=False,
                )

                if event.kind == "processed" and result is not None:
                    stats.add_result(result)
                    record = {
                        "source": str(event.source),
                        "status": event.kind,
                        "output": str(result.output),
                        "elapsed": round(result.elapsed, 6),
                        "pages": result.page_count,
                        "watermarks_removed": result.watermarks_removed,
                        "hidden_text_removed": result.hidden_text_removed,
                        "image_metadata_cleared": result.image_metadata_cleared,
                        "document_metadata_cleared": result.document_metadata_cleared,
                        "was_encrypted": result.was_encrypted,
                        "changed": result.changed,
                        "step_timings": {k: round(v, 6) for k, v in result.step_timings.items()},
                    }
                    if event.message:
                        record["detail"] = event.message
                    file_records.append(record)
                elif event.kind == "skipped":
                    stats.skipped += 1
                elif event.kind == "copied":
                    stats.copied += 1
                elif event.kind == "failed":
                    message = str(event.error) if event.error else (event.message or "")
                    stats.add_failure(event.source, message)
                    console.print(
                        f"    -> [bold red]{message or 'Unknown error'}[/bold red]",
                        highlight=False,
                    )

            if not use_parallel:
                for _ in cleaner.process_path(args.source, progress_callback=handle_event):
                    pass
            else:
                output_root = cleaner.output_dir
                output_root.mkdir(parents=True, exist_ok=True)
                pdf_tasks: list[tuple[Path, Path]] = []
                for path in _iter_paths_in_order(args.source):
                    if path.is_dir():
                        continue
                    relative = path.relative_to(display_base)
                    destination = output_root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if path.suffix.lower() == ".pdf":
                        if path.name.startswith("._") or path.stat().st_size == 0:
                            handle_event(ProcessingEvent(source=path, kind="skipped", message="resource-fork"))
                            continue
                        if destination.exists() and not cleaner.overwrite:
                            handle_event(ProcessingEvent(source=path, kind="skipped", message="exists"))
                            continue
                        pdf_tasks.append((path, destination))
                    else:
                        if destination.exists() and not cleaner.overwrite:
                            handle_event(ProcessingEvent(source=path, kind="skipped", message="asset-exists"))
                            continue
                        shutil.copy2(path, destination)
                        handle_event(ProcessingEvent(source=path, kind="copied", message="asset"))

                if pdf_tasks:
                    worker_config = config.model_dump()
                    worker_config.setdefault("performance", {})["processes"] = 1
                    worker_config["performance"]["batch_size"] = 1
                    task_args = [
                        (worker_config, str(source), str(dest))
                        for source, dest in pdf_tasks
                    ]
                    with ProcessPoolExecutor(max_workers=processes) as executor:
                        results = executor.map(_worker_clean_task, task_args, chunksize=batch_size)
                        for (source, _dest), outcome in zip(pdf_tasks, results):
                            if outcome[0] == "ok":
                                result = outcome[1]
                                message = "changed" if result.changed else "no-change"
                                handle_event(
                                    ProcessingEvent(
                                        source=source,
                                        kind="processed",
                                        result=result,
                                        message=message,
                                    )
                                )
                            else:
                                _status, (err_path, err_msg) = outcome
                                handle_event(
                                    ProcessingEvent(
                                        source=Path(err_path),
                                        kind="failed",
                                        error=RuntimeError(err_msg),
                                        message="worker-error",
                                    )
                                )

            run_completed_at = datetime.now(timezone.utc)
            status.update("Processing complete")

    stats.total_elapsed = (run_completed_at - run_started_at).total_seconds()

    summary_lines = _format_summary(stats)
    console.print(f"[bold green]Processing complete[/bold green] -> {cleaner.output_dir.resolve()}")
    for line in summary_lines:
        console.print(f"  - {line}")
    if stats.failures:
        console.print("[bold red]Failures[/bold red]")
        for failed_path, reason in stats.failures:
            display_path = _truncate_middle(
                _relative_path_for_display(failed_path, display_base, prefer_name=prefer_name),
                FILE_COLUMN_WIDTH,
            )
            console.print(f"    - {display_path} | {reason}")

    processes_used = processes if use_parallel else 1
    _write_run_log(
        console,
        config,
        cleaner,
        run_started_at,
        run_completed_at,
        stats,
        file_records,
        processes_used,
        batch_size,
        args.source,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
