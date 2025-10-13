"""Command-line interface for the PDF cleaner tool."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Optional, Sequence

import fitz  # type: ignore

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
from .processor import DocumentProcessingResult, PDFCleaner, ProcessingEvent, RunStatistics
from .watermarks import find_watermark_matches

FILE_COLUMN_WIDTH = 48
BAR_WIDTH = 20
COUNT_COLUMN_WIDTH = 7
ELAPSED_COLUMN_WIDTH = 8
PAGES_COLUMN_WIDTH = 11
SIZE_COLUMN_WIDTH = 32

_STAGE_TIMING_FIELDS: tuple[tuple[str, str], ...] = (
    ("open", "open_ms"),
    ("detect_password", "detect_password_ms"),
    ("sanitize_metadata", "sanitize_ms"),
    ("remove_hidden_text", "hidden_text_ms"),
    ("watermark_scan", "watermark_ms"),
    ("save", "save_ms"),
    ("extract_text", "extract_ms"),
)

_STAGE_SUMMARY_LABELS = {
    ms_key: stage.replace("_", " ")
    for stage, ms_key in _STAGE_TIMING_FIELDS
}


def _format_stage_timings_ms(step_timings: dict[str, float]) -> dict[str, float]:
    """Convert per-stage timings (seconds) into millisecond fields for logging."""
    return {
        ms_key: round(step_timings.get(stage, 0.0) * 1000, 3)
        for stage, ms_key in _STAGE_TIMING_FIELDS
    }


def _add_process_arguments(parser: argparse.ArgumentParser) -> None:
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
        "--profile",
        type=str,
        help="Configuration profile to apply (configs/profiles/<name>.toml).",
    )


def _add_config_only_arguments(parser: argparse.ArgumentParser) -> None:
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
        "--profile",
        type=str,
        help="Configuration profile to apply (configs/profiles/<name>.toml).",
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Clean PDFs and related utilities using PyMuPDF."
    )
    parser.add_argument(
        "--log-level",
        choices=["critical", "error", "warning", "info", "debug"],
        default="info",
        help="Logging verbosity (default: info).",
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    process_parser = subparsers.add_parser("process", help="Clean PDFs (default workflow).")
    _add_process_arguments(process_parser)

    dry_run_parser = subparsers.add_parser(
        "dry-run",
        help="Inspect PDFs, run detections, and report results without writing outputs.",
    )
    _add_process_arguments(dry_run_parser)

    rules_parser = subparsers.add_parser("rules", help="Watermark rule utilities.")
    rules_subparsers = rules_parser.add_subparsers(dest="rules_command")
    rules_subparsers.required = True
    rules_test_parser = rules_subparsers.add_parser(
        "test",
        help="Evaluate configured watermark rules against text samples or a PDF page clip.",
    )
    _add_config_only_arguments(rules_test_parser)
    rules_test_parser.add_argument(
        "--text",
        action="append",
        help="Text sample to test against all rules. Provide multiple times for multiple samples.",
    )
    rules_test_parser.add_argument(
        "--rule",
        action="append",
        help="Limit evaluation to the specified rule name(s).",
    )
    rules_test_parser.add_argument(
        "--pdf",
        type=Path,
        help="Optional PDF path to evaluate watermark rules against.",
    )
    rules_test_parser.add_argument(
        "--page",
        type=int,
        default=0,
        help="Zero-based page number when testing against a specific PDF (default: 0).",
    )
    rules_test_parser.add_argument(
        "--clip-bottom-mm",
        type=float,
        help="Override the configured bottom clip height when extracting watermark text.",
    )
    rules_test_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of matches reported from a PDF page clip (0 = all).",
    )

    print_config_parser = subparsers.add_parser(
        "print-config",
        help="Display the merged configuration after applying overrides.",
    )
    _add_config_only_arguments(print_config_parser)

    stats_parser = subparsers.add_parser(
        "stats",
        help="Summarise one or more NDJSON run log files.",
    )
    stats_parser.add_argument(
        "--run-log-path",
        type=Path,
        help="Path to a run_log.ndjson file (defaults to configured io.run_log.path).",
    )
    stats_parser.add_argument(
        "--latest",
        action="store_true",
        help="Only display statistics from the most recent run in the NDJSON log.",
    )
    _add_config_only_arguments(stats_parser)

    return parser


def _bool_to_scalar(value: bool) -> str:
    return "true" if value else "false"


def _collect_overrides(args: argparse.Namespace) -> list[str]:
    overrides: list[str] = list(args.overrides or [])
    output_dir = getattr(args, "output_dir", None)
    if output_dir:
        overrides.append(f"io.output_dir={args.output_dir}")
    keep_metadata = getattr(args, "keep_metadata", None)
    if keep_metadata is not None:
        # keep_metadata True -> sanitize_metadata False
        overrides.append(f"clean.sanitize_metadata={_bool_to_scalar(not keep_metadata)}")
    skip_existing = getattr(args, "skip_existing", None)
    if skip_existing is not None:
        overrides.append(f"io.overwrite_existing={_bool_to_scalar(not skip_existing)}")
    password = getattr(args, "password", None)
    if password is not None:
        overrides.append(f"passwords.default={password}")
    password_file = getattr(args, "password_file", None)
    if password_file is not None:
        overrides.append(f"passwords.password_file={password_file}")
    disable_hints = getattr(args, "disable_password_hints", False)
    if disable_hints:
        overrides.append("passwords.hint_filename=")
    else:
        password_hints = getattr(args, "password_hints", None)
        if password_hints is not None:
            overrides.append(f"passwords.hint_filename={password_hints}")
    run_log = getattr(args, "run_log", None)
    if run_log is not None:
        overrides.append(f"io.run_log.enabled={_bool_to_scalar(run_log)}")
    run_log_path = getattr(args, "run_log_path", None)
    if run_log_path is not None:
        overrides.append(f"io.run_log.path={run_log_path}")
    run_log_quiet = getattr(args, "run_log_quiet", None)
    if run_log_quiet is not None:
        overrides.append(f"io.run_log.quiet={_bool_to_scalar(run_log_quiet)}")
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


def _format_status_counts(stats: RunStatistics) -> str:
    """Render a compact status line summarising current run counts."""
    return (
        f"done:{stats.processed} "
        f"chg:{stats.changed} "
        f"unlock:{stats.unlocked} "
        f"skip:{stats.skipped} "
        f"fail:{stats.failed}"
    )


def _print_stage_summary(console: Console, file_records: Iterable[dict[str, Any]]) -> None:
    """Display average per-stage timings gathered during the run."""
    processed_records = [record for record in file_records if record.get("status") == "processed"]
    if not processed_records:
        return
    totals = {key: 0.0 for key in _STAGE_SUMMARY_LABELS}
    for record in processed_records:
        timings = record.get("stage_timings_ms") or {}
        for key in totals:
            totals[key] += float(timings.get(key, 0.0))
    count = len(processed_records)
    console.print("[bold cyan]Stage timings (avg ms)[/bold cyan]", highlight=False)
    for key, label in _STAGE_SUMMARY_LABELS.items():
        average = totals[key] / count if count else 0.0
        console.print(f"  - {label}: {average:.1f} ms", highlight=False)


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


def _format_pages_info(result: Optional[DocumentProcessingResult], event_kind: str) -> str:
    if result is None:
        return "   -- pages"
    delta = result.page_count if result.changed else 0
    return f"{result.page_count} pages (Δ{delta})"


def _format_size_delta(delta: int) -> str:
    """Render a human-readable size delta."""
    if delta == 0:
        return "±0B"
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(abs(delta))
    unit = units[0]
    for candidate in units:
        unit = candidate
        if value < 1024 or candidate == units[-1]:
            break
        value /= 1024
    if unit == "B":
        magnitude = f"{int(value)}"
    else:
        magnitude = f"{value:.1f}".rstrip("0").rstrip(".")
    sign = "+" if delta > 0 else "-"
    return f"{sign}{magnitude}{unit}"


def _format_size_bytes(value: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(value)
    unit = units[0]
    for candidate in units:
        unit = candidate
        if abs(size) < 1024 or candidate == units[-1]:
            break
        size /= 1024
    if unit == "B":
        return f"{int(size)}B"
    return f"{size:.2f}{unit}"


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
                "scan_mode": config.watermarks.scan_mode,
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
    record["dry_run"] = bool(getattr(cleaner, "dry_run", False))
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


def _worker_clean_task(args: tuple[dict[str, Any], str, str, bool]) -> tuple[str, Any]:
    """
    Execute a single PDF cleaning task in a worker process.

    Returns:
        ("ok", DocumentProcessingResult) on success or ("error", (path, message)) on failure.
    """
    config_dict, source_str, output_str, dry_run = args
    pdf_path = Path(source_str)
    output_path = Path(output_str)
    try:
        config = Config.model_validate(config_dict)
        config.performance.processes = 1
        config.performance.batch_size = 1
        cleaner = PDFCleaner.from_config(
            config,
            password_provider=_build_password_provider(config),
            dry_run=dry_run,
        )
        result = cleaner.clean_document(pdf_path, output_path=output_path)
        return ("ok", result)
    except Exception as exc:  # pragma: no cover - executed in worker
        return ("error", (source_str, repr(exc)))


def _run_process_command(args: argparse.Namespace, *, dry_run: bool) -> int:
    console = Console()
    source_path = args.source
    if not source_path.exists():
        console.print(f"[bold red]Source path not found:[/bold red] {source_path}", highlight=False)
        return 2

    overrides = _collect_overrides(args)
    profile = getattr(args, "profile", None)
    config = load_config(args.config, overrides, profile=profile)

    password_provider = _build_password_provider(config)
    cleaner = PDFCleaner.from_config(
        config,
        password_provider=password_provider,
        dry_run=dry_run,
    )

    processes = _resolve_process_count(config.performance.processes)
    batch_size = max(1, config.performance.batch_size)

    return _execute_processing(console, args, config, cleaner, processes, batch_size)


def _execute_processing(
    console: Console,
    args: argparse.Namespace,
    config: Config,
    cleaner: PDFCleaner,
    processes: int,
    batch_size: int,
) -> int:
    source = args.source
    pdf_candidates = _discover_pdf_files(source)
    total_pdfs = len(pdf_candidates)
    use_parallel = processes > 1 and total_pdfs > 1

    source_root = source.resolve()
    prefer_name = source_root.is_file()
    display_base = source_root if source_root.is_dir() else source_root.parent

    run_started_at = datetime.now(timezone.utc)
    run_completed_at: Optional[datetime] = None
    stats = RunStatistics()
    file_records: list[dict[str, Any]] = []

    console.print(
        f"[bold cyan]Cleaning[/bold cyan] {source} -> {cleaner.output_dir.resolve()}",
        highlight=False,
    )
    if cleaner.dry_run:
        console.print("[dim]Dry run: outputs will not be written.[/dim]", highlight=False)
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
            TextColumn(f"| {{task.fields[step]:>{ELAPSED_COLUMN_WIDTH}}}", justify="right"),
            TextColumn(f"| {{task.fields[pages]:<{PAGES_COLUMN_WIDTH + 8}}}", justify="left"),
            TextColumn(f"| {{task.fields[size]:<{SIZE_COLUMN_WIDTH}}}", justify="left"),
            TextColumn(f"| {{task.fields[file]:<{FILE_COLUMN_WIDTH}}}", justify="left"),
        )

        with Progress(*progress_columns, console=console, transient=True) as progress:
            task_id = progress.add_task(
                "Cleaning PDFs",
                total=task_total,
                file="",
                step="",
                pages="",
                size="",
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
                pages_str = (
                    _format_pages_info(result, event.kind)
                    if event.kind == "processed"
                    else ("asset" if event.kind == "copied" else "   -- pages")
                )
                filename = _truncate_middle(
                    _relative_path_for_display(event.source, display_base, prefer_name=prefer_name),
                    FILE_COLUMN_WIDTH,
                )
                if result is not None:
                    size_info = (
                        f"{_format_size_bytes(result.original_size_bytes)} -> "
                        f"{_format_size_bytes(result.output_size_bytes)} "
                        f"({_format_size_delta(result.size_delta_bytes)})"
                    )
                else:
                    size_info = "--"

                progress.update(
                    task_id,
                    fields={
                        "file": filename,
                        "step": _format_step_time(step_time),
                        "pages": pages_str,
                        "size": size_info,
                    },
                )
                if event.kind in {"processed", "skipped", "failed"}:
                    progress.advance(task_id, 1)

                if event.kind == "processed" and result is not None:
                    stats.add_result(result)
                    stage_timings_ms = _format_stage_timings_ms(result.step_timings)
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
                        "original_size_bytes": result.original_size_bytes,
                        "output_size_bytes": result.output_size_bytes,
                        "size_delta_bytes": result.size_delta_bytes,
                        "was_encrypted": result.was_encrypted,
                        "changed": result.changed,
                        "step_timings": {k: round(v, 6) for k, v in result.step_timings.items()},
                    }
                    record.update(stage_timings_ms)
                    record["stage_timings_ms"] = stage_timings_ms
                    if event.message:
                        record["detail"] = event.message
                    file_records.append(record)
                elif event.kind == "skipped":
                    stats.skipped += 1
                elif event.kind == "copied":
                    if event.message != "asset-dry-run":
                        stats.copied += 1
                elif event.kind == "failed":
                    message = str(event.error) if event.error else (event.message or "")
                    stats.add_failure(event.source, message)
                    console.print(
                        f"    -> [bold red]{message or 'Unknown error'}[/bold red]",
                        highlight=False,
                    )

                status.update(f"{spinner} {label} | {_format_status_counts(stats)}")
                console.print(
                    f"{spinner} {label:<17} | {bar} | {count_str:>{COUNT_COLUMN_WIDTH}} | {elapsed_str} | "
                    f"{eta_str} | {_format_step_time(step_time)} | {pages_str:<{PAGES_COLUMN_WIDTH + 8}} | "
                    f"{size_info:<{SIZE_COLUMN_WIDTH}} | {filename}",
                    highlight=False,
                )

            if not use_parallel:
                for _ in cleaner.process_path(source, progress_callback=handle_event):
                    pass
            else:
                output_root = cleaner.output_dir
                if not cleaner.dry_run:
                    output_root.mkdir(parents=True, exist_ok=True)
                pdf_tasks: list[tuple[Path, Path]] = []
                for path in _iter_paths_in_order(source):
                    if path.is_dir():
                        continue
                    relative = path.relative_to(display_base)
                    destination = output_root / relative
                    if not cleaner.dry_run:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                    if path.suffix.lower() == ".pdf":
                        if path.name.startswith("._") or path.stat().st_size == 0:
                            handle_event(ProcessingEvent(source=path, kind="skipped", message="resource-fork"))
                            continue
                        if not cleaner.dry_run and destination.exists() and not cleaner.overwrite:
                            handle_event(ProcessingEvent(source=path, kind="skipped", message="exists"))
                            continue
                        pdf_tasks.append((path, destination))
                    else:
                        if not cleaner.dry_run and destination.exists() and not cleaner.overwrite:
                            handle_event(ProcessingEvent(source=path, kind="skipped", message="asset-exists"))
                            continue
                        if cleaner.dry_run:
                            handle_event(ProcessingEvent(source=path, kind="copied", message="asset-dry-run"))
                        else:
                            shutil.copy2(path, destination)
                            handle_event(ProcessingEvent(source=path, kind="copied", message="asset"))

                if pdf_tasks:
                    worker_config = config.model_dump(mode="json")
                    with ProcessPoolExecutor(max_workers=processes) as executor:
                        for start in range(0, len(pdf_tasks), batch_size):
                            chunk = pdf_tasks[start : start + batch_size]
                            task_args = [
                                (worker_config, str(source), str(dest), cleaner.dry_run)
                                for source, dest in chunk
                            ]
                            results = executor.map(_worker_clean_task, task_args, chunksize=1)
                            for (task_source, _dest), outcome in zip(chunk, results):
                                if outcome[0] == "ok":
                                    result = outcome[1]
                                    message = "changed" if result.changed else "no-change"
                                    handle_event(
                                        ProcessingEvent(
                                            source=task_source,
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

    stats.total_elapsed = (run_completed_at - run_started_at).total_seconds() if run_completed_at else 0.0

    summary_lines = _format_summary(stats)
    console.print(f"[bold green]Processing complete[/bold green] -> {cleaner.output_dir.resolve()}", highlight=False)
    for line in summary_lines:
        console.print(f"  - {line}", highlight=False)
    _print_stage_summary(console, file_records)
    if stats.failures:
        console.print("[bold red]Failures[/bold red]", highlight=False)
        for failed_path, reason in stats.failures:
            display_path = _truncate_middle(
                _relative_path_for_display(failed_path, display_base, prefer_name=prefer_name),
                FILE_COLUMN_WIDTH,
            )
            console.print(f"    - {display_path} | {reason}", highlight=False)

    processes_used = processes if use_parallel else 1
    run_log_path = _write_run_log(
        console,
        config,
        cleaner,
        run_started_at,
        run_completed_at or run_started_at,
        stats,
        file_records,
        processes_used,
        batch_size,
        source,
    )
    if config.io.run_log.enabled:
        if not config.io.run_log.quiet:
            if run_log_path is not None:
                console.print(f"[dim]Run log: {run_log_path}[/dim]", highlight=False)
            else:
                console.print("[bold red]Failed to write run log[/bold red]", highlight=False)
    else:
        console.print("[dim]Run log disabled (enable with --run-log)[/dim]", highlight=False)
    return 0


def _run_rules_command(args: argparse.Namespace) -> int:
    if args.rules_command == "test":
        return _run_rules_test_command(args)
    Console().print(f"[bold red]Unknown rules subcommand:[/bold red] {args.rules_command}", highlight=False)
    return 1


def _run_rules_test_command(args: argparse.Namespace) -> int:
    console = Console()
    overrides = _collect_overrides(args)
    profile = getattr(args, "profile", None)
    config = load_config(args.config, overrides, profile=profile)
    rules = config.compile_watermark_rules()
    if args.rule:
        requested = {name.lower() for name in args.rule}
        rules = [rule for rule in rules if rule.name.lower() in requested]
        if not rules:
            console.print("[bold red]No matching rules found for provided --rule filters.[/bold red]", highlight=False)
            return 1

    if not args.text and not args.pdf:
        console.print(
            "[bold yellow]No inputs provided.[/bold yellow] Supply --text to test strings or --pdf to inspect a page.",
            highlight=False,
        )
        return 1

    if args.text:
        console.print("[bold cyan]Text sample evaluation[/bold cyan]", highlight=False)
        for sample in args.text:
            matches = [rule.name for rule in rules if rule.pattern.search(sample)]
            if matches:
                console.print(f"  '{sample}' -> matches {', '.join(matches)}", highlight=False)
            else:
                console.print(f"  '{sample}' -> no matches", style="dim", highlight=False)

    if args.pdf:
        pdf_path = args.pdf
        if not pdf_path.exists():
            console.print(f"[bold red]PDF not found:[/bold red] {pdf_path}", highlight=False)
            return 2
        clip_mm = args.clip_bottom_mm
        if clip_mm is None:
            clip_mm = config.watermarks.clip_bottom_mm or 0.0
        try:
            with fitz.open(pdf_path) as document:
                if document.page_count == 0:
                    console.print("[bold yellow]PDF has no pages to inspect.[/bold yellow]", highlight=False)
                    return 0
                page_index = min(max(args.page, 0), document.page_count - 1)
                page = document[page_index]
                clip_rect = None
                if clip_mm and clip_mm > 0:
                    height_pts = min(float(clip_mm) * 72.0 / 25.4, float(page.rect.height))
                    y0 = max(float(page.rect.y1) - height_pts, float(page.rect.y0))
                    clip_rect = fitz.Rect(page.rect.x0, y0, page.rect.x1, page.rect.y1)
                matches = find_watermark_matches(
                    page,
                    rules=rules,
                    clip_rect=clip_rect,
                    stop_after_first=False,
                )
                if args.limit and args.limit > 0:
                    matches = matches[: args.limit]
                console.print(
                    f"[bold cyan]PDF evaluation[/bold cyan] {pdf_path} (page {page_index})",
                    highlight=False,
                )
                if clip_rect is not None:
                    console.print(
                        f"  clip: y0={clip_rect.y0:.1f}, y1={clip_rect.y1:.1f}",
                        style="dim",
                        highlight=False,
                    )
                if not matches:
                    console.print("  No watermark matches detected.", highlight=False)
                else:
                    for match in matches:
                        console.print(
                            f"  - {match.rule.name}: '{match.text}' at {match.rect}",
                            highlight=False,
                        )
        except fitz.FileDataError as exc:  # pragma: no cover - passthrough to user
            console.print(f"[bold red]Failed to open PDF:[/bold red] {exc}", highlight=False)
            return 2

    return 0


def _run_print_config_command(args: argparse.Namespace) -> int:
    console = Console()
    overrides = _collect_overrides(args)
    profile = getattr(args, "profile", None)
    config = load_config(args.config, overrides, profile=profile)
    console.print(json.dumps(config.model_dump(mode="json"), indent=2))
    return 0


def _resolve_run_log_path(config: Config, run_log_override: Optional[Path]) -> Path:
    if run_log_override is not None:
        path = Path(run_log_override).expanduser()
    else:
        configured = config.io.run_log.path
        if configured:
            path = Path(configured).expanduser()
        else:
            path = Path(config.io.output_dir) / "run_log.ndjson"
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _run_stats_command(args: argparse.Namespace) -> int:
    console = Console()
    overrides = _collect_overrides(args)
    profile = getattr(args, "profile", None)
    config = load_config(args.config, overrides, profile=profile)
    log_path = _resolve_run_log_path(config, args.run_log_path)
    if not log_path.exists():
        console.print(f"[bold red]Run log not found:[/bold red] {log_path}", highlight=False)
        return 1

    records: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        console.print("No run records found in the log.", highlight=False)
        return 0

    total_runs = len(records)
    analysed = records[-1:] if args.latest else records

    agg_processed = sum(entry.get("stats", {}).get("processed", 0) for entry in analysed)
    agg_changed = sum(entry.get("stats", {}).get("changed", 0) for entry in analysed)
    agg_skipped = sum(entry.get("stats", {}).get("skipped", 0) for entry in analysed)
    agg_failed = sum(entry.get("stats", {}).get("failed", 0) for entry in analysed)
    agg_unlocked = sum(entry.get("stats", {}).get("unlocked", 0) for entry in analysed)
    durations = [float(entry.get("duration", 0.0)) for entry in analysed]

    console.print(f"[bold cyan]Run log summary[/bold cyan] {log_path}", highlight=False)
    console.print(
        f"Runs analysed: {len(analysed)} (of {total_runs} total)", highlight=False
    )
    console.print(
        f"Processed: {agg_processed}, Changed: {agg_changed}, Skipped: {agg_skipped}, "
        f"Failed: {agg_failed}, Unlocked: {agg_unlocked}",
        highlight=False,
    )
    if durations:
        total_duration = sum(durations)
        average_duration = total_duration / len(durations)
        console.print(
            f"Duration - total: {total_duration:.2f}s, average: {average_duration:.2f}s",
            highlight=False,
        )
        console.print(
            f"Latest run completed at: {analysed[-1].get('completed_at', 'unknown')}",
            highlight=False,
        )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Execute the CLI entry point."""
    parser = build_parser()
    raw_args = list(argv if argv is not None else sys.argv[1:])
    known_commands = {"process", "dry-run", "rules", "print-config", "stats"}
    if not raw_args:
        parser.print_help()
        return 0
    if raw_args[0] not in known_commands and not raw_args[0].startswith("-"):
        raw_args = ["process", *raw_args]

    args = parser.parse_args(raw_args)
    _configure_logging(args.log_level)

    if args.command == "process":
        return _run_process_command(args, dry_run=False)
    if args.command == "dry-run":
        return _run_process_command(args, dry_run=True)
    if args.command == "rules":
        return _run_rules_command(args)
    if args.command == "print-config":
        return _run_print_config_command(args)
    if args.command == "stats":
        return _run_stats_command(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
