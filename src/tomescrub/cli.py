"""Command-line interface for the PDF cleaner tool."""

from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path
from time import perf_counter
from typing import Iterable, Optional, Sequence

from rich.console import Console

from .config import load_config
from .passwords import PasswordProvider, load_password_file
from .processor import PDFCleaner, ProcessingEvent, RunStatistics

FILE_COLUMN_WIDTH = 48
BAR_WIDTH = 20
COUNT_COLUMN_WIDTH = 7



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
    return overrides


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


def _build_bar(completed: int, total: Optional[int], width: int = BAR_WIDTH) -> str:
    if not total or total <= 0:
        return "-" * width
    ratio = max(0.0, min(1.0, completed / total))
    filled = int(ratio * width)
    return "█" * filled + "─" * (width - filled)


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


def _print_legend(console: Console, total: Optional[int]) -> None:
    """Display a legend explaining each column in the progress output."""
    count_example = "count" if not total else f"0/{total}"
    header = (
        f"Format: spin | action            | progress{' ' * (BAR_WIDTH - 8)}| "
        f"{count_example.rjust(COUNT_COLUMN_WIDTH)} | elapsed  | eta      | step   | pages       | file"
    )
    detail = (
        "        life | current step      | █ done, ─ remaining     | "
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

    pdf_candidates = _discover_pdf_files(args.source)
    total_pdfs = len(pdf_candidates)

    source_root = args.source.resolve()
    prefer_name = source_root.is_file()
    display_base = source_root if source_root.is_dir() else source_root.parent

    console.print(
        f"[bold cyan]Cleaning[/bold cyan] {args.source} -> {cleaner.output_dir.resolve()}",
        highlight=False,
    )
    _print_legend(console, total_pdfs)

    spinner_cycle = itertools.cycle(["|", "/", "-", "\\"])
    start_time = perf_counter()
    completed = 0

    with console.status("Preparing run...", spinner="dots") as status:

        def handle_event(event: ProcessingEvent) -> None:
            nonlocal completed
            if event.kind in {"processed", "skipped", "failed"}:
                completed += 1

            spinner = next(spinner_cycle)
            label = _label_for_event(event)
            bar = _build_bar(completed, total_pdfs)
            count_str = _format_count(completed, total_pdfs)
            elapsed_total = perf_counter() - start_time
            elapsed_str = _format_duration(elapsed_total)
            eta_str = _format_eta(elapsed_total, completed, total_pdfs)

            if event.kind == "processed" and event.result:
                step_time = event.result.elapsed
                pages = event.result.page_count
            elif event.kind == "failed":
                step_time = None
                pages = None
            elif event.kind == "copied":
                step_time = None
                pages = None
            else:
                step_time = None
                pages = None

            pages_str = _format_pages(pages) if event.kind != "copied" else "   asset   "
            filename = _truncate_middle(
                _relative_path_for_display(event.source, display_base, prefer_name=prefer_name),
                FILE_COLUMN_WIDTH,
            )
            line = (
                f"{spinner} {label:<17} | {bar} | {count_str:>{COUNT_COLUMN_WIDTH}} | {elapsed_str} | "
                f"{eta_str} | {_format_step_time(step_time)} | {pages_str:<11} | {filename}"
            )
            console.print(line, highlight=False)

            status.update(f"{label} {filename}")

            if event.kind == "failed":
                details = f"{event.error}" if event.error else "Unknown error"
                console.print(
                    f"    -> [bold red]{details}[/bold red]",
                    highlight=False,
                )

        for _ in cleaner.process_path(args.source, progress_callback=handle_event):
            pass
        status.update("Processing complete")

    stats = cleaner.last_run_stats or RunStatistics()
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
