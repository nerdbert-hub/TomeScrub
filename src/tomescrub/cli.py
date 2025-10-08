"""Command-line interface for the PDF cleaner tool."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .config import load_config
from .passwords import PasswordProvider, load_password_file
from .processor import PDFCleaner


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

    overrides = _collect_overrides(args)
    config = load_config(args.config, overrides)

    password_provider = _build_password_provider(config)

    cleaner = PDFCleaner.from_config(config, password_provider=password_provider)
    count = 0
    unlocked = 0
    removed_watermarks = 0
    hidden_text_removed = 0
    docs_metadata_cleared = 0
    image_metadata_cleared = 0
    for result in cleaner.process_path(args.source):
        count += 1
        removed_watermarks += result.watermarks_removed
        hidden_text_removed += result.hidden_text_removed
        image_metadata_cleared += result.image_metadata_cleared
        if result.document_metadata_cleared:
            docs_metadata_cleared += 1
        if result.was_encrypted:
            unlocked += 1
    summary = f"Processed {count} PDF file(s) into {cleaner.output_dir.resolve()}."
    if unlocked:
        summary += f" Unlocked {unlocked} encrypted file(s)."
    if removed_watermarks:
        summary += f" Removed {removed_watermarks} watermark span(s)."
    if hidden_text_removed:
        summary += f" Removed {hidden_text_removed} hidden text span(s)."
    if docs_metadata_cleared:
        summary += f" Sanitised metadata in {docs_metadata_cleared} file(s)."
    if image_metadata_cleared:
        summary += f" Cleared metadata on {image_metadata_cleared} embedded image(s)."
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
