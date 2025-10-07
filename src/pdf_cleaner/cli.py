"""Command-line interface for the PDF cleaner tool."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

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
        "-o",
        "--output-dir",
        type=Path,
        default=Path("_processed"),
        help="Directory where processed files will be stored (default: ./_processed).",
    )
    parser.add_argument(
        "--keep-metadata",
        action="store_true",
        help="Retain original metadata instead of stripping it.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip PDFs when the target file already exists.",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=None,
        help="Default password applied to encrypted PDFs when no specific match is found.",
    )
    parser.add_argument(
        "--password-file",
        type=Path,
        default=None,
        help="File containing PDF-specific passwords (format: name=password).",
    )
    parser.add_argument(
        "--password-hints",
        type=str,
        default="passwords.txt",
        help="Filename searched within each directory for PDF passwords (default: passwords.txt).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Execute the CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    password_hint_name = args.password_hints
    if password_hint_name and password_hint_name.lower() in {"none", "null"}:
        password_hint_name = None

    password_provider = PasswordProvider(
        default=args.password,
        global_mapping=load_password_file(args.password_file) if args.password_file else None,
        hint_filename=password_hint_name,
    )

    cleaner = PDFCleaner(
        output_dir=args.output_dir,
        sanitize_metadata=not args.keep_metadata,
        overwrite=not args.skip_existing,
        password_provider=password_provider,
    )
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
    summary = f"Processed {count} PDF file(s) into {args.output_dir.resolve()}."
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
