"""Core cleaning logic leveraging PyMuPDF."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Callable, Iterator, Optional, Sequence

import fitz  # PyMuPDF

from .passwords import PasswordProvider
from .sanitizer import (
    clear_document_metadata,
    clear_image_metadata,
    remove_hidden_text,
)
from .watermarks import WatermarkRule, remove_watermarks, DEFAULT_WATERMARK_RULES

PasswordResolver = Callable[[Path], Optional[str]]


@dataclass(frozen=True)
class DocumentProcessingResult:
    """Contain metadata about a processed PDF."""

    source: Path
    output: Path
    text: str
    was_encrypted: bool
    original_permissions: int
    cleaned_permissions: int
    watermarks_removed: int
    hidden_text_removed: int
    image_metadata_cleared: int
    document_metadata_cleared: bool


class PasswordAuthenticationError(RuntimeError):
    """Raised when a password is required but unavailable or incorrect."""


class _CallablePasswordProvider:
    """Adapter enabling callables to behave like a PasswordProvider."""

    def __init__(self, resolver: PasswordResolver) -> None:
        self._resolver = resolver

    def resolve(self, pdf_path: Path) -> Optional[str]:  # pragma: no cover - trivial
        return self._resolver(pdf_path)


class PDFCleaner:
    """Provide utilities for sanitising and saving cleaned PDF documents."""

    def __init__(
        self,
        output_dir: Path,
        *,
        sanitize_metadata: bool = True,
        overwrite: bool = True,
        password_provider: Optional[PasswordProvider] = None,
        password_resolver: Optional[PasswordResolver] = None,
        watermark_rules: Optional[Sequence[WatermarkRule]] = None,
    ) -> None:
        """
        Create a new cleaner.

        Args:
            output_dir: Directory where cleaned PDF files will be written.
            sanitize_metadata: When True, strip document metadata during cleaning.
            overwrite: When False, skip PDFs whose processed version already exists.
            password_provider: Object capable of providing passwords for encrypted PDFs.
            password_resolver: Callable variant of ``password_provider``.
            watermark_rules: Optional override for default watermark removal rules.
        """
        if password_provider and password_resolver:
            raise ValueError("Provide either password_provider or password_resolver, not both.")

        if password_resolver is not None:
            password_provider = _CallablePasswordProvider(password_resolver)

        self.output_dir = output_dir
        self.sanitize_metadata = sanitize_metadata
        self.overwrite = overwrite
        self.password_provider = password_provider
        self.watermark_rules = watermark_rules or DEFAULT_WATERMARK_RULES
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_path(
        self,
        source: Path,
        *,
        source_root: Optional[Path] = None,
    ) -> Iterator[DocumentProcessingResult]:
        """
        Recursively process a file or directory, mirroring structure in the output dir.

        Args:
            source: File or directory scheduled for processing.
            source_root: Base directory used to calculate relative output paths. When
                omitted, the immediate parent of ``source`` (for files) or ``source``
                itself (for directories) is used.

        Yields:
            A :class:`DocumentProcessingResult` for every PDF encountered.
        """
        source = source.resolve()
        if source_root is None:
            source_root = source if source.is_dir() else source.parent
        else:
            source_root = source_root.resolve()

        if source.is_dir():
            for child in sorted(source.iterdir()):
                yield from self.process_path(child, source_root=source_root)
            return

        relative_path = source.relative_to(source_root)
        destination = self.output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)

        if source.suffix.lower() == ".pdf":
            if destination.exists() and not self.overwrite:
                return
            yield self.clean_document(source, output_path=destination)
        else:
            if destination.exists() and not self.overwrite:
                return
            shutil.copy2(source, destination)
            return

    def clean_document(
        self,
        pdf_path: Path,
        *,
        suffix: Optional[str] = None,
        output_path: Optional[Path] = None,
    ) -> DocumentProcessingResult:
        """
        Clean a PDF file and persist the processed version.

        Args:
            pdf_path: Path to the source PDF document.
            suffix: Optional suffix (before extension) for the cleaned file.
            output_path: Optional explicit output path. When omitted the file is saved
                directly under ``self.output_dir`` with ``suffix`` applied.

        Returns:
            Metadata about the processed document.

        Raises:
            FileNotFoundError: If the input file does not exist.
            PasswordAuthenticationError: If the document is encrypted and cannot be opened.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if output_path is None:
            output_path = self._build_output_path(pdf_path, suffix)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        with self._open_document(pdf_path) as (document, was_encrypted):
            original_permissions = document.permissions
            text_chunks: list[str] = []
            watermarks_removed = 0
            hidden_text_removed = 0
            image_xrefs: list[int] = []
            document_metadata_cleared = False

            if self.sanitize_metadata:
                document_metadata_cleared = clear_document_metadata(document)

            for page in document:
                image_xrefs.extend(xref for xref, *_ in page.get_images(full=True))
                hidden_text_removed += remove_hidden_text(page)
                matches = remove_watermarks(page, rules=self.watermark_rules)
                watermarks_removed += len(matches)
                text_chunks.append(page.get_text())

            if self.sanitize_metadata:
                image_metadata_cleared = clear_image_metadata(document, image_xrefs)
            else:
                image_metadata_cleared = 0

            document.save(
                output_path,
                encryption=fitz.PDF_ENCRYPT_NONE,
                garbage=4,
                deflate=True,
            )

        with fitz.open(output_path) as cleaned:
            cleaned_permissions = cleaned.permissions

        return DocumentProcessingResult(
            source=pdf_path,
            output=output_path,
            text="".join(text_chunks),
            was_encrypted=was_encrypted,
            original_permissions=original_permissions,
            cleaned_permissions=cleaned_permissions,
            watermarks_removed=watermarks_removed,
            hidden_text_removed=hidden_text_removed,
            image_metadata_cleared=image_metadata_cleared,
            document_metadata_cleared=document_metadata_cleared,
        )

    @contextmanager
    def _open_document(self, pdf_path: Path) -> Iterator[tuple[fitz.Document, bool]]:
        document = fitz.open(pdf_path)
        try:
            needs_pass_initial = document.needs_pass
            is_encrypted_initial = document.is_encrypted
            if needs_pass_initial:
                password = self._resolve_password(pdf_path)
                if password is None:
                    raise PasswordAuthenticationError(f"Password required for encrypted PDF: {pdf_path}")
                if document.authenticate(password) == 0:
                    raise PasswordAuthenticationError(f"Failed to unlock PDF with provided password: {pdf_path}")
            was_encrypted = bool(needs_pass_initial or is_encrypted_initial)
            yield document, was_encrypted
        finally:
            document.close()

    def _resolve_password(self, pdf_path: Path) -> Optional[str]:
        if self.password_provider is None:
            return None
        return self.password_provider.resolve(pdf_path)

    def _build_output_path(self, pdf_path: Path, suffix: Optional[str]) -> Path:
        target_name = pdf_path.stem
        if suffix:
            target_name = f"{target_name}_{suffix}"
        return self.output_dir / f"{target_name}.pdf"
