"""Core cleaning logic leveraging PyMuPDF."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import logging
import shutil
from time import perf_counter
from typing import Callable, Iterator, Optional, Sequence

import fitz  # PyMuPDF

from .config import Config, load_defaults
from .passwords import PasswordProvider
from .sanitizer import (
    clear_document_metadata,
    clear_image_metadata,
    remove_hidden_text,
)
from .watermarks import WatermarkRule, remove_watermarks

PasswordResolver = Callable[[Path], Optional[str]]

try:  # pragma: no cover - varies across PyMuPDF versions
    FileDataError = fitz.FileDataError  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - fallback
    FileDataError = RuntimeError  # type: ignore[assignment]


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
    page_count: int
    elapsed: float
    step_timings: dict[str, float]

    @property
    def changed(self) -> bool:
        """Return True when any sanitisation altered the document."""
        return any(
            (
                self.watermarks_removed,
                self.hidden_text_removed,
                self.image_metadata_cleared,
                self.document_metadata_cleared,
            )
        )


@dataclass
class RunStatistics:
    """Aggregate metrics for a run of ``PDFCleaner.process_path``."""

    processed: int = 0
    skipped: int = 0
    copied: int = 0
    unlocked: int = 0
    changed: int = 0
    failed: int = 0
    total_pages: int = 0
    total_elapsed: float = 0.0
    total_watermarks_removed: int = 0
    total_hidden_text_removed: int = 0
    total_image_metadata_cleared: int = 0
    documents_metadata_cleared: int = 0
    files: list[tuple[Path, float]] = field(default_factory=list)
    failures: list[tuple[Path, str]] = field(default_factory=list)

    def add_result(self, result: DocumentProcessingResult) -> None:
        self.processed += 1
        self.total_pages += result.page_count
        self.total_watermarks_removed += result.watermarks_removed
        self.total_hidden_text_removed += result.hidden_text_removed
        self.total_image_metadata_cleared += result.image_metadata_cleared
        if result.document_metadata_cleared:
            self.documents_metadata_cleared += 1
        if result.was_encrypted:
            self.unlocked += 1
        if result.changed:
            self.changed += 1
        self.files.append((result.source, result.elapsed))

    def add_failure(self, source: Path, reason: str) -> None:
        self.failed += 1
        self.failures.append((source, reason))


@dataclass(frozen=True)
class ProcessingEvent:
    """Event emitted during processing for progress reporting."""

    source: Path
    kind: str
    result: Optional[DocumentProcessingResult] = None
    error: Optional[Exception] = None
    message: Optional[str] = None


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

    DEFAULT_CONFIG = Config.model_validate(load_defaults())

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        *,
        sanitize_metadata: Optional[bool] = None,
        overwrite: Optional[bool] = None,
        password_provider: Optional[PasswordProvider] = None,
        password_resolver: Optional[PasswordResolver] = None,
        watermark_rules: Optional[Sequence[WatermarkRule]] = None,
        hidden_text_alpha_threshold: Optional[int] = None,
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
            hidden_text_alpha_threshold: Alpha value (0-255) used to determine hidden text.
        """
        if output_dir is None:
            output_dir = self.DEFAULT_CONFIG.io.output_dir
        if sanitize_metadata is None:
            sanitize_metadata = self.DEFAULT_CONFIG.clean.sanitize_metadata
        if overwrite is None:
            overwrite = self.DEFAULT_CONFIG.io.overwrite_existing
        if hidden_text_alpha_threshold is None:
            hidden_text_alpha_threshold = self.DEFAULT_CONFIG.clean.hidden_text_alpha_threshold

        if password_provider and password_resolver:
            raise ValueError("Provide either password_provider or password_resolver, not both.")

        if password_resolver is not None:
            password_provider = _CallablePasswordProvider(password_resolver)

        self.output_dir = Path(output_dir)
        self.sanitize_metadata = sanitize_metadata
        self.overwrite = overwrite
        self.password_provider = password_provider
        self.watermark_rules = (
            list(watermark_rules)
            if watermark_rules is not None
            else self.DEFAULT_CONFIG.compile_watermark_rules()
        )
        self.hidden_text_alpha_threshold = hidden_text_alpha_threshold
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(__name__)
        self._current_stats: Optional[RunStatistics] = None
        self._last_stats: Optional[RunStatistics] = None
        self._run_started_at: Optional[float] = None

    @classmethod
    def from_config(
        cls,
        config: Config,
        *,
        password_provider: Optional[PasswordProvider] = None,
        password_resolver: Optional[PasswordResolver] = None,
    ) -> "PDFCleaner":
        """Construct a cleaner from a Config object."""
        return cls(
            output_dir=config.io.output_dir,
            sanitize_metadata=config.clean.sanitize_metadata,
            overwrite=config.io.overwrite_existing,
            password_provider=password_provider,
            password_resolver=password_resolver,
            watermark_rules=config.compile_watermark_rules(),
            hidden_text_alpha_threshold=config.clean.hidden_text_alpha_threshold,
        )

    def process_path(
        self,
        source: Path,
        *,
        source_root: Optional[Path] = None,
        progress_callback: Optional[Callable[[ProcessingEvent], None]] = None,
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
        is_root_invocation = source_root is None
        if is_root_invocation:
            source_root = source if source.is_dir() else source.parent
        if source_root is None:
            source_root = source.parent
        source_root = source_root.resolve()

        def _notify(
            kind: str,
            result: Optional[DocumentProcessingResult] = None,
            error: Optional[Exception] = None,
            message: Optional[str] = None,
        ) -> None:
            if progress_callback is None:
                return
            progress_callback(
                ProcessingEvent(
                    source=source,
                    kind=kind,
                    result=result,
                    error=error,
                    message=message,
                )
            )

        if is_root_invocation and self._current_stats is None:
            self._begin_run()

        try:
            if source.is_dir():
                for child in sorted(source.iterdir()):
                    yield from self.process_path(
                        child,
                        source_root=source_root,
                        progress_callback=progress_callback,
                    )
                return

            relative_path = source.relative_to(source_root)
            destination = self.output_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)

            if source.suffix.lower() == ".pdf":
                if destination.exists() and not self.overwrite:
                    self._logger.debug("Skipping processed PDF %s (already exists).", destination)
                    if self._current_stats:
                        self._current_stats.skipped += 1
                    _notify("skipped", message="exists")
                    return
                if source.name.startswith("._") or source.stat().st_size == 0:
                    self._logger.debug("Skipping likely resource fork or empty PDF %s.", source)
                    if self._current_stats:
                        self._current_stats.skipped += 1
                    _notify("skipped", message="resource-fork")
                    return
                try:
                    result = self.clean_document(source, output_path=destination)
                except FileDataError as exc:
                    self._logger.debug("Failed to process %s: %s", source, exc)
                    if self._current_stats:
                        reason = getattr(exc, "args", [""])[0] or str(exc)
                        self._current_stats.add_failure(source, reason)
                    _notify("failed", error=exc, message="open-error")
                    return
                if self._current_stats:
                    self._current_stats.add_result(result)
                status = "changed" if result.changed else "no-change"
                self._logger.debug(
                    "Processed %s in %.2fs (%d pages)",
                    source,
                    result.elapsed,
                    result.page_count,
                )
                _notify("processed", result, message=status)
                yield result
            else:
                if destination.exists() and not self.overwrite:
                    self._logger.debug("Skipping asset %s (already exists).", destination)
                    if self._current_stats:
                        self._current_stats.skipped += 1
                    _notify("skipped", message="asset-exists")
                    return
                shutil.copy2(source, destination)
                if self._current_stats:
                    self._current_stats.copied += 1
                self._logger.debug("Copied asset %s -> %s", source, destination)
                _notify("copied", message="asset")
        finally:
            if is_root_invocation:
                self._finalise_run()

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

        timings: dict[str, float] = {}
        overall_start = perf_counter()

        step_start = perf_counter()
        with self._open_document(pdf_path) as (document, was_encrypted):
            timings["open"] = perf_counter() - step_start
            original_permissions = document.permissions
            text_chunks: list[str] = []
            watermarks_removed = 0
            hidden_text_removed = 0
            image_xrefs: list[int] = []
            document_metadata_cleared = False
            page_count = document.page_count

            if self.sanitize_metadata:
                step_start = perf_counter()
                document_metadata_cleared = clear_document_metadata(document)
                timings["clear_document_metadata"] = perf_counter() - step_start
            else:
                timings["clear_document_metadata"] = 0.0

            step_start = perf_counter()
            for page in document:
                image_xrefs.extend(xref for xref, *_ in page.get_images(full=True))
                hidden_text_removed += remove_hidden_text(
                    page,
                    alpha_threshold=self.hidden_text_alpha_threshold,
                )
                matches = remove_watermarks(page, rules=self.watermark_rules)
                watermarks_removed += len(matches)
                text_chunks.append(page.get_text())
            timings["process_pages"] = perf_counter() - step_start

            if self.sanitize_metadata:
                step_start = perf_counter()
                image_metadata_cleared = clear_image_metadata(document, image_xrefs)
                timings["clear_image_metadata"] = perf_counter() - step_start
            else:
                image_metadata_cleared = 0
                timings["clear_image_metadata"] = 0.0

            step_start = perf_counter()
            document.save(
                output_path,
                encryption=fitz.PDF_ENCRYPT_NONE,
                garbage=4,
                deflate=True,
            )
            timings["save"] = perf_counter() - step_start

        step_start = perf_counter()
        with fitz.open(output_path) as cleaned:
            cleaned_permissions = cleaned.permissions
        timings["verify"] = perf_counter() - step_start

        elapsed = perf_counter() - overall_start
        timings["total"] = elapsed

        self._logger.debug(
            "Completed %s in %.3fs (%s)",
            pdf_path,
            elapsed,
            ", ".join(f"{name}={value:.3f}s" for name, value in timings.items()),
        )

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
            page_count=page_count,
            elapsed=elapsed,
            step_timings=timings,
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

    def _begin_run(self) -> None:
        self._logger.debug("Starting processing run into %s", self.output_dir)
        self._current_stats = RunStatistics()
        self._run_started_at = perf_counter()

    def _finalise_run(self) -> None:
        if self._current_stats is None:
            return
        if self._run_started_at is not None:
            self._current_stats.total_elapsed = perf_counter() - self._run_started_at
        self._last_stats = self._current_stats
        self._current_stats = None
        self._run_started_at = None

    @property
    def last_run_stats(self) -> Optional[RunStatistics]:
        """Return statistics collected during the most recent run."""
        return self._last_stats
