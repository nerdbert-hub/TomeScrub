"""Core cleaning logic leveraging PyMuPDF."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import logging
import os
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


def _safe_process_count(value: int | str) -> int:
    """Resolve configured process counts into a usable integer."""
    if isinstance(value, str):
        return max(1, os.cpu_count() or 1)
    return max(1, int(value))


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
        ) or self.was_encrypted or (self.original_permissions != self.cleaned_permissions)


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
        strip_document_metadata: Optional[bool] = None,
        strip_image_metadata: Optional[bool] = None,
        remove_hidden_text: Optional[bool] = None,
        extract_text: Optional[bool] = None,
        overwrite: Optional[bool] = None,
        skip_unchanged: Optional[bool] = None,
        password_provider: Optional[PasswordProvider] = None,
        password_resolver: Optional[PasswordResolver] = None,
        watermark_rules: Optional[Sequence[WatermarkRule]] = None,
        hidden_text_alpha_threshold: Optional[int] = None,
        watermarks_enabled: Optional[bool] = None,
        watermarks_clip_bottom_mm: Optional[float] = None,
        watermarks_stop_after_first: Optional[bool] = None,
        watermarks_max_pages: Optional[int] = None,
        save_linearize: Optional[bool] = None,
        save_garbage: Optional[int] = None,
        save_deflate: Optional[bool] = None,
        processes: Optional[int] = None,
        batch_size: Optional[int] = None,
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
            watermarks_enabled: Enable or disable watermark detection/removal.
            watermarks_clip_bottom_mm: Restrict watermark detection to the bottom band (millimetres).
            watermarks_stop_after_first: Halt detection once a rule matches on a page.
            watermarks_max_pages: Limit the number of pages scanned for watermarks (0 = all).
            save_linearize: Optimise PDF for web view during save.
            save_garbage: MuPDF garbage collection level (0-4) when saving.
            save_deflate: Compress object streams during save.
            processes: Preferred worker count for parallel processing (used by CLI).
            batch_size: Chunk size for dispatching parallel tasks.
        """
        if output_dir is None:
            output_dir = self.DEFAULT_CONFIG.io.output_dir
        if overwrite is None:
            overwrite = self.DEFAULT_CONFIG.io.overwrite_existing
        if hidden_text_alpha_threshold is None:
            hidden_text_alpha_threshold = self.DEFAULT_CONFIG.clean.hidden_text_alpha_threshold
        if strip_document_metadata is None:
            strip_document_metadata = (
                sanitize_metadata
                if sanitize_metadata is not None
                else self.DEFAULT_CONFIG.clean.strip_document_metadata
            )
        if strip_image_metadata is None:
            strip_image_metadata = (
                sanitize_metadata
                if sanitize_metadata is not None
                else self.DEFAULT_CONFIG.clean.strip_image_metadata
            )
        if remove_hidden_text is None:
            remove_hidden_text = self.DEFAULT_CONFIG.clean.remove_hidden_text
        if extract_text is None:
            extract_text = self.DEFAULT_CONFIG.clean.extract_text
        if skip_unchanged is None:
            skip_unchanged = self.DEFAULT_CONFIG.io.skip_unchanged
        if watermarks_enabled is None:
            watermarks_enabled = self.DEFAULT_CONFIG.watermarks.enabled
        if watermarks_clip_bottom_mm is None:
            watermarks_clip_bottom_mm = self.DEFAULT_CONFIG.watermarks.clip_bottom_mm
        if watermarks_stop_after_first is None:
            watermarks_stop_after_first = self.DEFAULT_CONFIG.watermarks.stop_after_first
        if watermarks_max_pages is None:
            watermarks_max_pages = self.DEFAULT_CONFIG.watermarks.max_pages
        if save_linearize is None:
            save_linearize = self.DEFAULT_CONFIG.save.linearize
        if save_garbage is None:
            save_garbage = self.DEFAULT_CONFIG.save.garbage
        if save_deflate is None:
            save_deflate = self.DEFAULT_CONFIG.save.deflate
        if processes is None:
            processes = _safe_process_count(self.DEFAULT_CONFIG.performance.processes)
        else:
            processes = max(1, processes)
        if batch_size is None:
            batch_size = self.DEFAULT_CONFIG.performance.batch_size
        else:
            batch_size = max(1, batch_size)

        if password_provider and password_resolver:
            raise ValueError("Provide either password_provider or password_resolver, not both.")

        if password_resolver is not None:
            password_provider = _CallablePasswordProvider(password_resolver)

        self.output_dir = Path(output_dir)
        self.strip_document_metadata = bool(strip_document_metadata)
        self.strip_image_metadata = bool(strip_image_metadata)
        self.remove_hidden_text = bool(remove_hidden_text)
        self.extract_text = bool(extract_text)
        self.overwrite = overwrite
        self.skip_unchanged = bool(skip_unchanged)
        self.password_provider = password_provider
        base_rules = list(watermark_rules) if watermark_rules is not None else self.DEFAULT_CONFIG.compile_watermark_rules()
        self.watermark_rules = base_rules if watermarks_enabled else []
        self.watermarks_enabled = bool(watermarks_enabled)
        self.watermarks_clip_bottom_mm = watermarks_clip_bottom_mm
        self.watermarks_clip_height_pts = (
            None
            if watermarks_clip_bottom_mm in (None, 0)
            else float(watermarks_clip_bottom_mm) * 72.0 / 25.4
        )
        self.watermarks_stop_after_first = bool(watermarks_stop_after_first)
        self.watermarks_max_pages = int(watermarks_max_pages or 0)
        self.hidden_text_alpha_threshold = hidden_text_alpha_threshold
        self.save_linearize = bool(save_linearize)
        self.save_garbage = save_garbage
        self.save_deflate = bool(save_deflate)
        self.processes = processes
        self.batch_size = batch_size
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
            strip_document_metadata=config.clean.strip_document_metadata,
            strip_image_metadata=config.clean.strip_image_metadata,
            remove_hidden_text=config.clean.remove_hidden_text,
            extract_text=config.clean.extract_text,
            overwrite=config.io.overwrite_existing,
            skip_unchanged=config.io.skip_unchanged,
            password_provider=password_provider,
            password_resolver=password_resolver,
            watermark_rules=config.compile_watermark_rules(),
            hidden_text_alpha_threshold=config.clean.hidden_text_alpha_threshold,
            watermarks_enabled=config.watermarks.enabled,
            watermarks_clip_bottom_mm=config.watermarks.clip_bottom_mm,
            watermarks_stop_after_first=config.watermarks.stop_after_first,
            watermarks_max_pages=config.watermarks.max_pages,
            save_linearize=config.save.linearize,
            save_garbage=config.save.garbage,
            save_deflate=config.save.deflate,
            processes=_safe_process_count(config.performance.processes),
            batch_size=config.performance.batch_size,
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
            image_xrefs: list[int] = [] if self.strip_image_metadata else []
            document_metadata_cleared = False
            page_count = document.page_count

            if self.strip_document_metadata:
                step_start = perf_counter()
                document_metadata_cleared = clear_document_metadata(document)
                timings["clear_document_metadata"] = perf_counter() - step_start
            else:
                timings["clear_document_metadata"] = 0.0

            watermark_page_indices: Optional[set[int]] = None
            if self.watermarks_enabled and self.watermarks_max_pages > 0:
                count = min(self.watermarks_max_pages, page_count)
                front = set(range(count))
                back = set(range(max(0, page_count - count), page_count))
                watermark_page_indices = front | back

            step_start = perf_counter()
            for index, page in enumerate(document):
                if self.strip_image_metadata:
                    image_xrefs.extend(xref for xref, *_ in page.get_images(full=True))

                if self.remove_hidden_text:
                    hidden_text_removed += remove_hidden_text(
                        page,
                        alpha_threshold=self.hidden_text_alpha_threshold,
                    )

                if self.watermarks_enabled and self.watermark_rules:
                    if watermark_page_indices is None or index in watermark_page_indices:
                        clip_rect = None
                        if self.watermarks_clip_height_pts:
                            height = min(self.watermarks_clip_height_pts, float(page.rect.height))
                            y0 = max(float(page.rect.y1) - height, float(page.rect.y0))
                            clip_rect = fitz.Rect(page.rect.x0, y0, page.rect.x1, page.rect.y1)
                        matches = remove_watermarks(
                            page,
                            rules=self.watermark_rules,
                            clip_rect=clip_rect,
                            stop_after_first=self.watermarks_stop_after_first,
                        )
                        watermarks_removed += len(matches)

                if self.extract_text:
                    text_chunks.append(page.get_text())
            timings["process_pages"] = perf_counter() - step_start

            if self.strip_image_metadata and image_xrefs:
                step_start = perf_counter()
                image_metadata_cleared = clear_image_metadata(document, image_xrefs)
                timings["clear_image_metadata"] = perf_counter() - step_start
            else:
                image_metadata_cleared = 0
                timings["clear_image_metadata"] = 0.0

            result_changed = any(
                (
                    document_metadata_cleared,
                    image_metadata_cleared,
                    hidden_text_removed,
                    watermarks_removed,
                )
            ) or was_encrypted
            should_save = not (self.skip_unchanged and not result_changed)

            if should_save:
                step_start = perf_counter()
                document.save(
                    output_path,
                    encryption=fitz.PDF_ENCRYPT_NONE,
                    garbage=self.save_garbage,
                    deflate=self.save_deflate,
                    linear=self.save_linearize,
                )
                timings["save"] = perf_counter() - step_start
            else:
                timings["save"] = 0.0

        if should_save:
            step_start = perf_counter()
            with fitz.open(output_path) as cleaned:
                cleaned_permissions = cleaned.permissions
            timings["verify"] = perf_counter() - step_start
        else:
            if not output_path.exists():
                shutil.copy2(pdf_path, output_path)
            cleaned_permissions = original_permissions
            timings["verify"] = 0.0

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
            text="".join(text_chunks) if self.extract_text else "",
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
