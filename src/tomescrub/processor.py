"""Core cleaning logic leveraging PyMuPDF."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import logging
import os
import shutil
from time import perf_counter
from typing import Callable, Optional, Sequence

import fitz  # PyMuPDF

from .config import Config, load_defaults
from .passwords import PasswordProvider
from .pipeline import (
    ClearImageMetadataStage,
    DetectPasswordStage,
    OpenDocumentStage,
    PipelineContext,
    ProcessPagesStage,
    SanitizeMetadataStage,
    SaveDocumentStage,
    STAGE_ORDER,
)
from .watermarks import WatermarkRule

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
    original_size_bytes: int
    output_size_bytes: int
    size_delta_bytes: int
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
        watermarks_scan_mode: Optional[str] = None,
        watermarks_clip_bottom_mm: Optional[float] = None,
        watermarks_stop_after_first: Optional[bool] = None,
        watermarks_max_pages: Optional[int] = None,
        save_linearize: Optional[bool] = None,
        save_garbage: Optional[int] = None,
        save_deflate: Optional[bool] = None,
        dry_run: bool = False,
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
            watermarks_scan_mode: Select between full-page scanning and bottom-band scanning.
            watermarks_clip_bottom_mm: Restrict watermark detection to the bottom band (millimetres).
            watermarks_stop_after_first: Halt detection once a rule matches on a page.
            watermarks_max_pages: Limit the number of pages scanned for watermarks (0 = all).
            save_linearize: Optimise PDF for web view during save.
            save_garbage: MuPDF garbage collection level (0-4) when saving.
            save_deflate: Compress object streams during save.
            processes: Preferred worker count for parallel processing (used by CLI).
            batch_size: Chunk size for dispatching parallel tasks.
            dry_run: When True, skip writing cleaned outputs (analysis only).
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
        if watermarks_scan_mode is None:
            watermarks_scan_mode = self.DEFAULT_CONFIG.watermarks.scan_mode
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
        self.dry_run = bool(dry_run)
        base_rules = list(watermark_rules) if watermark_rules is not None else self.DEFAULT_CONFIG.compile_watermark_rules()
        self.watermark_rules = base_rules if watermarks_enabled else []
        self.watermarks_enabled = bool(watermarks_enabled)
        scan_mode_normalized = str(watermarks_scan_mode).strip().lower()
        if scan_mode_normalized not in {"full", "bottom"}:
            raise ValueError('watermarks_scan_mode must be either "full" or "bottom"')
        self.watermarks_scan_mode = scan_mode_normalized
        self.watermarks_clip_bottom_mm = watermarks_clip_bottom_mm
        if self.watermarks_scan_mode == "bottom":
            self.watermarks_clip_height_pts = (
                None
                if watermarks_clip_bottom_mm in (None, 0)
                else float(watermarks_clip_bottom_mm) * 72.0 / 25.4
            )
        else:
            self.watermarks_clip_height_pts = None
        self.watermarks_stop_after_first = bool(watermarks_stop_after_first)
        self.watermarks_max_pages = int(watermarks_max_pages or 0)
        self.hidden_text_alpha_threshold = hidden_text_alpha_threshold
        self.save_linearize = bool(save_linearize)
        self.save_garbage = save_garbage
        self.save_deflate = bool(save_deflate)
        self.processes = processes
        self.batch_size = batch_size
        if not self.dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(__name__)
        self._current_stats: Optional[RunStatistics] = None
        self._last_stats: Optional[RunStatistics] = None
        self._run_started_at: Optional[float] = None
        self._stages = [
            OpenDocumentStage(),
            DetectPasswordStage(),
            SanitizeMetadataStage(),
            ProcessPagesStage(),
            ClearImageMetadataStage(),
            SaveDocumentStage(),
        ]

    @classmethod
    def from_config(
        cls,
        config: Config,
        *,
        password_provider: Optional[PasswordProvider] = None,
        password_resolver: Optional[PasswordResolver] = None,
        dry_run: bool = False,
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
            watermarks_scan_mode=config.watermarks.scan_mode,
            watermarks_clip_bottom_mm=config.watermarks.clip_bottom_mm,
            watermarks_stop_after_first=config.watermarks.stop_after_first,
            watermarks_max_pages=config.watermarks.max_pages,
            save_linearize=config.save.linearize,
            save_garbage=config.save.garbage,
            save_deflate=config.save.deflate,
            dry_run=dry_run,
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
            if not self.dry_run:
                destination.parent.mkdir(parents=True, exist_ok=True)

            if source.suffix.lower() == ".pdf":
                if not self.dry_run and destination.exists() and not self.overwrite:
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
                if not self.dry_run and destination.exists() and not self.overwrite:
                    self._logger.debug("Skipping asset %s (already exists).", destination)
                    if self._current_stats:
                        self._current_stats.skipped += 1
                    _notify("skipped", message="asset-exists")
                    return
                if self.dry_run:
                    self._logger.debug("Dry run: would copy asset %s -> %s", source, destination)
                    message = "asset-dry-run"
                else:
                    shutil.copy2(source, destination)
                    if self._current_stats:
                        self._current_stats.copied += 1
                    self._logger.debug("Copied asset %s -> %s", source, destination)
                    message = "asset"
                _notify("copied", message=message)
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
        if not self.dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        original_size = pdf_path.stat().st_size
        ctx = PipelineContext(pdf_path=pdf_path, output_path=output_path)
        ctx.stage_timings = {name: 0.0 for name in STAGE_ORDER}
        ctx.original_size_bytes = original_size

        overall_start = perf_counter()
        try:
            for stage in self._stages:
                stage.run(self, ctx)
        finally:
            if ctx.document is not None:
                ctx.document.close()

        total_elapsed = perf_counter() - overall_start
        ctx.stage_timings["total"] = total_elapsed

        if output_path.exists():
            output_size = output_path.stat().st_size
        else:
            output_size = original_size
        ctx.output_size_bytes = output_size
        size_delta = output_size - original_size

        text = "".join(ctx.text_chunks) if self.extract_text else ""

        self._logger.debug(
            "Completed %s in %.3fs (%s)",
            pdf_path,
            total_elapsed,
            ", ".join(
                f"{name}={ctx.stage_timings[name]:.3f}s"
                for name in [*STAGE_ORDER, "total"]
                if name in ctx.stage_timings
            ),
        )

        return DocumentProcessingResult(
            source=pdf_path,
            output=output_path,
            text=text,
            was_encrypted=ctx.was_encrypted,
            original_permissions=ctx.original_permissions,
            cleaned_permissions=ctx.cleaned_permissions,
            watermarks_removed=ctx.watermarks_removed,
            hidden_text_removed=ctx.hidden_text_removed,
            image_metadata_cleared=ctx.image_metadata_cleared,
            document_metadata_cleared=ctx.document_metadata_cleared,
            page_count=ctx.page_count,
            original_size_bytes=original_size,
            output_size_bytes=output_size,
            size_delta_bytes=size_delta,
            elapsed=total_elapsed,
            step_timings=ctx.stage_timings,
        )

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
