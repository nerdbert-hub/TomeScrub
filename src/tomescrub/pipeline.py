"""Pipeline stages and context objects for PDF cleaning."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
from time import perf_counter
from typing import Dict, List, Optional, TYPE_CHECKING

import fitz  # PyMuPDF

from .sanitizer import (
    clear_document_metadata,
    clear_image_metadata,
    remove_hidden_text,
)
from .watermarks import remove_watermarks

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from .processor import PDFCleaner


STAGE_ORDER = [
    "open",
    "detect_password",
    "sanitize_metadata",
    "remove_hidden_text",
    "watermark_scan",
    "save",
    "extract_text",
]


def _record_timing(timings: Dict[str, float], key: str, elapsed: float) -> None:
    timings[key] = timings.get(key, 0.0) + elapsed


@dataclass
class PipelineContext:
    """Mutable state threaded through each pipeline stage."""

    pdf_path: Path
    output_path: Path
    document: Optional[fitz.Document] = None
    was_encrypted: bool = False
    original_permissions: int = 0
    cleaned_permissions: int = 0
    page_count: int = 0
    document_metadata_cleared: bool = False
    image_metadata_cleared: int = 0
    hidden_text_removed: int = 0
    watermarks_removed: int = 0
    text_chunks: List[str] = field(default_factory=list)
    image_xrefs: List[int] = field(default_factory=list)
    stage_timings: Dict[str, float] = field(default_factory=dict)
    result_changed: bool = False
    should_save: bool = True
    original_size_bytes: int = 0
    output_size_bytes: int = 0


class PipelineStage:
    """Base class for sequential PDF processing stages."""

    name: str

    def run(self, cleaner: "PDFCleaner", ctx: PipelineContext) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class OpenDocumentStage(PipelineStage):
    """Load the source PDF into memory."""

    name = "open"

    def run(self, cleaner: "PDFCleaner", ctx: PipelineContext) -> None:
        start = perf_counter()
        ctx.document = fitz.open(ctx.pdf_path)
        _record_timing(ctx.stage_timings, self.name, perf_counter() - start)


class DetectPasswordStage(PipelineStage):
    """Resolve and authenticate any required password for the document."""

    name = "detect_password"

    def run(self, cleaner: "PDFCleaner", ctx: PipelineContext) -> None:
        document = ctx.document
        if document is None:
            raise RuntimeError("PDF document not opened before password detection stage.")
        start = perf_counter()
        needs_pass_initial = document.needs_pass
        is_encrypted_initial = document.is_encrypted
        if needs_pass_initial:
            password = cleaner._resolve_password(ctx.pdf_path)
            if password is None:
                from .processor import PasswordAuthenticationError  # noqa: PLC0415

                raise PasswordAuthenticationError(f"Password required for encrypted PDF: {ctx.pdf_path}")
            if document.authenticate(password) == 0:
                from .processor import PasswordAuthenticationError  # noqa: PLC0415

                raise PasswordAuthenticationError(f"Failed to unlock PDF with provided password: {ctx.pdf_path}")
        ctx.was_encrypted = bool(needs_pass_initial or is_encrypted_initial)
        ctx.page_count = document.page_count
        ctx.original_permissions = document.permissions
        _record_timing(ctx.stage_timings, self.name, perf_counter() - start)


class SanitizeMetadataStage(PipelineStage):
    """Clear document-level metadata and defer image metadata stripping."""

    name = "sanitize_metadata"

    def run(self, cleaner: "PDFCleaner", ctx: PipelineContext) -> None:
        document = ctx.document
        if document is None:
            raise RuntimeError("PDF document not opened before metadata sanitisation stage.")
        start = perf_counter()
        if cleaner.strip_document_metadata:
            ctx.document_metadata_cleared = clear_document_metadata(document)
        else:
            ctx.document_metadata_cleared = False
        _record_timing(ctx.stage_timings, self.name, perf_counter() - start)


class ProcessPagesStage(PipelineStage):
    """Iterate through pages to apply hidden-text removal, watermark scans, and extraction."""

    name = "process_pages"

    def run(self, cleaner: "PDFCleaner", ctx: PipelineContext) -> None:
        document = ctx.document
        if document is None:
            raise RuntimeError("PDF document not opened before page processing stage.")

        hidden_text_time = 0.0
        watermark_time = 0.0
        extract_time = 0.0
        hidden_text_removed = 0
        watermarks_removed = 0
        image_xrefs: List[int] = []
        text_chunks: List[str] = []

        watermark_page_indices: Optional[set[int]] = None
        if cleaner.watermarks_enabled and cleaner.watermarks_max_pages > 0 and ctx.page_count:
            count = min(cleaner.watermarks_max_pages, ctx.page_count)
            front = set(range(count))
            back = set(range(max(0, ctx.page_count - count), ctx.page_count))
            watermark_page_indices = front | back

        for index, page in enumerate(document):
            if cleaner.strip_image_metadata:
                image_xrefs.extend(xref for xref, *_ in page.get_images(full=True))

            if cleaner.remove_hidden_text:
                stage_start = perf_counter()
                hidden_text_removed += remove_hidden_text(
                    page,
                    alpha_threshold=cleaner.hidden_text_alpha_threshold,
                )
                hidden_text_time += perf_counter() - stage_start

            if cleaner.watermarks_enabled and cleaner.watermark_rules:
                if watermark_page_indices is None or index in watermark_page_indices:
                    clip_rect = None
                    if (
                        cleaner.watermarks_scan_mode == "bottom"
                        and cleaner.watermarks_clip_height_pts
                    ):
                        height = min(cleaner.watermarks_clip_height_pts, float(page.rect.height))
                        y0 = max(float(page.rect.y1) - height, float(page.rect.y0))
                        clip_rect = fitz.Rect(page.rect.x0, y0, page.rect.x1, page.rect.y1)
                    stage_start = perf_counter()
                    matches = remove_watermarks(
                        page,
                        rules=cleaner.watermark_rules,
                        clip_rect=clip_rect,
                        stop_after_first=cleaner.watermarks_stop_after_first,
                    )
                    watermarks_removed += len(matches)
                    watermark_time += perf_counter() - stage_start

            if cleaner.extract_text:
                stage_start = perf_counter()
                text_chunks.append(page.get_text())
                extract_time += perf_counter() - stage_start

        ctx.hidden_text_removed = hidden_text_removed
        ctx.watermarks_removed = watermarks_removed
        ctx.image_xrefs = image_xrefs
        ctx.text_chunks = text_chunks

        _record_timing(ctx.stage_timings, "remove_hidden_text", hidden_text_time)
        _record_timing(ctx.stage_timings, "watermark_scan", watermark_time)
        if cleaner.extract_text:
            _record_timing(ctx.stage_timings, "extract_text", extract_time)
        else:
            ctx.stage_timings.setdefault("extract_text", 0.0)


class ClearImageMetadataStage(PipelineStage):
    """Strip metadata embedded within image objects, if requested."""

    name = "clear_image_metadata"

    def run(self, cleaner: "PDFCleaner", ctx: PipelineContext) -> None:
        document = ctx.document
        if document is None:
            raise RuntimeError("PDF document not opened before image metadata stage.")
        if not cleaner.strip_image_metadata or not ctx.image_xrefs:
            ctx.image_metadata_cleared = 0
            return
        start = perf_counter()
        ctx.image_metadata_cleared = clear_image_metadata(document, ctx.image_xrefs)
        _record_timing(ctx.stage_timings, "sanitize_metadata", perf_counter() - start)


class SaveDocumentStage(PipelineStage):
    """Persist the processed PDF (and verify permissions) based on configuration."""

    name = "save"

    def run(self, cleaner: "PDFCleaner", ctx: PipelineContext) -> None:
        document = ctx.document
        if document is None:
            raise RuntimeError("PDF document not opened before save stage.")

        start = perf_counter()
        result_changed = any(
            (
                ctx.document_metadata_cleared,
                ctx.image_metadata_cleared,
                ctx.hidden_text_removed,
                ctx.watermarks_removed,
            )
        ) or ctx.was_encrypted

        ctx.result_changed = result_changed
        ctx.should_save = not cleaner.dry_run and not (cleaner.skip_unchanged and not result_changed)

        if cleaner.dry_run:
            ctx.cleaned_permissions = ctx.original_permissions
            ctx.stage_timings["save"] = 0.0
            ctx.stage_timings["verify"] = 0.0
            return

        if ctx.should_save:
            outcome = cleaner.save_backend.save(cleaner, ctx, document)
            total_elapsed = perf_counter() - start
            save_time = max(total_elapsed - outcome.verify_time, 0.0)
            ctx.stage_timings["save"] = save_time
            ctx.stage_timings["verify"] = outcome.verify_time
            ctx.cleaned_permissions = outcome.cleaned_permissions
        else:
            if not ctx.output_path.exists():
                ctx.output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ctx.pdf_path, ctx.output_path)
            ctx.cleaned_permissions = ctx.original_permissions
            ctx.stage_timings["save"] = 0.0
            ctx.stage_timings["verify"] = 0.0


__all__ = [
    "PipelineContext",
    "PipelineStage",
    "OpenDocumentStage",
    "DetectPasswordStage",
    "SanitizeMetadataStage",
    "ProcessPagesStage",
    "ClearImageMetadataStage",
    "SaveDocumentStage",
    "STAGE_ORDER",
]
