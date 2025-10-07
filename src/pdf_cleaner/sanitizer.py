"""Utilities for sanitising PDF documents (metadata, hidden content)."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Set

import fitz  # type: ignore


def clear_document_metadata(document: fitz.Document) -> bool:
    """
    Remove document-level metadata (Info dictionary + XML metadata).

    Returns:
        True when metadata was removed, False when nothing was present.
    """
    removed = False
    if document.metadata:
        document.set_metadata({})
        removed = True

    get_xml = getattr(document, "get_xml_metadata", None)
    del_xml = getattr(document, "del_xml_metadata", None)
    if callable(get_xml) and callable(del_xml):
        xml_metadata = get_xml()
        if xml_metadata:
            del_xml()
            removed = True

    return removed


def clear_image_metadata(document: fitz.Document, image_xrefs: Iterable[int]) -> int:
    """
    Strip metadata streams from embedded images.

    Args:
        document: The PDF document being processed.
        image_xrefs: Iterable of image object numbers encountered in the document.

    Returns:
        Number of image objects whose metadata was removed.
    """
    removed = 0
    seen: Set[int] = set()
    for xref in image_xrefs:
        if xref in seen or xref <= 0:
            continue
        seen.add(xref)
        try:
            key_type, key_value = document.xref_get_key(xref, "Metadata")
        except RuntimeError:
            continue
        if key_type == "null" or key_value in (None, "null"):
            continue
        document.xref_set_key(xref, "Metadata", "null")
        removed += 1
    return removed


def _collect_hidden_text_rects(page: fitz.Page, alpha_threshold: int = 0) -> List[fitz.Rect]:
    rects: List[fitz.Rect] = []
    page_dict = page.get_text("dict", sort=True)
    for block in page_dict.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            bbox = line.get("bbox")
            if not bbox:
                continue
            rect = fitz.Rect(bbox)
            for span in line.get("spans", []):
                alpha = span.get("alpha", 255)
                text = span.get("text", "")
                if not text.strip():
                    continue
                if alpha <= alpha_threshold:
                    rects.append(rect)
                    break
    return rects


def remove_hidden_text(page: fitz.Page, alpha_threshold: int = 0) -> int:
    """
    Remove hidden text spans from the page.

    Args:
        page: Page to sanitize.
        alpha_threshold: Maximum alpha regarded as hidden (default: fully transparent).

    Returns:
        Number of spans removed.
    """
    rects = _collect_hidden_text_rects(page, alpha_threshold=alpha_threshold)
    if not rects:
        return 0

    for rect in rects:
        expanded = fitz.Rect(rect)
        expanded.x0 -= 0.5
        expanded.x1 += 0.5
        expanded.y0 -= 0.5
        expanded.y1 += 0.5
        page.add_redact_annot(expanded, fill=None)

    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
    return len(rects)
