"""Detection and removal of watermark text spans."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import FrozenSet, Iterable, List, Optional, Pattern, Sequence

import fitz  # type: ignore

from .config import Config, load_defaults


def _normalize_text(value: str) -> str:
    """Coalesce whitespace and trim stray spaces before punctuation."""
    collapsed = " ".join(value.split())
    # Remove spaces before punctuation marks we expect in watermark patterns.
    return re.sub(r"\s+([.,;:?!])", r"\1", collapsed).strip()


@dataclass(frozen=True)
class WatermarkRule:
    """Describe a watermark pattern and basic heuristics."""

    name: str
    pattern: Pattern[str]
    max_font_size: Optional[float] = None
    min_font_size: Optional[float] = None
    max_distance_from_bottom: float = 120.0  # points (~1.67 inch)
    allowed_fonts: Optional[FrozenSet[str]] = None

    def matches(
        self,
        *,
        text: str,
        font_sizes: Sequence[float],
        fonts: Sequence[str],
        distance_from_bottom: float,
    ) -> bool:
        if not self.pattern.search(text):
            return False
        if self.max_font_size is not None and any(size > self.max_font_size for size in font_sizes):
            return False
        if self.min_font_size is not None and any(size < self.min_font_size for size in font_sizes):
            return False
        if distance_from_bottom > self.max_distance_from_bottom:
            return False
        if self.allowed_fonts is not None:
            lowered = {font.lower() for font in fonts if font}
            if not lowered or not lowered.issubset(self.allowed_fonts):
                return False
        return True


@dataclass(frozen=True)
class WatermarkMatch:
    """Represent a detected watermark line."""

    rule: WatermarkRule
    rect: fitz.Rect
    text: str


_DEFAULT_RULES = Config.model_validate(load_defaults()).compile_watermark_rules()


def get_default_watermark_rules() -> List[WatermarkRule]:
    """Return a copy of the packaged default watermark rules."""
    return list(_DEFAULT_RULES)


def _iter_text_lines(
    page: fitz.Page,
    clip_rect: Optional[fitz.Rect] = None,
) -> Iterable[tuple[fitz.Rect, str, List[float], List[str]]]:
    """Yield (rect, text, font_sizes, fonts) tuples for each textual line on a page."""
    page_dict = page.get_text("dict", sort=True, clip=clip_rect)
    for block in page_dict.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = "".join(span.get("text", "") for span in spans).strip()
            if not text:
                continue
            rect = fitz.Rect(line.get("bbox"))
            font_sizes = [float(span.get("size", 0.0)) for span in spans if span.get("size") is not None]
            fonts = [str(span.get("font", "")).lower() for span in spans if span.get("font")]
            yield rect, text, font_sizes, fonts


def find_watermark_matches(
    page: fitz.Page,
    rules: Optional[Sequence[WatermarkRule]] = None,
    *,
    clip_rect: Optional[fitz.Rect] = None,
    stop_after_first: bool = False,
) -> List[WatermarkMatch]:
    """Detect watermark candidates on a page according to the provided rules."""
    active_rules = list(rules or get_default_watermark_rules())
    if not active_rules:
        return []

    matches: List[WatermarkMatch] = []
    page_bottom = float(page.rect.y1)

    for rect, raw_text, font_sizes, fonts in _iter_text_lines(page, clip_rect=clip_rect):
        if not font_sizes:
            continue
        normalized = _normalize_text(raw_text)
        if not normalized:
            continue
        distance_from_bottom = page_bottom - float(rect.y1)
        for rule in active_rules:
            if rule.matches(
                text=normalized,
                font_sizes=font_sizes,
                fonts=fonts,
                distance_from_bottom=distance_from_bottom,
            ):
                matches.append(
                    WatermarkMatch(
                        rule=rule,
                        rect=rect,
                        text=normalized,
                    )
                )
                if stop_after_first:
                    return matches
                break
    return matches


def remove_watermarks(
    page: fitz.Page,
    rules: Optional[Sequence[WatermarkRule]] = None,
    *,
    clip_rect: Optional[fitz.Rect] = None,
    stop_after_first: bool = False,
) -> List[WatermarkMatch]:
    """
    Remove detected watermark lines from a page.

    Returns:
        The list of watermark matches that were removed.
    """
    matches = find_watermark_matches(page, rules=rules, clip_rect=clip_rect, stop_after_first=stop_after_first)
    if not matches:
        return []

    for match in matches:
        # Expand slightly to make sure the redaction covers the full text height.
        rect = fitz.Rect(match.rect)
        rect.y0 -= 0.5
        rect.y1 += 0.5
        rect.x0 -= 0.5
        rect.x1 += 0.5
        page.add_redact_annot(rect, fill=None)

    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
    return matches


# Compatibility alias to expose the configured rules at import time.
DEFAULT_WATERMARK_RULES = get_default_watermark_rules()
