"""TomeScrub package exposing high-level cleaning utilities."""

from __future__ import annotations

import warnings

# Silence noisy PyMuPDF deprecation warnings stemming from upstream SWIG types.
warnings.filterwarnings(
    "ignore",
    message=r"builtin type .* has no __module__ attribute",
    category=DeprecationWarning,
)

from .cli import main as cli_main
from .config import Config, load_config
from .passwords import PasswordProvider, load_password_file
from .processor import (
    DocumentProcessingResult,
    PDFCleaner,
    PasswordAuthenticationError,
)
from .sanitizer import (
    clear_document_metadata,
    clear_image_metadata,
    remove_hidden_text,
)
from .watermarks import (
    DEFAULT_WATERMARK_RULES,
    WatermarkRule,
    remove_watermarks,
)

__all__ = [
    "Config",
    "load_config",
    "PDFCleaner",
    "DocumentProcessingResult",
    "PasswordAuthenticationError",
    "PasswordProvider",
    "WatermarkRule",
    "DEFAULT_WATERMARK_RULES",
    "clear_document_metadata",
    "clear_image_metadata",
    "remove_hidden_text",
    "remove_watermarks",
    "cli_main",
    "load_password_file",
]
