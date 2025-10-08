"""TomeScrub package exposing high-level cleaning utilities."""

from .cli import main as cli_main
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
