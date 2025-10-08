"""Pydantic configuration schema for TomeScrub."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class IOConfig(BaseModel):
    """Input/output related configuration."""

    output_dir: Path = Path("_processed")
    overwrite_existing: bool = True

    @field_validator("output_dir", mode="before")
    @classmethod
    def _coerce_path(cls, value: object) -> Path:
        if isinstance(value, Path):
            return value.expanduser()
        return Path(str(value)).expanduser()


class CleanConfig(BaseModel):
    """Cleaning behaviour configuration."""

    sanitize_metadata: bool = True
    hidden_text_alpha_threshold: int = 0

    @field_validator("hidden_text_alpha_threshold")
    @classmethod
    def _validate_alpha(cls, value: int) -> int:
        if not (0 <= value <= 255):
            raise ValueError("hidden_text_alpha_threshold must be between 0 and 255")
        return value


class PasswordsConfig(BaseModel):
    """Password resolution configuration."""

    default: Optional[str] = None
    hint_filename: Optional[str] = "passwords.txt"
    password_file: Optional[Path] = None

    @field_validator("default", "hint_filename", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("password_file", mode="before")
    @classmethod
    def _coerce_password_file(cls, value: object) -> Optional[Path]:
        if value in (None, ""):
            return None
        if isinstance(value, Path):
            return value.expanduser()
        return Path(str(value)).expanduser()


class WatermarkRuleConfig(BaseModel):
    """Declarative watermark rule definition."""

    name: str
    pattern: str
    ignore_case: bool = False
    max_font_size: Optional[float] = None
    min_font_size: Optional[float] = None
    max_distance_from_bottom: float = 120.0
    fonts: Optional[List[str]] = None

    def compile(self):
        """Compile to an executable WatermarkRule."""
        import re

        from ..watermarks import WatermarkRule

        flags = re.IGNORECASE if self.ignore_case else 0
        font_set = None
        if self.fonts:
            font_set = frozenset(font.lower() for font in self.fonts if font)
        return WatermarkRule(
            name=self.name,
            pattern=re.compile(self.pattern, flags=flags),
            max_font_size=self.max_font_size,
            min_font_size=self.min_font_size,
            max_distance_from_bottom=self.max_distance_from_bottom,
            allowed_fonts=font_set,
        )


class WatermarkConfig(BaseModel):
    """Watermark detection configuration."""

    rules: List[WatermarkRuleConfig] = Field(default_factory=list)

    def compile_rules(self) -> List["WatermarkRule"]:
        return [rule.compile() for rule in self.rules]


class Config(BaseModel):
    """Top-level TomeScrub configuration model."""

    io: IOConfig = Field(default_factory=IOConfig)
    clean: CleanConfig = Field(default_factory=CleanConfig)
    passwords: PasswordsConfig = Field(default_factory=PasswordsConfig)
    watermarks: WatermarkConfig = Field(default_factory=WatermarkConfig)

    def compile_watermark_rules(self) -> List["WatermarkRule"]:
        return self.watermarks.compile_rules()

    def model_update_from(self, other: "Config") -> "Config":
        """Return a copy updated with another Config's values."""
        return self.model_copy(update=other.model_dump())

    def override(self, updates: dict) -> "Config":
        """Return a copy updated with nested dictionary overrides."""
        merged = _merge_dicts(self.model_dump(), updates)
        return Config.model_validate(merged)


def _merge_dicts(base: dict, overrides: dict) -> dict:
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result
