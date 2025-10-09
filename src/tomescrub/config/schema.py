"""Pydantic configuration schema for TomeScrub."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# new RunLog model
class RunLogConfig(BaseModel):
    """Run logging options."""

    enabled: bool = False
    path: Optional[Path] = None
    quiet: bool = False

    @field_validator("path", mode="before")
    @classmethod
    def _coerce_path(cls, value: object) -> Optional[Path]:
        if value in (None, "", False):
            return None
        if isinstance(value, Path):
            return value.expanduser()
        return Path(str(value)).expanduser()


class IOConfig(BaseModel):
    """Input/output related configuration."""

    output_dir: Path = Path("_processed")
    overwrite_existing: bool = True
    skip_unchanged: bool = False
    run_log: RunLogConfig = Field(default_factory=RunLogConfig)

    @field_validator("output_dir", mode="before")
    @classmethod
    def _coerce_path(cls, value: object) -> Path:
        if isinstance(value, Path):
            return value.expanduser()
        return Path(str(value)).expanduser()


class CleanConfig(BaseModel):
    """Cleaning behaviour configuration."""

    sanitize_metadata: Optional[bool] = None  # legacy alias
    strip_document_metadata: bool = True
    strip_image_metadata: bool = True
    remove_hidden_text: bool = True
    hidden_text_alpha_threshold: int = 0
    extract_text: bool = True

    @field_validator("hidden_text_alpha_threshold")
    @classmethod
    def _validate_alpha(cls, value: int) -> int:
        if not (0 <= value <= 255):
            raise ValueError("hidden_text_alpha_threshold must be between 0 and 255")
        return value

    @model_validator(mode="after")
    def _apply_sanitize_alias(self) -> "CleanConfig":
        if self.sanitize_metadata is not None:
            flag = bool(self.sanitize_metadata)
            object.__setattr__(self, "strip_document_metadata", flag)
            object.__setattr__(self, "strip_image_metadata", flag)
        return self


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

    enabled: bool = True
    clip_bottom_mm: Optional[float] = None
    stop_after_first: bool = False
    max_pages: int = 0
    rules: List[WatermarkRuleConfig] = Field(default_factory=list)

    @field_validator("max_pages")
    @classmethod
    def _validate_max_pages(cls, value: int) -> int:
        if value < 0:
            raise ValueError("watermarks.max_pages must be >= 0")
        return value

    def compile_rules(self) -> List["WatermarkRule"]:
        return [rule.compile() for rule in self.rules]


class PerformanceConfig(BaseModel):
    """Performance tuning configuration."""

    processes: Union[int, str] = "auto"
    batch_size: int = 8

    @field_validator("processes")
    @classmethod
    def _validate_processes(cls, value: Union[int, str]) -> Union[int, str]:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered != "auto":
                raise ValueError('performance.processes must be an integer or "auto"')
            return "auto"
        if value < 1:
            raise ValueError("performance.processes must be >= 1")
        return value

    @field_validator("batch_size")
    @classmethod
    def _validate_batch_size(cls, value: int) -> int:
        if value < 1:
            raise ValueError("performance.batch_size must be >= 1")
        return value


class SaveConfig(BaseModel):
    """PDF saving options."""

    linearize: bool = False
    garbage: int = 4
    deflate: bool = True

    @field_validator("garbage")
    @classmethod
    def _validate_garbage(cls, value: int) -> int:
        if not (0 <= value <= 4):
            raise ValueError("save.garbage must be between 0 and 4")
        return value


class Config(BaseModel):
    """Top-level TomeScrub configuration model."""

    io: IOConfig = Field(default_factory=IOConfig)
    clean: CleanConfig = Field(default_factory=CleanConfig)
    passwords: PasswordsConfig = Field(default_factory=PasswordsConfig)
    watermarks: WatermarkConfig = Field(default_factory=WatermarkConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    save: SaveConfig = Field(default_factory=SaveConfig)

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
