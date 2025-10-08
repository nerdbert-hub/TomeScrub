"""Configuration loading and merging utilities."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import tomllib
from platformdirs import user_config_dir
from pydantic import ValidationError

from .schema import Config

APP_NAME = "tomescrub"
ENV_PREFIX = "TOMESCRUB__"


def read_toml(path: Path) -> dict[str, Any]:
    """Read a TOML file if it exists."""
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def discover_config_path(explicit: str | Path | None) -> Path | None:
    """Determine the config file to load based on precedence."""
    if explicit:
        candidate = Path(explicit).expanduser()
        return candidate if candidate.exists() else None

    cwd_candidate = Path("tomescrub.toml")
    if cwd_candidate.exists():
        return cwd_candidate

    user_candidate = Path(user_config_dir(APP_NAME)) / "config.toml"
    if user_candidate.exists():
        return user_candidate

    return None


def load_defaults() -> dict[str, Any]:
    """Load project defaults bundled with the package."""
    defaults_path = Path(__file__).parent / "defaults.toml"
    return read_toml(defaults_path)


def parse_scalar(value: str) -> Any:
    """Coerce scalar strings (from env or CLI) into native Python types."""
    lower_value = value.lower()
    if lower_value in {"true", "false"}:
        return lower_value == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def env_to_dict(env: Mapping[str, str]) -> dict[str, Any]:
    """Parse environment variables into a nested dictionary."""
    result: dict[str, Any] = {}
    prefix = ENV_PREFIX
    for key, value in env.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix) :].lower().split("__")
        ref = result
        for part in parts[:-1]:
            ref = ref.setdefault(part, {})
        ref[parts[-1]] = parse_scalar(value)
    return result


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge ``override`` values into ``base`` recursively."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = merge_dicts(dict(base[key]), value)
        else:
            base[key] = value
    return base


def parse_cli_overrides(entries: Iterable[str]) -> dict[str, Any]:
    """Parse ``key=value`` pairs into a nested dictionary."""
    result: dict[str, Any] = {}
    for entry in entries:
        if "=" not in entry:
            continue
        key, raw = entry.split("=", 1)
        ref = result
        parts = key.split(".")
        for part in parts[:-1]:
            ref = ref.setdefault(part, {})
        ref[parts[-1]] = parse_scalar(raw)
    return result


def load_config(config_path: str | Path | None, cli_sets: Iterable[str]) -> Config:
    """Load configuration using the precedence rules."""
    merged = load_defaults()
    discovered = discover_config_path(config_path)
    if discovered:
        merged = merge_dicts(merged, read_toml(discovered))

    merged = merge_dicts(merged, env_to_dict(os.environ))
    merged = merge_dicts(merged, parse_cli_overrides(cli_sets))

    try:
        return Config.model_validate(merged)
    except ValidationError as exc:
        print("Configuration error:", file=sys.stderr)
        print(exc, file=sys.stderr)
        raise
