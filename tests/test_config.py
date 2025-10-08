"""Tests for the configuration loading helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from tomescrub.config import load_config


def test_load_config_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bundled defaults should produce a usable configuration."""
    # ensure environment overrides do not leak in
    monkeypatch.delenv("TOMESCRUB__IO__OUTPUT_DIR", raising=False)
    config = load_config(config_path=None, cli_sets=[])
    assert config.io.output_dir == Path("_processed")
    assert config.clean.sanitize_metadata is True
    assert config.clean.hidden_text_alpha_threshold == 0
    assert config.passwords.hint_filename == "passwords.txt"


def test_load_config_cli_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """CLI overrides should take highest priority."""
    monkeypatch.delenv("TOMESCRUB__IO__OUTPUT_DIR", raising=False)
    overrides = [
        f"io.output_dir={tmp_path / 'out'}",
        "clean.sanitize_metadata=false",
        "passwords.hint_filename=",
    ]
    config = load_config(config_path=None, cli_sets=overrides)
    assert config.io.output_dir == tmp_path / "out"
    assert config.clean.sanitize_metadata is False
    assert config.passwords.hint_filename is None


def test_load_config_env_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables should override defaults but not CLI entries."""
    monkeypatch.setenv("TOMESCRUB__IO__OUTPUT_DIR", "env_output")
    config_env_only = load_config(config_path=None, cli_sets=[])
    assert config_env_only.io.output_dir == Path("env_output")

    config_cli = load_config(config_path=None, cli_sets=["io.output_dir=cli_output"])
    assert config_cli.io.output_dir == Path("cli_output")
