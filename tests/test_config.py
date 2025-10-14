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
    assert config.clean.strip_document_metadata is True
    assert config.clean.strip_image_metadata is True
    assert config.clean.remove_hidden_text is True
    assert config.clean.extract_text is True
    assert config.passwords.hint_filename == "passwords.txt"
    assert config.watermarks.enabled is True
    assert config.watermarks.clip_bottom_mm == 0.0
    assert config.watermarks.stop_after_first is False
    assert config.watermarks.max_pages == 0
    assert config.performance.processes == "auto"
    assert config.performance.batch_size == 8
    assert config.save.linearize is False
    assert config.save.garbage == 4
    assert config.save.deflate is True
    assert config.io.run_log.enabled is False
    assert config.io.run_log.path is None
    assert config.io.run_log.quiet is False
    assert config.io.skip_unchanged is False


def test_load_config_cli_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """CLI overrides should take highest priority."""
    monkeypatch.delenv("TOMESCRUB__IO__OUTPUT_DIR", raising=False)
    overrides = [
        f"io.output_dir={tmp_path / 'out'}",
        "clean.sanitize_metadata=false",
        "passwords.hint_filename=",
        "io.run_log.enabled=true",
        f"io.run_log.path={tmp_path / 'log.ndjson'}",
        "io.run_log.quiet=true",
    ]
    config = load_config(config_path=None, cli_sets=overrides)
    assert config.io.output_dir == tmp_path / "out"
    assert config.clean.sanitize_metadata is False
    assert config.passwords.hint_filename is None
    assert config.io.run_log.enabled is True
    assert config.io.run_log.path == tmp_path / "log.ndjson"
    assert config.io.run_log.quiet is True


def test_load_config_env_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables should override defaults but not CLI entries."""
    monkeypatch.setenv("TOMESCRUB__IO__OUTPUT_DIR", "env_output")
    monkeypatch.setenv("TOMESCRUB__IO__RUN_LOG__ENABLED", "true")
    monkeypatch.setenv("TOMESCRUB__IO__RUN_LOG__PATH", "env_log.ndjson")
    monkeypatch.setenv("TOMESCRUB__IO__RUN_LOG__QUIET", "true")
    config_env_only = load_config(config_path=None, cli_sets=[])
    assert config_env_only.io.output_dir == Path("env_output")
    assert config_env_only.io.run_log.enabled is True
    assert config_env_only.io.run_log.path == Path("env_log.ndjson")
    assert config_env_only.io.run_log.quiet is True

    config_cli = load_config(config_path=None, cli_sets=["io.output_dir=cli_output"])
    assert config_cli.io.output_dir == Path("cli_output")
    # CLI should not override environment-derived run log settings unless explicitly set
    assert config_cli.io.run_log.enabled is True
    assert config_cli.io.run_log.path == Path("env_log.ndjson")
    assert config_cli.io.run_log.quiet is True


def test_profile_fast_applied() -> None:
    config = load_config(config_path=None, cli_sets=[], profile="fast")
    assert config.clean.sanitize_metadata is False
    assert config.watermarks.rules[0].name == "order_reference"


def test_profile_web_settings() -> None:
    config = load_config(config_path=None, cli_sets=[], profile="web")
    assert config.save.pdf_version == "1.7"
    assert config.save.linearize is False
    assert config.save.images.color_target_ppi == 180
    assert config.save.images.gray_target_ppi == 180
    assert config.save.images.photo_compression == "jpeg"
    assert config.save.ghostscript.enabled is True
    assert config.save.ghostscript.pass_through_jpeg_images is True
    assert config.save.ghostscript.max_inline_image_size == 2048


def test_save_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sanity check defaults for save configuration."""
    monkeypatch.delenv("TOMESCRUB__SAVE__BACKEND", raising=False)
    config = load_config(config_path=None, cli_sets=[])
    save = config.save
    assert save.backend == "pymupdf"
    assert save.linearize is False
    assert save.pdf_version is None
    assert save.fonts.subset is True
    assert save.images.threshold_factor == 2.0
    assert save.images.color_target_ppi == 150
    assert save.images.gray_target_ppi == 150
    assert save.images.jpeg.qfactor == pytest.approx(0.08)
    assert save.links.preserve_links is True
    assert save.misc.detect_duplicate_images is True
    assert save.ghostscript.compatibility_level == "1.7"
    assert save.ghostscript.pass_through_jpeg_images is True
    assert save.ghostscript.max_inline_image_size == 2048
    assert save.ghostscript.enabled is False


def test_save_backend_cli_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI overrides should update nested save configuration."""
    overrides = [
        "save.backend=qpdf",
        "save.fonts.subset=false",
        "save.images.photo_compression=zip",
    ]
    config = load_config(config_path=None, cli_sets=overrides)
    assert config.save.backend == "qpdf"
    assert config.save.fonts.subset is False
    assert config.save.images.photo_compression == "zip"
