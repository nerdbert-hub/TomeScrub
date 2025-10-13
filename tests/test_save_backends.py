"""Tests for save backend implementations."""

from __future__ import annotations

from pathlib import Path

import pytest

from tomescrub.config.schema import SaveConfig
from tomescrub.save_backends import (
    ChainSaveBackend,
    GhostscriptSaver,
    PikepdfCleaner,
    PyMuPDFSaveBackend,
    QpdfLinearizer,
    get_save_backend,
)


def test_get_save_backend_ghostscript() -> None:
    """Selecting the ghostscript backend should return GhostscriptSaver."""
    config = SaveConfig(backend="ghostscript")
    backend = get_save_backend(config)
    assert isinstance(backend, GhostscriptSaver)


def test_ghostscript_command_includes_expected_flags(tmp_path: Path) -> None:
    """Ghostscript command should reflect configured compression and linearisation."""
    config = SaveConfig(backend="ghostscript")
    config.linearize = True
    config.pdf_version = "1.7"
    config.images.photo_compression = "jpeg"
    config.images.lineart_compression = "zip"
    config.images.jpeg.qfactor = 0.07
    config.images.jpeg.blend = 1
    config.images.jpeg.h_samples = [2, 1, 1, 2]
    config.images.jpeg.v_samples = [2, 1, 1, 2]
    config.misc.fast_web_view = False
    config.links.preserve_links = True
    config.misc.leave_color_unchanged = True
    config.ghostscript.extra = ["-dSomeFlag=true"]

    saver = GhostscriptSaver(config)
    command = saver._build_command(
        executable="/usr/bin/gs",
        input_path=tmp_path / "input.pdf",
        output_path=tmp_path / "output.pdf",
        linearize_with_gs=True,
    )

    assert "/usr/bin/gs" in command[0]
    assert "-sDEVICE=pdfwrite" in command
    assert "-dFastWebView=true" in command
    assert "-dPrinted=false" in command
    assert "-sColorImageFilter=/DCTEncode" in command
    assert "-sGrayImageFilter=/DCTEncode" in command
    assert "-sMonoImageFilter=/FlateEncode" in command
    assert "-dColorImageResolution=300" in command
    assert "-dColorImageDownsampleThreshold=1.5" in command
    assert "-dCompatibilityLevel=1.7" in command
    assert "-dSomeFlag=true" in command
    assert any("/QFactor 0.07" in arg for arg in command), "Distiller parameters missing"


def test_ghostscript_command_zip_without_distiller(tmp_path: Path) -> None:
    """ZIP compression should use Flate filters without distiller params."""
    config = SaveConfig(backend="ghostscript")
    config.images.photo_compression = "zip"
    config.images.lineart_compression = "zip"
    config.images.jpeg.qfactor = None

    saver = GhostscriptSaver(config)
    command = saver._build_command(
        executable="gs",
        input_path=tmp_path / "input.pdf",
        output_path=tmp_path / "output.pdf",
        linearize_with_gs=False,
    )

    assert "-sColorImageFilter=/FlateEncode" in command
    assert "-sGrayImageFilter=/FlateEncode" in command
    assert "-sMonoImageFilter=/FlateEncode" in command
    assert all("/QFactor" not in arg for arg in command)


def test_default_backend_factory_pymupdf() -> None:
    """Ensure pymupdf remains default backend."""
    backend = get_save_backend(SaveConfig())
    assert isinstance(backend, PyMuPDFSaveBackend)


def test_qpdf_command_linearize(tmp_path: Path) -> None:
    """QPDF command should include linearisation and version flags."""
    config = SaveConfig(backend="qpdf")
    config.linearize = True
    config.pdf_version = "1.7"
    config.qpdf.extra = ["--stream-data=compress"]
    saver = QpdfLinearizer(config)
    command = saver._build_command(
        executable="/usr/bin/qpdf",
        input_path=tmp_path / "in.pdf",
        output_path=tmp_path / "out.pdf",
        linearize=True,
    )
    assert command[0] == "/usr/bin/qpdf"
    assert "--linearize" in command
    assert "--force-version=1.7" in command
    assert "--object-streams=generate" in command
    assert "--compress-streams=y" in command
    assert "--stream-data=compress" in command
    assert command[-2:] == [str(tmp_path / "in.pdf"), str(tmp_path / "out.pdf")]


def test_pikepdf_cleaner_removes_thumbnails(tmp_path: Path) -> None:
    """PikePDF cleaner should drop thumbnails and OCG metadata."""
    pikepdf = pytest.importorskip("pikepdf")
    from pikepdf import Name
    config = SaveConfig(backend="pikepdf")
    config.pikepdf.enabled = True
    config.misc.remove_thumbnails = True
    config.layers.remove_ocg_metadata = True
    cleaner = PikepdfCleaner(config)

    input_pdf = tmp_path / "with_thumb.pdf"
    with pikepdf.Pdf.new() as pdf:
        page = pdf.pages.append(pikepdf.Page(pdf))
        page.obj[Name("/Thumb")] = pikepdf.Stream(pdf, b"thumb")
        pdf.root[Name("/OCProperties")] = pikepdf.Dictionary(
            {Name("/D"): pikepdf.Dictionary({Name("/OFF"): pikepdf.Array([])})}
        )
        pdf.save(input_pdf)

    output_pdf = tmp_path / "cleaned.pdf"
    cleaner.clean_file(input_pdf, output_pdf)

    with pikepdf.Pdf.open(output_pdf) as pdf:
        assert all(Name("/Thumb") not in page.obj for page in pdf.pages)
        assert Name("/OCProperties") not in pdf.root


def test_chain_backend_selection() -> None:
    """Chain backend should resolve to ChainSaveBackend."""
    backend = get_save_backend(SaveConfig(backend="chain"))
    assert isinstance(backend, ChainSaveBackend)
