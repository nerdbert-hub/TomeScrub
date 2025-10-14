"""Tests for save backend implementations."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tomescrub.config.schema import SaveConfig
from tomescrub.save_backends import (
    ChainSaveBackend,
    PikepdfCleaner,
    PyMuPDFSaveBackend,
    QpdfLinearizer,
    get_save_backend,
)
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




def test_qpdf_command_preserve_when_not_linear(tmp_path: Path) -> None:
    """QPDF should preserve object streams when not linearising."""
    config = SaveConfig(backend="qpdf")
    config.linearize = False
    saver = QpdfLinearizer(config)
    command = saver._build_command(
        executable="qpdf",
        input_path=tmp_path / "input.pdf",
        output_path=tmp_path / "output.pdf",
        linearize=False,
    )
    assert "--linearize" not in command
    assert "--object-streams=preserve" in command
    assert "--compress-streams=y" in command

def test_pikepdf_cleaner_removes_thumbnails(tmp_path: Path) -> None:
    """PikePDF cleaner should drop thumbnails and OCG metadata."""
    pikepdf = pytest.importorskip("pikepdf")
    config = SaveConfig()
    config.pikepdf.enabled = True
    config.misc.remove_thumbnails = True
    config.layers.remove_ocg_metadata = True
    cleaner = PikepdfCleaner(config)

    input_pdf = tmp_path / "with_thumb.pdf"
    with pikepdf.Pdf.new() as pdf:
        page = pdf.add_blank_page()
        page.obj["/Thumb"] = pikepdf.Stream(pdf, b"thumb")
        pdf.Root["/OCProperties"] = pikepdf.Dictionary(
            {"/D": pikepdf.Dictionary({"/OFF": pikepdf.Array([])})}
        )
        pdf.save(input_pdf)

    output_pdf = tmp_path / "cleaned.pdf"
    cleaner.clean_file(input_pdf, output_pdf)

    with pikepdf.Pdf.open(output_pdf) as pdf:
        assert all("/Thumb" not in page.obj for page in pdf.pages)
        assert "/OCProperties" not in pdf.Root


def test_chain_backend_selection() -> None:
    """Chain backend should resolve to ChainSaveBackend."""
    backend = get_save_backend(SaveConfig(backend="chain"))
    assert isinstance(backend, ChainSaveBackend)


def test_chain_backend_writes_output(tmp_path: Path) -> None:
    """Chain backend should produce a PDF when required tools are available."""
    fitz = pytest.importorskip("fitz")
    config = SaveConfig(backend="chain")
    backend = ChainSaveBackend(config)

    document = fitz.open()
    document.new_page(width=72, height=72)
    ctx = SimpleNamespace(output_path=tmp_path / "output.pdf")

    outcome = backend.save(cleaner=None, ctx=ctx, document=document)
    document.close()

    assert ctx.output_path.exists()
    assert isinstance(outcome.cleaned_permissions, int)
