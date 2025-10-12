"""Tests for the PDFCleaner core behaviour."""

from __future__ import annotations

from pathlib import Path
import base64
import json

import fitz  # type: ignore
import pytest

from tomescrub.cli import main as cli_main, _worker_clean_task
from tomescrub.config import Config, load_defaults, WatermarkRuleConfig
from tomescrub.passwords import PasswordProvider
from tomescrub.processor import (
    PDFCleaner,
    PasswordAuthenticationError,
)


def _create_pdf(path: Path, text: str) -> None:
    """Utility to generate a simple PDF for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def _create_pdf_with_watermarks(
    path: Path,
    body_text: str,
    watermarks: list[str],
    *,
    watermark_font: str = "helv",
) -> None:
    """Create a PDF with body text and watermark lines near the bottom."""
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), body_text)
    for idx, watermark in enumerate(watermarks):
        y = page.rect.y1 - 30 - (idx * 10)
        page.insert_text(
            (36, y),
            watermark,
            fontname=watermark_font,
            fontsize=8,
        )
    document.save(path)
    document.close()


def _create_pdf_with_header_and_footer(path: Path, header: str, footer: str) -> None:
    """Create a PDF that contains header and footer lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), header)
    bottom_y = float(page.rect.y1) - 18  # keep within bottom ~6mm band
    page.insert_text((72, bottom_y), footer)
    document.save(path)
    document.close()


def _create_encrypted_pdf(path: Path, text: str, password: str) -> None:
    """Create a password protected PDF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(
        path,
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw=password,
        user_pw=password,
    )
    document.close()


def test_missing_file_raises(tmp_path: Path) -> None:
    """Ensure we surface an informative error when the source file is absent."""
    cleaner = PDFCleaner(output_dir=tmp_path / "out")
    with pytest.raises(FileNotFoundError):
        cleaner.clean_document(tmp_path / "missing.pdf")


def test_process_path_mirrors_structure_and_reads_text(tmp_path: Path) -> None:
    """Processing a directory should mirror layout and copy non-PDF assets."""
    source_root = tmp_path / "_unprocessed"
    nested_dir = source_root / "folder" / "nested"
    pdf_path = nested_dir / "doc.pdf"
    txt_path = nested_dir / "notes.txt"

    _create_pdf(pdf_path, "Hello from fitz")
    txt_path.write_text("sidecar data", encoding="utf-8")

    output_root = tmp_path / "_processed"
    cleaner = PDFCleaner(output_dir=output_root)
    results = list(cleaner.process_path(source_root))

    expected_pdf = output_root / "folder" / "nested" / "doc.pdf"
    expected_txt = output_root / "folder" / "nested" / "notes.txt"

    assert expected_pdf.exists(), "PDF should be written to mirrored structure"
    assert expected_txt.exists(), "Non-PDF assets should be copied"

    assert len(results) == 1
    result = results[0]
    assert result.output == expected_pdf
    assert "Hello from fitz" in result.text
    assert result.was_encrypted is False
    assert result.cleaned_permissions == -4
    assert result.watermarks_removed == 0
    assert result.hidden_text_removed == 0
    assert result.image_metadata_cleared >= 0
    assert result.document_metadata_cleared is True
    assert result.page_count == 1
    assert result.elapsed >= 0
    assert "total" in result.step_timings
    assert "open" in result.step_timings
    assert "save" in result.step_timings
    assert result.original_size_bytes > 0
    assert result.output_size_bytes > 0
    assert result.size_delta_bytes == result.output_size_bytes - result.original_size_bytes
    assert result.changed is True

    stats = cleaner.last_run_stats
    assert stats is not None
    assert stats.processed == 1
    assert stats.copied == 1
    assert stats.skipped == 0
    assert stats.total_pages == 1


def test_process_single_file_defaults_to_parent_root(tmp_path: Path) -> None:
    """Single file processing should place output directly under output_dir."""
    pdf_path = tmp_path / "single.pdf"
    _create_pdf(pdf_path, "standalone document")

    output_root = tmp_path / "_processed"
    cleaner = PDFCleaner(output_dir=output_root)
    results = list(cleaner.process_path(pdf_path))

    expected_pdf = output_root / "single.pdf"
    assert expected_pdf.exists()
    assert [r.output for r in results] == [expected_pdf]
    assert results[0].cleaned_permissions == -4
    assert results[0].watermarks_removed == 0
    assert results[0].hidden_text_removed == 0
    assert results[0].document_metadata_cleared is True
    assert results[0].page_count == 1
    assert results[0].elapsed >= 0
    assert "total" in results[0].step_timings
    assert "open" in results[0].step_timings
    assert results[0].original_size_bytes > 0
    assert results[0].output_size_bytes > 0
    assert results[0].size_delta_bytes == results[0].output_size_bytes - results[0].original_size_bytes
    assert results[0].changed is True


def test_skip_existing_outputs(tmp_path: Path) -> None:
    """Re-running with overwrite disabled should skip already processed PDFs."""
    source_root = tmp_path / "_unprocessed"
    pdf_path = source_root / "doc.pdf"
    _create_pdf(pdf_path, "original pass")

    output_root = tmp_path / "_processed"
    cleaner = PDFCleaner(output_dir=output_root)
    list(cleaner.process_path(source_root))

    cleaner_skip = PDFCleaner(output_dir=output_root, overwrite=False)
    results = list(cleaner_skip.process_path(source_root))

    assert results == []
    assert (output_root / "doc.pdf").exists()
    stats = cleaner_skip.last_run_stats
    assert stats is not None
    assert stats.processed == 0
    assert stats.skipped == 1
    assert stats.failed == 0


def test_encrypted_pdf_requires_password(tmp_path: Path) -> None:
    """Encrypted PDFs should raise when no password can be resolved."""
    pdf_path = tmp_path / "locked.pdf"
    _create_encrypted_pdf(pdf_path, "classified", "secret")

    cleaner = PDFCleaner(output_dir=tmp_path / "_processed")
    with pytest.raises(PasswordAuthenticationError):
        cleaner.clean_document(pdf_path)


def test_encrypted_pdf_with_default_password(tmp_path: Path) -> None:
    """A default password should unlock encrypted PDFs and strip protection."""
    pdf_path = tmp_path / "locked.pdf"
    _create_encrypted_pdf(pdf_path, "classified", "secret")

    output_root = tmp_path / "_processed"
    provider = PasswordProvider(default="secret")
    cleaner = PDFCleaner(output_dir=output_root, password_provider=provider)

    result = cleaner.clean_document(pdf_path)
    assert result.output.exists()
    with fitz.open(result.output) as unlocked:
        assert not unlocked.needs_pass
        assert not unlocked.is_encrypted
    assert "classified" in result.text
    assert result.was_encrypted is True
    assert result.cleaned_permissions == -4
    assert result.watermarks_removed == 0
    assert result.hidden_text_removed == 0
    assert result.document_metadata_cleared is True
    assert result.page_count == 1
    assert result.elapsed >= 0
    assert "total" in result.step_timings
    assert result.changed is True


def test_encrypted_pdf_with_hint_file(tmp_path: Path) -> None:
    """Passwords in a local hint file should be detected automatically."""
    source_root = tmp_path / "_unprocessed"
    pdf_path = source_root / "locked.pdf"
    _create_encrypted_pdf(pdf_path, "classified", "hinted")

    hint_file = pdf_path.parent / "passwords.txt"
    hint_file.write_text("locked.pdf = hinted\n", encoding="utf-8")

    output_root = tmp_path / "_processed"
    cleaner = PDFCleaner(output_dir=output_root, password_provider=PasswordProvider())

    result = next(cleaner.process_path(source_root))
    assert result.output.exists()
    with fitz.open(result.output) as unlocked:
        assert not unlocked.needs_pass
        assert not unlocked.is_encrypted
    assert result.was_encrypted is True
    assert result.cleaned_permissions == -4
    assert "classified" in result.text
    assert result.watermarks_removed == 0
    assert result.hidden_text_removed == 0
    assert result.document_metadata_cleared is True
    assert result.page_count == 1
    assert result.elapsed >= 0
    assert "total" in result.step_timings
    assert result.changed is True


def test_watermark_patterns_are_removed(tmp_path: Path) -> None:
    """Known watermark patterns should be detected and removed."""
    source_root = tmp_path / "_unprocessed"
    pdf_path = source_root / "watermarked.pdf"
    watermarks = [
        "Downloaded by Test User on 1/1/2024. Unauthorized distribution prohibited.",
        "Test User (Order #12345678)",
    ]
    _create_pdf_with_watermarks(pdf_path, "Body content remains", watermarks)

    output_root = tmp_path / "_processed"
    cleaner = PDFCleaner(output_dir=output_root)
    result = next(cleaner.process_path(source_root))

    assert result.watermarks_removed == len(watermarks)
    assert result.hidden_text_removed == 0
    assert result.page_count == 1
    assert result.elapsed >= 0
    assert "total" in result.step_timings
    assert result.changed is True
    with fitz.open(result.output) as cleaned:
        for page in cleaned:
            text = page.get_text()
            assert "Downloaded by" not in text
            assert "Order #" not in text


def test_watermark_font_mismatch(tmp_path: Path) -> None:
    """Watermark rules should ignore lines that use unexpected fonts."""
    source_root = tmp_path / "_unprocessed"
    pdf_path = source_root / "fonts.pdf"
    watermarks = ["Downloaded by Different Font on 1/1/2024. Unauthorized distribution prohibited."]
    _create_pdf_with_watermarks(
        pdf_path,
        "Body content remains",
        watermarks,
        watermark_font="courier",
    )

    output_root = tmp_path / "_processed"
    cleaner = PDFCleaner(output_dir=output_root)
    result = next(cleaner.process_path(source_root))

    assert result.watermarks_removed == 0
    assert result.page_count == 1
    assert result.elapsed >= 0
    assert "total" in result.step_timings
    assert result.changed is True
    with fitz.open(result.output) as cleaned:
        text = cleaned[0].get_text()
        assert "Downloaded by Different Font" in text


def test_hidden_text_and_metadata_sanitised(tmp_path: Path) -> None:
    """Hidden text and metadata should be stripped while visible content remains."""
    source_root = tmp_path / "_unprocessed"
    pdf_path = source_root / "hidden.pdf"
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    )

    document = fitz.open()
    source_root.mkdir(parents=True, exist_ok=True)
    document.set_metadata({"title": "Secret Title", "author": "Hidden Author"})
    if hasattr(document, "set_xml_metadata"):
        document.set_xml_metadata(
            '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"></rdf:RDF></x:xmpmeta>'
        )
    page = document.new_page()
    page.insert_text((72, 72), "Visible content")
    writer_hidden = fitz.TextWriter(page.rect)
    writer_hidden.append((72, 90), "Hidden Layer Text", fontsize=12)
    writer_hidden.write_text(page, opacity=0)
    image_rect = fitz.Rect(72, 120, 132, 180)
    page.insert_image(image_rect, stream=png_bytes)
    img_xref = page.get_images(full=True)[0][0]
    meta_stream = b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"></rdf:RDF></x:xmpmeta>'
    meta_xref = document.get_new_xref()
    document.update_object(meta_xref, "<< /Type /Metadata /Subtype /XML >>")
    document.update_stream(meta_xref, meta_stream)
    document.xref_set_key(img_xref, "Metadata", f"{meta_xref} 0 R")
    document.save(pdf_path)
    document.close()

    cleaner = PDFCleaner(output_dir=tmp_path / "_processed")
    result = cleaner.clean_document(pdf_path)

    assert result.hidden_text_removed > 0
    assert result.document_metadata_cleared is True
    assert result.image_metadata_cleared == 1
    assert result.page_count == 1
    assert result.elapsed >= 0
    assert "total" in result.step_timings
    assert "sanitize_metadata" in result.step_timings
    assert "remove_hidden_text" in result.step_timings
    assert result.original_size_bytes > 0
    assert result.output_size_bytes > 0
    assert result.size_delta_bytes == result.output_size_bytes - result.original_size_bytes
    assert result.changed is True

    with fitz.open(result.output) as cleaned:
        page = cleaned[0]
        text = page.get_text()
        assert "Hidden Layer Text" not in text
        metadata_values = {k: v for k, v in cleaned.metadata.items() if v}
        assert "Secret Title" not in metadata_values.values()
        if hasattr(cleaned, "get_xml_metadata"):
            assert cleaned.get_xml_metadata() in ("", None)
        image_xref = page.get_images(full=True)[0][0]
        assert cleaned.xref_get_key(image_xref, "Metadata")[0] == "null"


def test_resource_fork_pdf_is_skipped(tmp_path: Path) -> None:
    """macOS resource fork artefacts should be ignored gracefully."""
    source_root = tmp_path / "_unprocessed"
    resource_pdf = source_root / "__MACOSX" / "._dummy.pdf"
    resource_pdf.parent.mkdir(parents=True, exist_ok=True)
    resource_pdf.write_bytes(b"")

    cleaner = PDFCleaner(output_dir=tmp_path / "_processed")
    results = list(cleaner.process_path(source_root))

    assert results == []
    stats = cleaner.last_run_stats
    assert stats is not None
    assert stats.processed == 0
    assert stats.skipped == 1
    assert stats.failed == 0


def test_corrupt_pdf_registers_failure(tmp_path: Path) -> None:
    """Unreadable PDFs should be reported without aborting the run."""
    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_text("not a real pdf", encoding="utf-8")

    cleaner = PDFCleaner(output_dir=tmp_path / "_processed")
    results = list(cleaner.process_path(pdf_path))

    assert results == []
    stats = cleaner.last_run_stats
    assert stats is not None
    assert stats.failed == 1
    assert stats.processed == 0
    assert stats.failures[0][0] == pdf_path


def test_watermark_scan_modes(tmp_path: Path) -> None:
    """Ensure scan mode switches between bottom-band and full page parsing."""
    source_root = tmp_path / "_unprocessed"
    pdf_path = source_root / "scan.pdf"
    _create_pdf_with_header_and_footer(pdf_path, "HEADER SCAN TEST", "FOOTER SCAN TEST")

    rule_kwargs = {
        "name": "scan_band",
        "pattern": "SCAN TEST",
        "ignore_case": True,
        "max_distance_from_bottom": 1000.0,
    }

    # Bottom-only scan should leave header text untouched.
    config_bottom = Config.model_validate(load_defaults())
    config_bottom.io.output_dir = tmp_path / "_processed_bottom"
    config_bottom.watermarks.enabled = True
    config_bottom.watermarks.scan_mode = "bottom"
    config_bottom.watermarks.clip_bottom_mm = 10.0
    config_bottom.watermarks.stop_after_first = False
    config_bottom.watermarks.max_pages = 0
    config_bottom.watermarks.rules = [WatermarkRuleConfig(**rule_kwargs)]
    cleaner_bottom = PDFCleaner.from_config(config_bottom)
    result_bottom = cleaner_bottom.clean_document(pdf_path)
    with fitz.open(result_bottom.output) as cleaned:
        text_bottom = cleaned[0].get_text()
        assert "HEADER SCAN TEST" in text_bottom
        assert "FOOTER SCAN TEST" not in text_bottom

    # Full scan should remove both header and footer instances.
    config_full = Config.model_validate(load_defaults())
    config_full.io.output_dir = tmp_path / "_processed_full"
    config_full.watermarks.enabled = True
    config_full.watermarks.scan_mode = "full"
    config_full.watermarks.clip_bottom_mm = 0.0
    config_full.watermarks.stop_after_first = False
    config_full.watermarks.max_pages = 0
    config_full.watermarks.rules = [WatermarkRuleConfig(**rule_kwargs)]
    cleaner_full = PDFCleaner.from_config(config_full)
    result_full = cleaner_full.clean_document(pdf_path)
    with fitz.open(result_full.output) as cleaned:
        text_full = cleaned[0].get_text()
        assert "HEADER SCAN TEST" not in text_full
        assert "FOOTER SCAN TEST" not in text_full


def test_worker_clean_task_runs_in_separate_process(tmp_path: Path) -> None:
    """Worker helper should open documents independently and return results."""
    pdf_path = tmp_path / "input.pdf"
    output_path = tmp_path / "output.pdf"
    _create_pdf(pdf_path, "Worker process test")

    config = Config.model_validate(load_defaults())
    config.io.output_dir = tmp_path / "_processed_worker"
    config.performance.processes = 2
    payload = (config.model_dump(mode="json"), str(pdf_path), str(output_path), False)
    status, result = _worker_clean_task(payload)

    assert status == "ok"
    assert result.output == output_path
    assert result.output.exists()
    with fitz.open(result.output) as cleaned:
        assert "Worker process test" in cleaned[0].get_text()


def test_cli_run_log_written(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI should persist a run log entry when enabled."""
    source_root = tmp_path / "_unprocessed"
    pdf_path = source_root / "doc.pdf"
    _create_pdf(pdf_path, "Run log test")

    output_root = tmp_path / "_processed"
    log_path = tmp_path / "run_log.ndjson"

    args = [
        str(source_root),
        "--output-dir",
        str(output_root),
        "--run-log",
        "--run-log-path",
        str(log_path),
        "--run-log-quiet",
    ]
    exit_code = cli_main(args)
    assert exit_code == 0
    assert log_path.exists()
    captured = capsys.readouterr()
    assert "Run log appended" not in captured.out

    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])

    assert record["stats"]["processed"] == 1
    processed_entries = [entry for entry in record["files"] if entry["status"] == "processed"]
    assert len(processed_entries) == 1
    processed_entry = processed_entries[0]
    assert processed_entry["output"].endswith(".pdf")
    assert processed_entry["pages"] == 1
    assert "step_timings" in processed_entry
    assert "stage_timings_ms" in processed_entry
    for key in ("open_ms", "sanitize_ms", "watermark_ms", "save_ms", "extract_ms"):
        assert key in processed_entry
        assert processed_entry["stage_timings_ms"][key] == processed_entry[key]
    assert processed_entry["original_size_bytes"] >= 0
    assert processed_entry["output_size_bytes"] >= 0
    assert processed_entry["size_delta_bytes"] == processed_entry["output_size_bytes"] - processed_entry["original_size_bytes"]
