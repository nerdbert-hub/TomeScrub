# TomeScrub

Tooling scaffold for cleaning PDF documents with [PyMuPDF](https://pymupdf.readthedocs.io/).

## Features
- Package-first layout under `src/` for simple installation and reuse.
- TomeScrub utilities encapsulating metadata stripping, text extraction, and saving workflow.
- Password-aware processing that unlocks encrypted PDFs before saving unprotected copies and reports how many were unlocked per run.
- Rule-based removal of page footer watermarks (e.g., `Downloaded by …` or `Name (Order #123)`) before writing the cleaned PDF.
- Automated sanitisation of document/image metadata and fully hidden text spans (keeps bookmarks, vector graphics, and visible content untouched).
- Directory-aware processing that mirrors an input tree into a `_processed` output, copying non-PDF assets.
- Streaming processor that avoids loading every result into memory; ideal for large collections.
- CLI entry point for quick command-line usage over files or entire folders.
- Pytest harness seeded with regression tests for the core behaviours.
- Rich `DocumentProcessingResult` metadata (original vs cleaned permissions, encryption flag, extracted text).

## Getting Started

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
python -m pip install --user -e .
```

The editable install registers `tomescrub` so you can invoke the CLI without adjusting `PYTHONPATH`.

## Usage

### Process a single file

```bash
python -m tomescrub path/to/file.pdf --output-dir _processed
```

### Mirror a directory structure

```bash
python -m tomescrub _unprocessed --output-dir _processed
```

### Skip already processed PDFs

```bash
python -m tomescrub _unprocessed --output-dir _processed --skip-existing
```

`--skip-existing` is helpful on repeated runs or when output files are open elsewhere.

### Unlock password protected PDFs

```bash
python -m tomescrub _unprocessed --output-dir _processed --password supersecret
```

Provide `--password` for a global default, `--password-file passwords.txt` for a top-level mapping file, and store per-directory hints in a `passwords.txt` that lives alongside the PDFs. Use `--password-hints none` to disable the directory scan. The CLI summary calls out how many encrypted files were unlocked.

Each password file uses `name.pdf = password` entries (lines starting with `#` are ignored), plus an optional `* = fallback` wildcard for that folder. Example:

```
# passwords.txt
locked.pdf = supersecret
*.pdf = fallback-password
```

## Run Tests

```bash
python -m pytest
```

## Project Layout

```
TomeScrub/
|-- README.md
|-- requirements.txt
|-- pyproject.toml
|-- src/
|   `-- tomescrub/
|       |-- __init__.py
|       |-- __main__.py
|       |-- cli.py
|       |-- passwords.py
|       |-- processor.py
|       |-- sanitizer.py
|       `-- watermarks.py
`-- tests/
    |-- __init__.py
    `-- test_processor.py
```

## Metadata & Hidden Text Sanitisation
- Each PDF run through the cleaner has its Info/XMP metadata cleared and embedded image metadata stripped.
- Text spans rendered with zero opacity (`alpha==0`) are removed automatically; this mirrors Acrobat's “Remove Hidden Information → Hidden Text”.
- Bookmarks, annotations, vector artwork, and visible imagery remain intact. We intentionally skip Acrobat's “Deleted or cropped content” / “Overlapping objects” options to avoid bloat.
- To tune hidden-text detection, adjust `remove_hidden_text` in `src/tomescrub/sanitizer.py` (e.g., raise `alpha_threshold` for semi-transparent layers).

## Watermark Rules
- Default patterns live in `src/tomescrub/watermarks.py` (`DEFAULT_WATERMARK_RULES`).
- Each rule specifies a regex, optional font-size guardrails, and a distance-from-bottom threshold.
- To add a new pattern:
  1. Duplicate an existing `WatermarkRule` entry and adjust the regex (use `_normalize_text` behaviour as a guide—whitespace is collapsed).
  2. Tune `max_distance_from_bottom` / font sizes if the watermark sits higher on the page or uses different styling.
  3. Optionally write a regression test in `tests/test_processor.py` mirroring the new footer to guarantee it stays covered.
  4. Run `python -m pytest` and `python -m tomescrub ...` to confirm the rule behaves as expected.

## Next Steps
- Add content-aware cleaning features (redaction, annotation flattening, etc.).
- Add structured logging and richer configuration support.
