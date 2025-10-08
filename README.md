# TomeScrub

Tooling scaffold for cleaning PDF documents with [PyMuPDF](https://pymupdf.readthedocs.io/).

## Features
- Package-first layout under `src/` for simple installation and reuse.
- TomeScrub utilities encapsulating metadata stripping, text extraction, and saving workflow.
- Password-aware processing that unlocks encrypted PDFs before saving unprotected copies and reports how many were unlocked per run.
- Rule-based removal of page footer watermarks (e.g., `Downloaded by ...` or `Name (Order #123)`) before writing the cleaned PDF.
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

Need a detailed walkthrough? See `docs/USAGE.txt` for a step-by-step guide that covers
environment setup, configuration layering, and extension tips.

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

````
TomeScrub/
|-- configs/
|   |-- example.toml
|   `-- profiles/
|       |-- fast.toml
|       `-- strict.toml
|-- README.md
|-- requirements.txt
|-- pyproject.toml
|-- src/
|   `-- tomescrub/
|       |-- __init__.py
|       |-- __main__.py
|       |-- cli.py
|       |-- config/
|       |   |-- __init__.py
|       |   |-- defaults.toml
|       |   |-- loader.py
|       |   `-- schema.py
|       |-- passwords.py
|       |-- processor.py
|       |-- sanitizer.py
|       `-- watermarks.py
`-- tests/
    |-- __init__.py
    `-- test_processor.py
````

## Configuration
- TomeScrub loads configuration with the following precedence (lowest -> highest): bundled defaults (`src/tomescrub/config/defaults.toml`), the first discovered config file (`--config`, `./tomescrub.toml`, or `%APPDATA%/tomescrub/config.toml`), environment variables prefixed with `TOMESCRUB__`, and finally explicit CLI overrides (`--set path=value` plus dedicated flags such as `--output-dir`).
- All settings are defined in `src/tomescrub/config/schema.py` and exposed through the `Config` model. Load them in code via `tomescrub.load_config(path, overrides)`.
- Example configuration (`configs/example.toml`):

````
[io]
output_dir = "_processed"
overwrite_existing = false

[clean]
sanitize_metadata = true
hidden_text_alpha_threshold = 16

[passwords]
default = ""
hint_filename = ""
password_file = ""

[[watermarks.rules]]
name = "download_notice"
pattern = "^Downloaded by\s+.+?\s+on\s+.+?\.\\s*Unauthorized distribution prohibited\.?$"
ignore_case = true
max_font_size = 14.0
max_distance_from_bottom = 140.0
fonts = ["helv", "helvetica", "helvetica-bold"]

[[watermarks.rules]]
name = "order_reference"
pattern = ".+\(Order #\d+\)$"
max_font_size = 14.0
max_distance_from_bottom = 140.0
fonts = ["helv", "helvetica", "helvetica-bold"]
````

- Override individual fields without editing files using environment variables such as `TOMESCRUB__IO__OUTPUT_DIR=build/_processed` or `TOMESCRUB__PASSWORDS__HINT_FILENAME=none`.
- CLI overrides support both friendly switches (e.g. `--skip-existing`, `--no-password-hints`) and dotted assignments (`--set clean.sanitize_metadata=false`).

## Metadata & Hidden Text Sanitisation
- Each PDF run through the cleaner has its Info/XMP metadata cleared and embedded image metadata stripped.
- Text spans rendered with zero opacity (`alpha==0`) are removed automatically; this mirrors Acrobat's "Remove Hidden Information -> Hidden Text".
- Bookmarks, annotations, vector artwork, and visible imagery remain intact. We intentionally skip Acrobat's "Deleted or cropped content" / "Overlapping objects" options to avoid bloat.
- To tune hidden-text detection globally, set `hidden_text_alpha_threshold` in your TomeScrub configuration or override `remove_hidden_text` directly for bespoke behaviour.

## Watermark Rules
- Declarative watermark patterns now live in configuration (`watermarks.rules` in TOML files) and are compiled at runtime.
- Each rule still supports regex patterns, case sensitivity, font-size guardrails, and a distance-from-bottom threshold -- identical to the previous hard-coded structures.
- To add or adjust a rule, edit your config and (optionally) add a matching regression test in `tests/test_processor.py`. Restart or reload your process to pick up the changes.

## Next Steps
- Add content-aware cleaning features (redaction, annotation flattening, etc.).
- Add structured logging and richer configuration support.
