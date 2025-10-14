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
- Optional NDJSON run logging with per-file timings and configurable verbosity.
- Rich CLI with subcommands (`process`, `dry-run`, `rules test`, `print-config`, `stats`) and live progress that surfaces per-stage summaries at the end of each run.
- Per-file multiprocessing with configurable worker counts and batching.
- Dirty-write guard so unchanged documents are skipped without losing mirrors.
- CLI entry point for quick command-line usage over files or entire folders.
- Pytest harness seeded with regression tests for the core behaviours.
- Rich `DocumentProcessingResult` metadata (original vs cleaned permissions, encryption flag, extracted text).

## Prerequisites
TomeScrub works out of the box with PyMuPDF. For the bundled "web" profile we recommend adding QPDF so

1. **QPDF** (optional, but recommended for the `web` profile)
   - **Windows:** install from <https://qpdf.sourceforge.io/> (or `choco install qpdf`), then ensure `qpdf.exe` is on `PATH` or set `save.qpdf.exe`.
   - **macOS:** `brew install qpdf`
   - **Linux:** `sudo apt install qpdf`
   - Verify with `qpdf --version`.


If QPDF is unavailable TomeScrub simply writes the PyMuPDF output and skips the linearisation step.

## Step-by-Step Setup

Follow these steps on a fresh machine:

1. Install Python 3.11 or newer and ensure `python` (or `py`) resolves on your shell `PATH`.
2. Clone this repository (or download a release ZIP) and open a terminal in the project root.
3. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows PowerShell
   source .venv/bin/activate  # POSIX shells
   ```
4. Install the Python dependencies and register TomeScrub in editable mode:
   ```bash
   pip install -r requirements.txt
   python -m pip install --user -e .
   ```
5. (Optional) Install QPDF using the instructions above, then confirm with:
   ```bash
   qpdf --version
   ```
   For portable installs you can point the config at an explicit executable with
   `--set save.qpdf.exe="C:/path/to/qpdf.exe"`.
6. Run a quick smoke test to make sure everything is wired correctly:
   ```bash
   python -m pytest tests/test_save_backends.py -k chain --maxfail=1
   ```
7. Process a sample file or directory:
   ```bash
   python -m tomescrub _unprocessed --output-dir _processed
   ```
   Replace `_unprocessed` with your input folder. TomeScrub creates `_processed` on first use.

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
python -m tomescrub --profile web path/to/file.pdf --output-dir _processed
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

### Capture run summaries

```bash
python -m tomescrub _unprocessed --output-dir _processed --run-log --run-log-path _processed/run_log.ndjson
```

`--run-log` writes an NDJSON record for each run (including per-file timings). Override the destination with `--run-log-path` and suppress the success note with `--run-log-quiet`.

### Subcommands

The CLI is organised into subcommands; `process` remains the default, so existing invocations still work (`python -m tomescrub input.pdf`). Additional commands:

- `process` — Clean files and write outputs (default when no command is supplied).
- `dry-run` — Execute the full detection pipeline and report results without writing cleaned PDFs.
- `rules test` — Evaluate configured watermark rules against sample text or a specific PDF page clip:

  ```bash
  python -m tomescrub rules test --text "Downloaded by Jane" --pdf sample.pdf --page 0
  ```

- `print-config` — Display the final configuration after applying environment variables and CLI overrides.
- `stats` — Summarise one or more runs from `run_log.ndjson` (supports `--latest` to restrict output to the most recent entry).

Both `process` and `dry-run` honour the existing CLI flags for configuration overrides, passwords, and run logging.

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
- Key options:
  - `io.output_dir`, `io.overwrite_existing`, `io.skip_unchanged`: control where outputs land and whether unchanged PDFs are rewritten.
  - `io.run_log.enabled`, `io.run_log.path`, `io.run_log.quiet`: configure NDJSON run summaries and console noise.
  - `clean.strip_document_metadata`, `clean.strip_image_metadata`, `clean.remove_hidden_text`, `clean.hidden_text_alpha_threshold`, `clean.extract_text`: fine-grained sanitisation controls.
  - `passwords.default`, `passwords.hint_filename`, `passwords.password_file`: control password discovery.
  - `watermarks.enabled`, `watermarks.scan_mode`, `watermarks.clip_bottom_mm`, `watermarks.stop_after_first`, `watermarks.max_pages`, `watermarks.rules`: tune watermark checks.
  - `performance.processes`, `performance.batch_size`: size the per-file worker pool. TomeScrub spins up a `ProcessPoolExecutor` so each worker opens PDFs independently (PyMuPDF objects stay process-local); batching limits how many documents are dispatched to the pool at a time.
  - `save.linearize`, `save.garbage`, `save.deflate`: match save-time performance/compatibility requirements.
  - Profiles live under `configs/profiles/` (e.g. `fast`, `strict`, `web`). Select one with `--profile <name>` or the `TOMESCRUB__PROFILE` environment variable. The bundled `web` profile keeps metadata hygiene, targets 180 PPI imagery with high-quality JPEG compression, forces PDF 1.7 output, and leaves linearisation off--ideal for online distribution.
  - Profiles in `configs/profiles/` (e.g. `fast`, `strict`, `web`) encapsulate curated settings. Select one with `--profile <name>` (or `TOMESCRUB__PROFILE=<name>`). The bundled `web` profile targets 180 PPI imagery, uses JPEG compression, forces PDF 1.7, and leaves linearisation off--ideal for online sharing.
- Example configuration (`configs/example.toml`):

````
[io]
output_dir = "_processed"
overwrite_existing = false
skip_unchanged = true

  [io.run_log]
  enabled = true
  path = "_processed/run_log.ndjson"
  quiet = false

[clean]
strip_document_metadata = true
strip_image_metadata = true
remove_hidden_text = true
hidden_text_alpha_threshold = 16
extract_text = true

[passwords]
default = ""
hint_filename = ""
password_file = ""

[watermarks]
enabled = true
clip_bottom_mm = 12.0
stop_after_first = true
max_pages = 2

[[watermarks.rules]]
name = "download_notice"
pattern = "^Downloaded by\\s+.+?\\s+on\\s+.+?\\.\\\\s*Unauthorized distribution prohibited\\\\.?$"
ignore_case = true
max_font_size = 14.0
max_distance_from_bottom = 140.0
fonts = ["helv", "helvetica", "helvetica-bold"]

[[watermarks.rules]]
name = "order_reference"
pattern = ".+\\(Order #\\d+\\)$"
max_font_size = 14.0
max_distance_from_bottom = 140.0
fonts = ["helv", "helvetica", "helvetica-bold"]

[performance]
processes = 4
batch_size = 4

[save]
linearize = false
garbage = 4
deflate = true
````

- CLI overrides support both friendly switches (e.g. `--skip-existing`, `--no-password-hints`) and dotted assignments (`--set clean.sanitize_metadata=false`).

### Run Log Output
- Enabling `io.run_log.enabled` appends an NDJSON document after each run. Every processed file entry now carries per-stage timings in milliseconds alongside the existing second-resolution `step_timings` block.
- Top-level fields mirror the primary pipeline stages: `open_ms`, `detect_password_ms`, `sanitize_ms`, `hidden_text_ms`, `watermark_ms`, `save_ms`, and `extract_ms`.
- The same values are grouped under `stage_timings_ms` for consumers that prefer a single structured object.
- File size metrics are included for each entry via `original_size_bytes`, `output_size_bytes`, and `size_delta_bytes` so you can spot growth or shrinkage quickly.
- `step_timings` remains for backwards compatibility (seconds precision, including the `total` duration).

## Metadata & Hidden Text Sanitisation
- Each PDF run through the cleaner has its Info/XMP metadata cleared and embedded image metadata stripped.
- Text spans rendered with zero opacity (`alpha==0`) are removed automatically; this mirrors Acrobat's "Remove Hidden Information -> Hidden Text".
- Bookmarks, annotations, vector artwork, and visible imagery remain intact. We intentionally skip Acrobat's "Deleted or cropped content" / "Overlapping objects" options to avoid bloat.
- To tune hidden-text detection globally, set `hidden_text_alpha_threshold` in your TomeScrub configuration or override `remove_hidden_text` directly for bespoke behaviour.

## Watermark Rules
- Declarative watermark patterns now live in configuration (`watermarks.rules` in TOML files) and are compiled at runtime.
- `watermarks.scan_mode` toggles between full-page scanning (`"full"`) and a restricted footer pass (`"bottom"`). When using `"bottom"`, set `watermarks.clip_bottom_mm` to the band height (in millimetres) to minimise text parsing work.
- Each rule still supports regex patterns, case sensitivity, font-size guardrails, and a distance-from-bottom threshold -- identical to the previous hard-coded structures.
- To add or adjust a rule, edit your config and (optionally) add a matching regression test in `tests/test_processor.py`. Restart or reload your process to pick up the changes.

## Next Steps
- Add content-aware cleaning features (redaction, annotation flattening, etc.).
- Build analysis tooling around the NDJSON run log (dashboards, trend reports).
