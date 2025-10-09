# Changelog

All notable changes to this project are documented below.

## [0.4.0] - 2025-10-08
### Added
- Configurable cleaning toggles (per-metadata/image/hidden-text, text extraction) and dirty-write guard.
- Watermark performance controls (bottom clip band, early exit, page limits) with faster detection pipeline.
- Save-time tuning (linearize, garbage, deflate) and multiprocessing support with batching knobs.
- NDJSON run log enhancements capturing config snapshot, per-file records, and quiet mode.
- Updated example configuration and README documentation for the new controls.

## [0.3.0] - 2025-10-08
### Added
- Timing instrumentation across PDFCleaner, per-step metrics, and change detection flagging.
- Rich CLI progress output with spinner heartbeat, aligned columns, run summary legend, and structured run logging.
- Run log configuration hooks and regression coverage for logging and statistics.

## [0.2.0] - 2025-10-08
### Added
- Centralised TOML configuration loader with defaults, env overrides, and CLI setters.
- Documentation updates and example config illustrating override usage.

## [0.1.1] - 2025-10-07
### Changed
- Renamed the project to TomeScrub and aligned package references.
- Expanded .gitignore rules to cover local exploration/input directories.

## [0.1.0] - 2025-10-07
### Added
- Initial PDF cleaner implementation using PyMuPDF with watermark removal, metadata sanitisation, and CLI entry point.
- Project scaffolding, tests, and base configuration assets.
