"""Password resolution helpers for encrypted PDF documents."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


PasswordMap = Dict[str, str]


def _normalise_key(key: str) -> str:
    """Normalise mapping keys to improve lookups."""
    return key.strip()


def load_password_file(path: Path) -> PasswordMap:
    """
    Parse a password mapping file.

    The format is ``key = value`` per line. ``key`` may be a full path,
    relative path or filename. Blank lines and lines starting with ``#``
    are ignored. A ``*`` key acts as a wildcard default for the file.
    """
    mapping: PasswordMap = {}
    if not path.exists():
        return mapping

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        mapping[_normalise_key(key)] = value.strip()
    return mapping


class PasswordProvider:
    """Resolve passwords using defaults, global mappings, and local hint files."""

    def __init__(
        self,
        *,
        default: Optional[str] = None,
        global_mapping: Optional[PasswordMap] = None,
        hint_filename: Optional[str] = "passwords.txt",
    ) -> None:
        self.default = default
        self.global_mapping = {k: v for k, v in (global_mapping or {}).items()}
        self.hint_filename = hint_filename
        self._hint_cache: Dict[Path, PasswordMap] = {}

    def resolve(self, pdf_path: Path) -> Optional[str]:
        """Return a password for ``pdf_path`` if one is known."""
        pdf_path = pdf_path.resolve()
        candidates = {
            _normalise_key(str(pdf_path)),
            _normalise_key(pdf_path.as_posix()),
            _normalise_key(pdf_path.name),
        }

        for candidate in candidates:
            if candidate in self.global_mapping:
                return self.global_mapping[candidate]

        if self.hint_filename:
            hint_path = pdf_path.parent / self.hint_filename
            hints = self._load_hint_file(hint_path)
            if hints:
                for candidate in candidates:
                    if candidate in hints:
                        return hints[candidate]
                if "*" in hints:
                    return hints["*"]

        return self.default

    def _load_hint_file(self, path: Path) -> PasswordMap:
        if path not in self._hint_cache:
            self._hint_cache[path] = load_password_file(path)
        return self._hint_cache[path]
