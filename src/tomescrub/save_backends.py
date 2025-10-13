"""Save backend abstractions for TomeScrub."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple

import fitz  # type: ignore

try:
    import pikepdf  # type: ignore
    from pikepdf import Name
    HAS_PIKEPDF = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pikepdf = None  # type: ignore
    Name = None  # type: ignore
    HAS_PIKEPDF = False

from .config.schema import SaveConfig

if TYPE_CHECKING:  # pragma: no cover
    from .processor import PDFCleaner
    from .pipeline import PipelineContext

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SaveOutcome:
    """Information returned by a save backend."""

    cleaned_permissions: int
    verify_time: float = 0.0


class BaseSaveBackend:
    """Common interface for save backends."""

    def __init__(self, config: SaveConfig) -> None:
        self.config = config

    def save(self, cleaner: "PDFCleaner", ctx: "PipelineContext", document: fitz.Document) -> SaveOutcome:
        """Persist the processed document and return metadata."""
        raise NotImplementedError


class PyMuPDFSaveBackend(BaseSaveBackend):
    """Default saver that delegates to PyMuPDF."""

    def save(self, cleaner: "PDFCleaner", ctx: "PipelineContext", document: fitz.Document) -> SaveOutcome:
        ctx.output_path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {
            "encryption": fitz.PDF_ENCRYPT_NONE,
            "garbage": self.config.garbage,
            "deflate": self.config.deflate,
            "linear": bool(self.config.linearize),
        }
        document.save(ctx.output_path, **save_kwargs)

        verify_start = perf_counter()
        with fitz.open(ctx.output_path) as cleaned:
            permissions = cleaned.permissions
        verify_time = perf_counter() - verify_start
        return SaveOutcome(cleaned_permissions=permissions, verify_time=verify_time)


class GhostscriptSaver(BaseSaveBackend):
    """Ghostscript-based save backend."""

    def __init__(self, config: SaveConfig) -> None:
        super().__init__(config)
        self._executable: Optional[str] = None

    def _build_command(
        self,
        executable: str,
        input_path: Path,
        output_path: Path,
        linearize_with_gs: bool,
    ) -> List[str]:
        cfg = self.config
        fonts = cfg.fonts
        images = cfg.images
        links = cfg.links
        misc = cfg.misc

        bool_str = lambda value: "true" if value else "false"
        cmd: List[str] = [
            executable,
            "-sDEVICE=pdfwrite",
            "-dBATCH",
            "-dNOPAUSE",
            "-dSAFER",
            f"-sOutputFile={output_path}",
        ]

        if cfg.pdf_version:
            cmd.append(f"-dCompatibilityLevel={cfg.pdf_version}")

        cmd.extend(
            [
                f"-dDetectDuplicateImages={bool_str(misc.detect_duplicate_images)}",
                f"-dPrinted={bool_str(not links.preserve_links)}",
                f"-dSubsetFonts={bool_str(fonts.subset)}",
                f"-dEmbedAllFonts={bool_str(fonts.embed_all)}",
                f"-dCompressFonts={bool_str(fonts.compress)}",
                "-dAutoFilterColorImages=false",
                "-dAutoFilterGrayImages=false",
                "-dColorImageDownsampleType=/Bicubic",
                "-dGrayImageDownsampleType=/Bicubic",
                "-dMonoImageDownsampleType=/Subsample",
                f"-dColorImageResolution={images.color_target_ppi}",
                f"-dGrayImageResolution={images.gray_target_ppi}",
                f"-dMonoImageResolution={images.mono_target_ppi}",
                f"-dColorImageDownsampleThreshold={images.threshold_factor}",
                f"-dGrayImageDownsampleThreshold={images.threshold_factor}",
                f"-dMonoImageDownsampleThreshold={images.threshold_factor}",
            ]
        )

        if linearize_with_gs or misc.fast_web_view or cfg.linearize:
            cmd.append("-dFastWebView=true")

        if misc.leave_color_unchanged:
            cmd.extend(
                [
                    "-sColorConversionStrategy=/LeaveColorUnchanged",
                    "-dColorConversionStrategy=/LeaveColorUnchanged",
                ]
            )

        photo_filter_map = {
            "jpeg": "/DCTEncode",
            "jpx": "/JPXEncode",
            "zip": "/FlateEncode",
        }
        lineart_filter_map = {
            "zip": "/FlateEncode",
            "fax": "/CCITTFaxEncode",
        }
        color_filter = photo_filter_map.get(images.photo_compression, "/FlateEncode")
        cmd.append(f"-sColorImageFilter={color_filter}")
        cmd.append(f"-sGrayImageFilter={color_filter}")
        cmd.append(f"-sMonoImageFilter={lineart_filter_map.get(images.lineart_compression, '/FlateEncode')}")

        if images.photo_compression == "jpeg" and images.jpeg.qfactor is not None:
            jpeg = images.jpeg
            color_params = [f"/QFactor {jpeg.qfactor}"]
            if jpeg.blend is not None:
                color_params.append(f"/Blend {jpeg.blend}")
            if jpeg.h_samples:
                h_samples = " ".join(str(val) for val in jpeg.h_samples)
                color_params.append(f"/HSamples [{h_samples}]")
            if jpeg.v_samples:
                v_samples = " ".join(str(val) for val in jpeg.v_samples)
                color_params.append(f"/VSamples [{v_samples}]")
            params = " ".join(color_params)
            distiller = f"<< /ColorImageDict << {params} >> /GrayImageDict << {params} >> >> setdistillerparams"
            cmd.extend(["-c", distiller])

        if cfg.ghostscript.extra:
            cmd.extend([str(extra) for extra in cfg.ghostscript.extra])
        cmd.extend(["-f", str(input_path)])
        return cmd

    def _resolve_executable(self) -> str:
        if self._executable:
            return self._executable
        preferred = self.config.ghostscript.exe.strip() if self.config.ghostscript.exe else ""
        candidates: List[str] = []
        if preferred:
            candidates.append(preferred)
        if os.name == "nt":
            candidates.extend(["gswin64c", "gswin32c", "gs"])
        else:
            candidates.append("gs")
        for candidate in candidates:
            resolved = shutil.which(candidate)
            if resolved:
                self._executable = resolved
                return resolved
        raise FileNotFoundError(
            "Ghostscript executable not found. Install Ghostscript or set save.ghostscript.exe."
        )

    def _run_ghostscript(self, input_path: Path, output_path: Path, linearize_with_gs: bool) -> SaveOutcome:
        executable = self._resolve_executable()
        command = self._build_command(executable, input_path, output_path, linearize_with_gs)
        redacted = [
            "<input>" if str(input_path) in arg else "<output>" if str(output_path) in arg else arg
            for arg in command
        ]
        LOGGER.debug("Running Ghostscript: %s", " ".join(redacted))

        start = perf_counter()
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - subprocess errors are rare
            LOGGER.error("Ghostscript failed: %s", exc.stderr)
            raise RuntimeError(f"Ghostscript failed: {exc.stderr or exc}") from exc
        gs_elapsed = perf_counter() - start

        verify_start = perf_counter()
        with fitz.open(output_path) as cleaned:
            permissions = cleaned.permissions
        verify_time = perf_counter() - verify_start
        LOGGER.debug("Ghostscript completed in %.3fs (verify %.3fs)", gs_elapsed, verify_time)
        return SaveOutcome(cleaned_permissions=permissions, verify_time=verify_time)

    def save(self, cleaner: "PDFCleaner", ctx: "PipelineContext", document: fitz.Document) -> SaveOutcome:
        ctx.output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="tomescrub-gs-") as tmpdir:
            temp_input = Path(tmpdir) / "input.pdf"
            document.save(temp_input, incremental=False, encryption=fitz.PDF_ENCRYPT_NONE)
            linearize_with_gs = bool(self.config.linearize or self.config.misc.fast_web_view)
            return self._run_ghostscript(temp_input, ctx.output_path, linearize_with_gs)


class QpdfLinearizer(BaseSaveBackend):
    """QPDF-based linearisation backend."""

    def __init__(self, config: SaveConfig) -> None:
        super().__init__(config)
        self._executable: Optional[str] = None

    def _resolve_executable(self) -> str:
        if self._executable:
            return self._executable
        preferred = self.config.qpdf.exe.strip() if self.config.qpdf.exe else ""
        candidates = [preferred] if preferred else []
        candidates.append("qpdf")
        for candidate in candidates:
            if not candidate:
                continue
            resolved = shutil.which(candidate)
            if resolved:
                self._executable = resolved
                return resolved
        raise FileNotFoundError("qpdf executable not found. Install qpdf or set save.qpdf.exe.")

    def _build_command(
        self,
        executable: str,
        input_path: Path,
        output_path: Path,
        linearize: bool,
    ) -> List[str]:
        cmd: List[str] = [executable]
        if linearize:
            cmd.append("--linearize")
        if self.config.pdf_version:
            cmd.append(f"--force-version={self.config.pdf_version}")
        cmd.extend(
            [
                "--object-streams=generate",
                "--compress-streams=y",
            ]
        )
        if self.config.qpdf.extra:
            cmd.extend([str(extra) for extra in self.config.qpdf.extra])
        cmd.extend([str(input_path), str(output_path)])
        return cmd

    def _run_qpdf(self, input_path: Path, output_path: Path, linearize: bool) -> SaveOutcome:
        executable = self._resolve_executable()
        command = self._build_command(executable, input_path, output_path, linearize)
        redacted = [
            "<input>" if str(input_path) in arg else "<output>" if str(output_path) in arg else arg
            for arg in command
        ]
        LOGGER.debug("Running qpdf: %s", " ".join(redacted))
        start = perf_counter()
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as exc:  # pragma: no cover
            LOGGER.error("qpdf failed: %s", exc.stderr)
            raise RuntimeError(f"qpdf failed: {exc.stderr or exc}") from exc
        qpdf_elapsed = perf_counter() - start

        verify_start = perf_counter()
        with fitz.open(output_path) as cleaned:
            permissions = cleaned.permissions
        verify_time = perf_counter() - verify_start
        LOGGER.debug("qpdf completed in %.3fs (verify %.3fs)", qpdf_elapsed, verify_time)
        return SaveOutcome(cleaned_permissions=permissions, verify_time=verify_time)

    def save(self, cleaner: "PDFCleaner", ctx: "PipelineContext", document: fitz.Document) -> SaveOutcome:
        ctx.output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="tomescrub-qpdf-") as tmpdir:
            temp_input = Path(tmpdir) / "input.pdf"
            document.save(temp_input, incremental=False, encryption=fitz.PDF_ENCRYPT_NONE)
            linearize = bool(self.config.linearize or self.config.misc.fast_web_view)
            return self._run_qpdf(temp_input, ctx.output_path, linearize)


class PikepdfCleaner(BaseSaveBackend):
    """PikePDF-based cleanup backend."""

    def __init__(self, config: SaveConfig) -> None:
        super().__init__(config)

    @property
    def enabled(self) -> bool:
        return bool(self.config.pikepdf.enabled)

    def _remove_thumbnails(self, pdf: "pikepdf.Pdf") -> bool:
        if not HAS_PIKEPDF or not self.config.misc.remove_thumbnails:
            return False
        modified = False
        for page in pdf.pages:
            if Name("/Thumb") in page.obj:
                del page.obj[Name("/Thumb")]
                modified = True
        return modified

    def _remove_bookmarks(self, pdf: "pikepdf.Pdf") -> bool:
        if self.config.links.preserve_bookmarks:
            return False
        if Name("/Outlines") in pdf.root:
            del pdf.root[Name("/Outlines")]
            return True
        return False

    def _remove_ocg_metadata(self, pdf: "pikepdf.Pdf") -> bool:
        if not self.config.layers.remove_ocg_metadata:
            return False
        if Name("/OCProperties") in pdf.root:
            del pdf.root[Name("/OCProperties")]
            return True
        return False

    def _collect_hidden_ocgs(self, pdf: "pikepdf.Pdf") -> set[Tuple[int, int]]:
        oc_props = pdf.root.get(Name("/OCProperties"))
        if not oc_props:
            return set()
        default = oc_props.get(Name("/D"))
        if not isinstance(default, pikepdf.Dictionary):
            return set()
        off = default.get(Name("/OFF"))
        if not isinstance(off, pikepdf.Array):
            return set()
        return {oc.objgen for oc in off if isinstance(oc, pikepdf.Object) and oc.indirect}

    def _resolve_oc_reference(
        self,
        page: "pikepdf.Page",
        operands: List["pikepdf.Object"],
    ) -> Optional["pikepdf.Object"]:
        if not operands:
            return None
        # Pattern [/OC /Name]
        if isinstance(operands[0], pikepdf.Name) and operands[0] == Name("/OC"):
            if len(operands) >= 2 and isinstance(operands[1], pikepdf.Name):
                props = page.obj.get(Name("/Resources"))
                if isinstance(props, pikepdf.Dictionary):
                    prop_dict = props.get(Name("/Properties"))
                    if isinstance(prop_dict, pikepdf.Dictionary) and operands[1] in prop_dict:
                        entry = prop_dict[operands[1]]
                        if isinstance(entry, pikepdf.Dictionary) and Name("/OC") in entry:
                            return entry[Name("/OC")]
                        return entry
        # Pattern <</OC ...>> BDC
        maybe_dict = operands[-1]
        if isinstance(maybe_dict, pikepdf.Dictionary) and Name("/OC") in maybe_dict:
            return maybe_dict[Name("/OC")]
        return None

    def _filter_content(self, page: "pikepdf.Page", ocgs_off: set[Tuple[int, int]]) -> bool:
        if not ocgs_off:
            return False
        try:
            content = pikepdf.ContentStream(page)
        except pikepdf.PdfError:  # pragma: no cover - corrupt content
            return False
        new_ops: List[Tuple[Iterable[pikepdf.Object], pikepdf.Name]] = []
        skip_stack: List[bool] = []
        modified = False
        for operands, operator in content.operations:
            if operator == Name("/BDC"):
                oc_ref = self._resolve_oc_reference(page, list(operands))
                skip = bool(oc_ref and oc_ref.objgen in ocgs_off)
                skip_stack.append(skip)
                if skip:
                    modified = True
                    continue
                new_ops.append((operands, operator))
            elif operator == Name("/EMC"):
                skip = skip_stack.pop() if skip_stack else False
                if skip:
                    modified = True
                    continue
                new_ops.append((operands, operator))
            else:
                if skip_stack and skip_stack[-1]:
                    modified = True
                    continue
                new_ops.append((operands, operator))
        if modified:
            page.Contents = pikepdf.ContentStream(page.pdf, new_ops)
        return modified

    def _flatten_hidden(self, pdf: "pikepdf.Pdf") -> bool:
        if not self.config.layers.flatten_hidden:
            return False
        ocgs_off = self._collect_hidden_ocgs(pdf)
        if not ocgs_off:
            return False
        modified = False
        for page in pdf.pages:
            modified |= self._filter_content(page, ocgs_off)
        return modified

    def clean_file(self, input_path: Path, output_path: Optional[Path] = None) -> Path:
        if not HAS_PIKEPDF:
            if output_path and output_path != input_path:
                shutil.copy2(input_path, output_path)
                return output_path
            return output_path or input_path

        if not self.enabled:
            if output_path and output_path != input_path:
                shutil.copy2(input_path, output_path)
                return output_path
            return output_path or input_path

        output_path = output_path or input_path
        with pikepdf.Pdf.open(input_path) as pdf:
            modified = False
            modified |= self._remove_thumbnails(pdf)
            modified |= self._flatten_hidden(pdf)
            modified |= self._remove_ocg_metadata(pdf)
            modified |= self._remove_bookmarks(pdf)
            if modified or output_path != input_path:
                pdf.save(output_path)
            elif output_path != input_path:
                shutil.copy2(input_path, output_path)
        return output_path

    def save(self, cleaner: "PDFCleaner", ctx: "PipelineContext", document: fitz.Document) -> SaveOutcome:
        ctx.output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="tomescrub-pikepdf-") as tmpdir:
            temp_input = Path(tmpdir) / "input.pdf"
            document.save(temp_input, incremental=False, encryption=fitz.PDF_ENCRYPT_NONE)
            output = self.clean_file(temp_input, ctx.output_path)
        verify_start = perf_counter()
        with fitz.open(output) as cleaned:
            permissions = cleaned.permissions
        verify_time = perf_counter() - verify_start
        return SaveOutcome(cleaned_permissions=permissions, verify_time=verify_time)


class ChainSaveBackend(BaseSaveBackend):
    """Composite backend: PikePDF pre/post, Ghostscript, and QPDF."""

    def __init__(self, config: SaveConfig) -> None:
        super().__init__(config)
        if config.pikepdf.enabled and not HAS_PIKEPDF:
            raise ModuleNotFoundError(
                "pikepdf is required for save backend 'chain' when save.pikepdf.enabled=true."
            )
        self._pikepdf = PikepdfCleaner(config) if HAS_PIKEPDF else None
        self._ghostscript = GhostscriptSaver(config)
        self._qpdf = QpdfLinearizer(config)

    def _use_qpdf(self) -> bool:
        return bool(self.config.linearize or self.config.qpdf.extra or self.config.pdf_version)

    def save(self, cleaner: "PDFCleaner", ctx: "PipelineContext", document: fitz.Document) -> SaveOutcome:
        ctx.output_path.parent.mkdir(parents=True, exist_ok=True)
        use_qpdf = self._use_qpdf()

        with tempfile.TemporaryDirectory(prefix="tomescrub-chain-") as tmpdir:
            stage_counter = 0

            def next_stage() -> Path:
                nonlocal stage_counter
                stage_counter += 1
                return Path(tmpdir) / f"stage{stage_counter}.pdf"

            current = next_stage()
            document.save(current, incremental=False, encryption=fitz.PDF_ENCRYPT_NONE)

            pikepdf_cleaner = self._pikepdf
            pikepdf_enabled = bool(pikepdf_cleaner and pikepdf_cleaner.enabled)

            if pikepdf_enabled and pikepdf_cleaner:
                current = pikepdf_cleaner.clean_file(current, next_stage())

            ghost_output = next_stage()
            linearize_with_gs = bool(self.config.linearize or self.config.misc.fast_web_view) and not use_qpdf
            self._ghostscript._run_ghostscript(current, ghost_output, linearize_with_gs)
            current = ghost_output

            if pikepdf_enabled and pikepdf_cleaner:
                if use_qpdf:
                    current = pikepdf_cleaner.clean_file(current, next_stage())
                else:
                    current = pikepdf_cleaner.clean_file(current, ctx.output_path)

            if use_qpdf:
                self._qpdf._run_qpdf(current, ctx.output_path, bool(self.config.linearize or self.config.misc.fast_web_view))
                current = ctx.output_path
            elif current != ctx.output_path:
                shutil.copy2(current, ctx.output_path)

        verify_start = perf_counter()
        with fitz.open(ctx.output_path) as cleaned:
            permissions = cleaned.permissions
        verify_time = perf_counter() - verify_start
        return SaveOutcome(cleaned_permissions=permissions, verify_time=verify_time)


def get_save_backend(config: SaveConfig) -> BaseSaveBackend:
    """Return the configured save backend implementation."""
    backend = (config.backend or "pymupdf").strip().lower()
    if backend == "pymupdf":
        return PyMuPDFSaveBackend(config)
    if backend == "ghostscript":
        return GhostscriptSaver(config)
    if backend == "qpdf":
        return QpdfLinearizer(config)
    if backend == "pikepdf":
        if not HAS_PIKEPDF:
            raise ModuleNotFoundError("pikepdf is required for save backend 'pikepdf'.")
        return PikepdfCleaner(config)
    if backend == "chain":
        return ChainSaveBackend(config)
    raise ValueError(f"Unsupported save backend: {backend}")
