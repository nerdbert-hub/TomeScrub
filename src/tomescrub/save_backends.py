"""Save backend abstractions for TomeScrub."""

from __future__ import annotations

from dataclasses import dataclass
import logging
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


def detect_has_ocg(path: Path) -> bool:
    """Return True when the PDF appears to contain optional content groups."""
    if HAS_PIKEPDF:
        try:
            with pikepdf.Pdf.open(path) as pdf:  # type: ignore[attr-defined]
                root = getattr(pdf, "Root", None)
                if root is None and hasattr(pdf, "trailer"):
                    root = pdf.trailer.get(Name("/Root"))
                if isinstance(root, pikepdf.Dictionary):
                    return Name("/OCProperties") in root
        except Exception:  # pragma: no cover - fall back to heuristic
            pass
    try:
        with path.open("rb") as handle:
            chunk = handle.read(65536)
            return b"/OCProperties" in chunk
    except OSError:  # pragma: no cover - IO error
        return False
    return False


def _should_run_qpdf(config: SaveConfig) -> bool:
    return bool(config.linearize or config.qpdf.extra or config.pdf_version)


def _get_root(pdf: "pikepdf.Pdf"):
    root = getattr(pdf, "Root", None)
    if root is None and hasattr(pdf, "trailer"):
        try:
            root = pdf.trailer.get(Name("/Root"))
        except Exception:
            root = None
    return root


def _save_with_pymupdf(input_path: Path, output_path: Path, config: SaveConfig) -> None:
    with fitz.open(input_path) as doc:
        doc.save(
            output_path,
            encryption=fitz.PDF_ENCRYPT_NONE,
            garbage=config.garbage,
            deflate=config.deflate,
            linear=bool(config.linearize),
        )


def _get_root(pdf: "pikepdf.Pdf"):
    root = getattr(pdf, "Root", None)
    if root is None and hasattr(pdf, "trailer"):
        try:
            root = pdf.trailer.get(Name("/Root"))
        except Exception:
            root = None
    return root


def _bool_to_gs(flag: bool) -> str:
    """Return a Ghostscript-friendly boolean string."""
    return "true" if flag else "false"


def _finalize_outcome(path: Path) -> SaveOutcome:
    verify_start = perf_counter()
    with fitz.open(path) as cleaned:
        permissions = cleaned.permissions
    verify_time = perf_counter() - verify_start
    return SaveOutcome(cleaned_permissions=permissions, verify_time=verify_time)


def _apply_qpdf_if_configured(config: SaveConfig, input_path: Path, output_path: Path) -> Path:
    if not _should_run_qpdf(config):
        if input_path != output_path:
            shutil.copy2(input_path, output_path)
        return output_path

    linear_flag = bool(config.linearize or config.misc.fast_web_view)
    qpdf = QpdfLinearizer(config)

    if input_path == output_path:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            temp_path = Path(tmp.name)
        try:
            qpdf._run_qpdf(input_path, temp_path, linear_flag)
            shutil.move(temp_path, output_path)
        finally:
            if 'temp_path' in locals() and temp_path.exists():
                temp_path.unlink(missing_ok=True)
        return output_path

    qpdf._run_qpdf(input_path, output_path, linear_flag)
    return output_path


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
        return _finalize_outcome(ctx.output_path)




class GhostscriptDistiller(BaseSaveBackend):
    """Invoke Ghostscript pdfwrite to re-save PDFs with distiller parameters."""

    def __init__(self, config: SaveConfig) -> None:
        super().__init__(config)
        self._executable: Optional[str] = None

    def _resolve_executable(self) -> str:
        if self._executable:
            return self._executable
        cfg = self.config.ghostscript
        candidates: List[str] = []
        preferred = cfg.exe.strip()
        if preferred:
            preferred_path = Path(preferred).expanduser()
            if preferred_path.exists():
                self._executable = str(preferred_path)
                return self._executable
            candidates.append(preferred)
        candidates.extend(["gswin64c", "gswin32c", "gs"])
        for candidate in candidates:
            resolved = shutil.which(candidate)
            if resolved:
                self._executable = resolved
                return resolved
        raise FileNotFoundError("Ghostscript executable not found. Install Ghostscript or set save.ghostscript.exe.")

    def _build_channel_args(
        self,
        prefix: str,
        channel: "GhostscriptChannelConfig",
        resolution: int,
    ) -> tuple[List[str], bool, bool]:
        args = [
            f"-dDownsample{prefix}Images={_bool_to_gs(channel.downsample)}",
            f"-d{prefix}ImageResolution={resolution}",
            f"-d{prefix}ImageDownsampleType=/{channel.downsample_type.capitalize()}",
            f"-d{prefix}ImageDownsampleThreshold={channel.downsample_threshold}",
        ]
        auto_filter = channel.auto_filter if channel.auto_filter is not None else False
        encode = channel.encode if channel.encode is not None else True
        args.append(f"-dAutoFilter{prefix}Images={_bool_to_gs(auto_filter)}")
        args.append(f"-dEncode{prefix}Images={_bool_to_gs(encode)}")
        return args, auto_filter, encode

    @staticmethod
    def _format_samples(values: List[int]) -> str:
        return "[ " + " ".join(str(v) for v in values) + " ]"

    def _build_distiller_params(self) -> Optional[str]:
        images = self.config.images
        if images.photo_compression != "jpeg":
            return None
        jpeg_cfg = images.jpeg
        parts: List[str] = []
        color_entries: List[str] = []
        gray_entries: List[str] = []
        if jpeg_cfg.qfactor is not None:
            color_entries.append(f"/QFactor {jpeg_cfg.qfactor:.6g}")
            gray_entries.append(f"/QFactor {jpeg_cfg.qfactor:.6g}")
        if jpeg_cfg.blend is not None:
            color_entries.append(f"/Blend {jpeg_cfg.blend}")
            gray_entries.append(f"/Blend {jpeg_cfg.blend}")
        if jpeg_cfg.h_samples:
            color_entries.append(f"/HSamples {self._format_samples(jpeg_cfg.h_samples)}")
            gray_entries.append(f"/HSamples {self._format_samples(jpeg_cfg.h_samples)}")
        if jpeg_cfg.v_samples:
            color_entries.append(f"/VSamples {self._format_samples(jpeg_cfg.v_samples)}")
            gray_entries.append(f"/VSamples {self._format_samples(jpeg_cfg.v_samples)}")
        if not color_entries and not gray_entries:
            return None
        if color_entries:
            block = " ".join(color_entries)
            parts.append(f"/ColorImageDict << {block} >>")
            parts.append(f"/ColorACSImageDict << {block} >>")
        if gray_entries:
            block = " ".join(gray_entries)
            parts.append(f"/GrayImageDict << {block} >>")
            parts.append(f"/GrayACSImageDict << {block} >>")
        if not parts:
            return None
        return "<< " + " ".join(parts) + " >> setdistillerparams"

    def _build_command(self, executable: str, input_path: Path, output_path: Path) -> List[str]:
        cfg = self.config
        gs_cfg = cfg.ghostscript
        images = cfg.images
        misc = cfg.misc

        command: List[str] = [
            executable,
            "-dSAFER",
            "-dNOPAUSE",
            "-dBATCH",
            "-sDEVICE=pdfwrite",
            f"-dCompatibilityLevel={gs_cfg.compatibility_level}",
            f"-dDetectDuplicateImages={_bool_to_gs(misc.detect_duplicate_images)}",
            f"-dFastWebView={_bool_to_gs(misc.fast_web_view)}",
            f"-dMaxInlineImageSize={gs_cfg.max_inline_image_size}",
        ]
        if misc.leave_color_unchanged:
            command.append("-sColorConversionStrategy=LeaveColorUnchanged")
        if gs_cfg.pass_through_jpeg_images is not None:
            command.append(f"-dPassThroughJPEGImages={_bool_to_gs(gs_cfg.pass_through_jpeg_images)}")
        if gs_cfg.pass_through_jpx_images is not None:
            command.append(f"-dPassThroughJPXImages={_bool_to_gs(gs_cfg.pass_through_jpx_images)}")

        color_args, color_auto_filter, _ = self._build_channel_args("Color", gs_cfg.color, images.color_target_ppi)
        gray_args, gray_auto_filter, _ = self._build_channel_args("Gray", gs_cfg.gray, images.gray_target_ppi)
        mono_args, mono_auto_filter, _ = self._build_channel_args("Mono", gs_cfg.mono, images.mono_target_ppi)
        command.extend(color_args)
        command.extend(gray_args)
        command.extend(mono_args)

        if not color_auto_filter:
            if images.photo_compression == "jpeg":
                command.append("-dColorImageFilter=/DCTEncode")
            else:
                command.append("-dColorImageFilter=/FlateEncode")
        if not gray_auto_filter:
            if images.photo_compression == "jpeg":
                command.append("-dGrayImageFilter=/DCTEncode")
            else:
                command.append("-dGrayImageFilter=/FlateEncode")
        if not mono_auto_filter:
            if images.lineart_compression == "fax":
                command.append("-dMonoImageFilter=/CCITTFaxEncode")
            else:
                command.append("-dMonoImageFilter=/FlateEncode")

        if gs_cfg.extra:
            command.extend(str(arg) for arg in gs_cfg.extra)

        command.extend(["-o", str(output_path)])
        distiller_code = self._build_distiller_params()
        if distiller_code:
            command.extend(["-c", distiller_code])
        command.extend(["-f", str(input_path)])
        return command

    def process_file(self, input_path: Path, output_path: Path) -> Path:
        executable = self._resolve_executable()
        command = self._build_command(executable, input_path, output_path)
        redacted = [
            "<input>" if str(input_path) in arg else "<output>" if str(output_path) in arg else arg
            for arg in command
        ]
        LOGGER.debug("Running Ghostscript: %s", " ".join(redacted))
        start = perf_counter()
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as exc:  # pragma: no cover
            LOGGER.error("Ghostscript failed: %s", exc.stderr)
            raise RuntimeError(f"Ghostscript failed: {exc.stderr or exc}") from exc
        LOGGER.debug("Ghostscript completed in %.3fs", perf_counter() - start)
        return output_path

    def save(self, cleaner: "PDFCleaner", ctx: "PipelineContext", document: fitz.Document) -> SaveOutcome:
        ctx.output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="tomescrub-ghostscript-") as tmpdir:
            temp_input = Path(tmpdir) / "input.pdf"
            document.save(temp_input, incremental=False, encryption=fitz.PDF_ENCRYPT_NONE)
            self.process_file(temp_input, ctx.output_path)
        return _finalize_outcome(ctx.output_path)


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
            cmd.append("--object-streams=generate")
        else:
            cmd.append("--object-streams=preserve")
        if self.config.pdf_version:
            cmd.append(f"--force-version={self.config.pdf_version}")
        cmd.append("--compress-streams=y")
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
        root = _get_root(pdf) if HAS_PIKEPDF else None
        if isinstance(root, pikepdf.Dictionary) and Name("/Outlines") in root:
            del root[Name("/Outlines")]
            return True
        return False

    def _remove_ocg_metadata(self, pdf: "pikepdf.Pdf") -> bool:
        if not self.config.layers.remove_ocg_metadata:
            return False
        root = _get_root(pdf) if HAS_PIKEPDF else None
        if isinstance(root, pikepdf.Dictionary) and Name("/OCProperties") in root:
            del root[Name("/OCProperties")]
            return True
        return False

    def _collect_hidden_ocgs(self, pdf: "pikepdf.Pdf") -> set[Tuple[int, int]]:
        root = _get_root(pdf) if HAS_PIKEPDF else None
        if not isinstance(root, pikepdf.Dictionary):
            return set()
        oc_props = root.get(Name("/OCProperties"))
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
    """Composite backend: optional PikePDF passes followed by QPDF linearisation."""

    def __init__(self, config: SaveConfig) -> None:
        super().__init__(config)
        if config.pikepdf.enabled and not HAS_PIKEPDF:
            raise ModuleNotFoundError(
                "pikepdf is required for save backend 'chain' when save.pikepdf.enabled=true."
            )
        self._pikepdf = PikepdfCleaner(config) if HAS_PIKEPDF else None
        self._ghostscript = GhostscriptDistiller(config) if config.ghostscript.enabled else None
        self._qpdf = QpdfLinearizer(config)

    def save(self, cleaner: "PDFCleaner", ctx: "PipelineContext", document: fitz.Document) -> SaveOutcome:
        ctx.output_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="tomescrub-chain-") as tmpdir:
            stage_counter = 0

            def next_stage(name: str) -> Path:
                nonlocal stage_counter
                stage_counter += 1
                return Path(tmpdir) / f"{stage_counter:02d}_{name}.pdf"

            current_path = next_stage("base")
            document.save(current_path, incremental=False, encryption=fitz.PDF_ENCRYPT_NONE)

            has_ocg = detect_has_ocg(current_path) if self.config.layers.preserve_layers else False
            pikepdf_cleaner = self._pikepdf if (self._pikepdf and self._pikepdf.enabled) else None

            if pikepdf_cleaner and self.config.layers.preserve_layers and has_ocg:
                LOGGER.warning(
                    "Detected optional content groups; preserving layers by skipping PikePDF adjustments."
                )
                pikepdf_cleaner = None

            if pikepdf_cleaner:
                current_path = pikepdf_cleaner.clean_file(current_path, next_stage("pike_pre"))

            if pikepdf_cleaner:
                current_path = pikepdf_cleaner.clean_file(current_path, next_stage("pike_post"))

            if self._ghostscript:
                current_path = self._ghostscript.process_file(current_path, next_stage("ghostscript"))

            final_path = _apply_qpdf_if_configured(self.config, current_path, ctx.output_path)

        return _finalize_outcome(final_path)


def get_save_backend(config: SaveConfig) -> BaseSaveBackend:
    """Return the configured save backend implementation."""
    backend = (config.backend or "pymupdf").strip().lower()
    if backend == "pymupdf":
        return PyMuPDFSaveBackend(config)
    if backend == "qpdf":
        return QpdfLinearizer(config)
    if backend == "pikepdf":
        if not HAS_PIKEPDF:
            raise ModuleNotFoundError("pikepdf is required for save backend 'pikepdf'.")
        return PikepdfCleaner(config)
    if backend == "ghostscript":
        return GhostscriptDistiller(config)
    if backend == "chain":
        return ChainSaveBackend(config)
    raise ValueError(f"Unsupported save backend: {backend}")
