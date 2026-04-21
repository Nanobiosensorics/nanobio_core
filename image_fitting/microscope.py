from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import time

import cv2
import numpy as np

from nanobio_core.alignment.pmc import find_translation_pmc
from nanobio_core.alignment.stochastic import find_translation_stochastic
from nanobio_core.alignment.utils import calculate_microscope_cell_centroids
from nanobio_core.epic_cardio.defs import WELL_NAMES

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
WellFileSelection = Dict[str, Dict[str, str]]


def _emit_progress(progress_callback: Optional[Callable[[int, int, str], None]], processed: int, total: int, status: str) -> None:
    if progress_callback is None:
        return
    progress_callback(int(processed), int(total), str(status))


def _log(msg: str, log_callback: Optional[Callable[[str], None]] = None) -> None:
    print(f"[microscope-import] {msg}", flush=True)
    if log_callback is not None:
        log_callback(msg)


def _read_cv2_any_path(
    path: Path,
    flags: int,
    label: str,
    log_callback: Optional[Callable[[str], None]] = None,
):
    image = cv2.imread(str(path), flags)
    if image is not None or not path.exists():
        return image
    _log(f"{label} direct read failed, retrying via imdecode: {path}", log_callback)
    try:
        payload = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        _log(f"{label} fallback read failed for {path}: {exc}", log_callback)
        return None
    if payload.size == 0:
        _log(f"{label} fallback payload is empty: {path}", log_callback)
        return None
    image = cv2.imdecode(payload, flags)
    if image is None:
        _log(f"{label} fallback decode returned None: {path}", log_callback)
    return image


@dataclass
class MicroscopeImportData:
    image: np.ndarray
    mask: np.ndarray
    centroids: np.ndarray
    centroid_labels: np.ndarray
    image_path: str
    mask_path: str


@dataclass
class MicroscopeAlignmentResult:
    method: str
    translation: np.ndarray
    scale: int
    projected_centroids: np.ndarray
    projected_seg_indices: np.ndarray


@dataclass(frozen=True)
class MicroscopeModel:
    microns_per_pixel: float

    def pixel_to_um(self, pixels):
        return np.asarray(pixels, dtype=np.float32) * float(self.microns_per_pixel)

    def um_to_pixel(self, microns):
        return np.asarray(microns, dtype=np.float32) / float(self.microns_per_pixel)

    def apply_distortion(self, coords: np.ndarray) -> np.ndarray:
        return np.asarray(coords, dtype=np.float32)


def _load_image(path: Path, log_callback: Optional[Callable[[str], None]] = None) -> Optional[np.ndarray]:
    _log(f"loading image: {path}", log_callback)
    t0 = time.perf_counter()
    image = _read_cv2_any_path(path, cv2.IMREAD_UNCHANGED, "image", log_callback)
    _log(f"image read finished in {time.perf_counter() - t0:.3f}s: {path.name}", log_callback)
    if image is None:
        _log(f"image is None: {path}", log_callback)
        return None
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _load_mask(path: Path, log_callback: Optional[Callable[[str], None]] = None) -> Optional[np.ndarray]:
    _log(f"loading mask: {path}", log_callback)
    t0 = time.perf_counter()
    suffix = path.suffix.lower()
    if suffix == ".npz":
        payload = np.load(str(path))
        if "im_markers" not in payload:
            _log(f"mask npz missing 'im_markers': {path}", log_callback)
            return None
        mask = payload["im_markers"]
    else:
        mask = _read_cv2_any_path(path, cv2.IMREAD_UNCHANGED, "mask", log_callback)
    _log(f"mask read finished in {time.perf_counter() - t0:.3f}s: {path.name}", log_callback)
    if mask is None:
        _log(f"mask is None: {path}", log_callback)
        return None
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = mask.astype(np.int32, copy=False)
    mask[mask < 0] = 0
    return mask


def discover_image_file(base: Path, well: str) -> Optional[Path]:
    for ext in IMAGE_EXTENSIONS:
        candidate = base / f"{well}{ext}"
        if candidate.exists():
            return candidate
    return None


def discover_mask_file(base: Path, well: str) -> Optional[Path]:
    npz_path = base / f"{well}.npz"
    if npz_path.exists():
        return npz_path
    for ext in (".tif", ".tiff"):
        candidate = base / f"{well}_mask{ext}"
        if candidate.exists():
            return candidate
    return None


def _load_well_files(
    well: str,
    image_path: Path,
    mask_path: Path,
    loaded: Dict[str, MicroscopeImportData],
    errors: List[str],
    log_callback: Optional[Callable[[str], None]] = None,
) -> bool:
    t_well = time.perf_counter()
    if not image_path.exists():
        errors.append(f"{well}: image file does not exist: {image_path}")
        _log(f"{well}: explicit image path missing", log_callback)
        return False
    if not mask_path.exists():
        errors.append(f"{well}: mask file does not exist: {mask_path}")
        _log(f"{well}: explicit mask path missing", log_callback)
        return False

    _log(f"{well}: image file -> {image_path.name}", log_callback)
    _log(f"{well}: mask file -> {mask_path.name}", log_callback)

    image = _load_image(image_path, log_callback)
    if image is None:
        errors.append(f"{well}: failed to load image '{image_path.name}'")
        _log(f"{well}: failed image load", log_callback)
        return False
    _log(f"{well}: image shape={image.shape}, dtype={image.dtype}", log_callback)

    mask = _load_mask(mask_path, log_callback)
    if mask is None:
        errors.append(f"{well}: failed to load mask '{mask_path.name}'")
        _log(f"{well}: failed mask load", log_callback)
        return False
    _log(f"{well}: mask shape={mask.shape}, dtype={mask.dtype}, labels_max={int(np.max(mask))}", log_callback)

    if mask.shape[:2] != image.shape[:2]:
        errors.append(
            f"{well}: image/mask shape mismatch image={image.shape[:2]} mask={mask.shape[:2]}"
        )
        _log(f"{well}: shape mismatch", log_callback)
        return False
    if np.max(mask) <= 0:
        errors.append(f"{well}: mask contains no positive labels")
        _log(f"{well}: mask has no positive labels", log_callback)
        return False

    t_centroids = time.perf_counter()
    centroids = calculate_microscope_cell_centroids(mask)
    centroid_labels = np.unique(mask)
    centroid_labels = centroid_labels[centroid_labels > 0].astype(np.int32, copy=False)
    _log(f"{well}: centroid extraction finished in {time.perf_counter() - t_centroids:.3f}s", log_callback)
    if centroids.size == 0:
        errors.append(f"{well}: no centroids extracted from mask")
        _log(f"{well}: no centroids extracted", log_callback)
        return False
    if len(centroid_labels) != len(centroids):
        errors.append(
            f"{well}: centroid/label count mismatch centroids={len(centroids)} labels={len(centroid_labels)}"
        )
        _log(f"{well}: centroid/label count mismatch", log_callback)
        return False

    loaded[well] = MicroscopeImportData(
        image=image,
        mask=mask,
        centroids=centroids.astype(np.float32, copy=False),
        centroid_labels=centroid_labels,
        image_path=str(image_path),
        mask_path=str(mask_path),
    )
    _log(f"{well}: loaded {len(centroids)} centroids in {time.perf_counter() - t_well:.3f}s", log_callback)
    return True


def import_microscope_dataset(
    folder: str,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    well_file_selection: Optional[WellFileSelection] = None,
) -> Tuple[Dict[str, MicroscopeImportData], List[str]]:
    base = Path(folder)
    loaded: Dict[str, MicroscopeImportData] = {}
    errors: List[str] = []
    selected_rows: List[Tuple[str, Path, Path]] = []
    if well_file_selection:
        for well in WELL_NAMES:
            payload = well_file_selection.get(well)
            if payload is None:
                continue
            image_path_raw = str(payload.get("image_path", "")).strip()
            mask_path_raw = str(payload.get("mask_path", "")).strip()
            if not image_path_raw or not mask_path_raw:
                continue
            selected_rows.append((well, Path(image_path_raw), Path(mask_path_raw)))

    total_wells = len(selected_rows) if selected_rows else len(WELL_NAMES)
    processed_wells = 0
    start_all = time.perf_counter()
    _log(f"starting import from folder: {base}", log_callback)
    _emit_progress(progress_callback, processed_wells, total_wells, "Initializing microscope import")
    if selected_rows:
        for well, image_path, mask_path in selected_rows:
            _emit_progress(progress_callback, processed_wells, total_wells, f"{well}: loading selected files")
            _log(f"{well}: loading selected files", log_callback)
            ok = _load_well_files(
                well,
                image_path,
                mask_path,
                loaded,
                errors,
                log_callback=log_callback,
            )
            processed_wells += 1
            status = f"{well}: loaded" if ok else f"{well}: failed"
            _emit_progress(progress_callback, processed_wells, total_wells, status)
    else:
        if not base.exists():
            msg = f"dataset folder does not exist: {base}"
            errors.append(msg)
            _log(msg, log_callback)
            return loaded, errors
        if not base.is_dir():
            msg = f"dataset path is not a directory: {base}"
            errors.append(msg)
            _log(msg, log_callback)
            return loaded, errors

        for well in WELL_NAMES:
            _emit_progress(progress_callback, processed_wells, total_wells, f"{well}: scanning files")
            _log(f"{well}: scanning files", log_callback)
            image_path = discover_image_file(base, well)
            if image_path is None:
                errors.append(f"{well}: missing microscope image ({well}.png/.jpg/.tif)")
                _log(f"{well}: missing image", log_callback)
                processed_wells += 1
                _emit_progress(progress_callback, processed_wells, total_wells, f"{well}: missing image")
                continue

            mask_path = discover_mask_file(base, well)
            if mask_path is None:
                errors.append(f"{well}: missing labeled mask ({well}.npz or {well}_mask.tif)")
                _log(f"{well}: missing mask", log_callback)
                processed_wells += 1
                _emit_progress(progress_callback, processed_wells, total_wells, f"{well}: missing mask")
                continue

            ok = _load_well_files(
                well,
                image_path,
                mask_path,
                loaded,
                errors,
                log_callback=log_callback,
            )
            processed_wells += 1
            status = f"{well}: loaded" if ok else f"{well}: failed"
            _emit_progress(progress_callback, processed_wells, total_wells, status)

    _emit_progress(progress_callback, total_wells, total_wells, "Import finished")
    _log(
        f"import finished in {time.perf_counter() - start_all:.3f}s, "
        f"loaded={len(loaded)}, errors={len(errors)}"
        ,
        log_callback,
    )
    return loaded, errors


def epic_points_to_scaled(epic_points: np.ndarray, scale: int) -> np.ndarray:
    if epic_points.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    return (((epic_points.astype(np.float32) + 0.5) / 80.0) * float(scale)).astype(np.float32)


def project_mic_centroids_to_epic(centroids: np.ndarray, translation: np.ndarray, scale: int) -> np.ndarray:
    if centroids.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    projected = ((centroids.astype(np.float32) - translation.astype(np.float32)) / float(scale)) * 80.0 - 0.5
    return projected.astype(np.float32)


def project_mask_to_epic(mask: np.ndarray, translation: np.ndarray, scale: int, out_shape: Tuple[int, int] = (80, 80)) -> np.ndarray:
    sx = 80.0 / float(scale)
    sy = 80.0 / float(scale)
    tx = -float(translation[0]) * sx
    ty = -float(translation[1]) * sy
    affine = np.array([[sx, 0.0, tx], [0.0, sy, ty]], dtype=np.float32)
    projected = cv2.warpAffine(
        mask.astype(np.int32, copy=False),
        affine,
        (int(out_shape[1]), int(out_shape[0])),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return projected.astype(np.int32, copy=False)


def run_auto_translation(
    method: str,
    epic_points: np.ndarray,
    mic_centroids: np.ndarray,
    scale: int,
    pmc_epsilon: float = 25.0,
    pmc_ratio: float = 0.2,
    stochastic_ratio: float = 0.8,
    stochastic_radius: int = 10,
    seed: int = 42,
) -> np.ndarray:
    source = epic_points_to_scaled(epic_points, scale)
    if len(source) == 0 or len(mic_centroids) == 0:
        return np.zeros(2, dtype=np.int32)

    rng_state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        if method == "stochastic":
            translation, _error = find_translation_stochastic(
                source,
                mic_centroids.astype(np.float32),
                source_indices_ratio=float(stochastic_ratio),
                optimizer_radius=int(stochastic_radius),
            )
        else:
            translation, _clique = find_translation_pmc(
                source,
                mic_centroids.astype(np.float32),
                epsilon=float(pmc_epsilon),
                correspondence_ratio=float(pmc_ratio),
            )
    finally:
        np.random.set_state(rng_state)

    return np.rint(translation).astype(np.int32)


def register_alignment(
    method: str,
    epic_points: np.ndarray,
    mic_mask: np.ndarray,
    mic_centroids: np.ndarray,
    translation: np.ndarray,
    scale: int,
) -> MicroscopeAlignmentResult:
    del epic_points  # Alignment translation is solved earlier; registration keeps all in-bounds segments.
    del mic_mask

    all_centroids = np.asarray(mic_centroids, dtype=np.float32)
    if all_centroids.size == 0:
        projected = np.empty((0, 2), dtype=np.float32)
        seg_indices = np.empty((0,), dtype=np.int32)
    else:
        projected_all = project_mic_centroids_to_epic(all_centroids, translation, scale)
        in_bounds = (
            (projected_all[:, 0] >= 0.0)
            & (projected_all[:, 0] <= 79.0)
            & (projected_all[:, 1] >= 0.0)
            & (projected_all[:, 1] <= 79.0)
        )
        projected = projected_all[in_bounds].astype(np.float32, copy=False)
        seg_indices = np.nonzero(in_bounds)[0].astype(np.int32, copy=False)

    return MicroscopeAlignmentResult(
        method=method,
        translation=translation.astype(np.int32),
        scale=int(scale),
        projected_centroids=projected.astype(np.float32, copy=False),
        projected_seg_indices=seg_indices,
    )


def alignment_to_metadata(result: MicroscopeAlignmentResult) -> Dict[str, object]:
    return {
        "method": result.method,
        "translation": [int(result.translation[0]), int(result.translation[1])],
        "scale": int(result.scale),
        "projected_centroids": result.projected_centroids.tolist(),
        "projected_seg_indices": result.projected_seg_indices.astype(np.int32, copy=False).tolist(),
    }
