"""Read microscope projects v1–v5 without rewriting source data."""
import copy
import hashlib
from pathlib import Path

import numpy as np
import tifffile
import yaml


def read_microscope_project(path, *, payload=None):
    path = Path(path).expanduser().resolve()
    if payload is None:
        try:
            payload = yaml.load(path.read_text(encoding="utf-8"), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
        except yaml.YAMLError as exc:
            raise ValueError(f"Could not parse microscope project: {path}") from exc
    if not isinstance(payload, dict) or payload.get("kind") != "nanobio-microscope-segmentation-project":
        raise ValueError("File is not a Nanobio Microscope Segmentation batch project")
    if payload.get("version") not in range(1, 6):
        raise ValueError(f"Unsupported microscope batch project version: {payload.get('version')}")
    rows = payload.get("samples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Microscope batch project must contain at least one sample")
    dataset = (path.parent / payload.get("dataset_root", ".")).resolve()
    results = (path.parent / payload.get("results_root", ".")).resolve()
    samples, seen = [], set()
    for index, original in enumerate(rows, 1):
        if not isinstance(original, dict):
            raise ValueError(f"Sample {index} must be an object")
        row = copy.deepcopy(original)
        sid = row.get("id")
        if not isinstance(sid, str) or not sid.strip() or sid in seen:
            raise ValueError(f"Missing or duplicate microscope sample ID: {sid}")
        seen.add(sid)
        if not row.get("image_path"):
            raise ValueError(f"Sample {index} has no image_path")
        row["image_path"] = str((dataset / row["image_path"]).resolve())
        channels = row.get("image_channels", [])
        for channel in channels:
            channel["path"] = str((dataset / channel["path"]).resolve())
        records = {}
        for kind, field in (("segmentation", "segmentation_masks"), ("composite", "composite_masks")):
            mapping = row.get(field, {})
            if not isinstance(mapping, dict):
                raise ValueError(f"{sid}: {field} must be a mapping")
            for key, entry in mapping.items():
                if not isinstance(entry, dict) or not entry.get("mask_path"):
                    raise ValueError(f"{sid}: mask {key} has no mask_path")
                entry["mask_path"] = str((results / entry["mask_path"]).resolve())
                records[f"{kind}:{key}"] = entry
        if not records and row.get("mask_path"):
            settings = payload.get("segmentation_settings", {})
            selected = settings.get("cellpose", {}).get("selected_channel_ids") or []
            primary = (selected[0] if settings.get("segmentation_mode") == "Cellpose" and selected else
                       settings.get("thresholding", {}).get("channel_id"))
            primary = primary or next((c["channel_id"] for c in channels if c["path"] == row["image_path"]), "c1")
            records[f"segmentation:{primary}"] = dict(mask_path=str((results / row["mask_path"]).resolve()))
        reference = row.get("active_mask_reference")
        active = f"{reference.get('kind')}:{reference.get('id')}" if reference else None
        active = active or (f"segmentation:{row['active_segmentation_channel_id']}" if row.get("active_segmentation_channel_id") else None)
        if active and active not in records:
            raise ValueError(f"{sid}: unknown active mask {active}")
        row["mask_records"] = records
        row["active_mask"] = active or next(iter(records), None)
        samples.append(row)
    return payload, samples


def select_project_mask(sample, reference=None):
    reference = reference or sample["active_mask"]
    records = sample["mask_records"]
    if reference not in records:
        raise ValueError(f"{sample['id']}: no mask for {reference}; export masks before importing")
    def validate(key, seen):
        if key in seen or key not in records:
            raise ValueError(f"{sample['id']}: missing or cyclic mask dependency {key}")
        entry = records[key]
        if not key.startswith("composite:"):
            return
        if not Path(entry["mask_path"]).is_file():
            raise ValueError(f"{sample['id']}: missing mask file {entry['mask_path']}")
        recipe = entry.get("recipe", {})
        sources = {recipe.get("main"), recipe.get("guide")}
        if (recipe.get("stale") or recipe.get("roi") != sample.get("metadata", {}).get("mask")
                or recipe.get("rule") != "guide_centroid_in_main" or None in sources
                or set(recipe.get("source_fingerprints", {})) != sources):
            raise ValueError(f"{sample['id']}: invalid or stale composite {key}; regenerate it")
        for source, expected in recipe["source_fingerprints"].items():
            source_key = ("composite:" + source[len("__composite__"):] if source.startswith("__composite__")
                          else "segmentation:" + source)
            validate(source_key, seen | {key})
            labels = np.ascontiguousarray(tifffile.imread(records[source_key]["mask_path"]), dtype=np.int32)
            digest = hashlib.sha256(str(labels.shape).encode())
            digest.update(memoryview(labels).cast("B"))
            if digest.hexdigest() != expected:
                raise ValueError(f"{sample['id']}: stale composite {key}; regenerate it")
    validate(reference, set())
    return records[reference]
