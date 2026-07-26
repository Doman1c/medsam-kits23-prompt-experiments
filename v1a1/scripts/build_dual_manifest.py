from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v1a1.components import build_dual_tracks
from v1a1.data import load_npz
from v1a1.utils import parse_int_list

CASE_RE = re.compile(r"case_\d+")
SLICE_RE = re.compile(r"slice_(\d+)")


def infer_case_id(path: Path) -> str:
    match = CASE_RE.search(str(path))
    if not match:
        raise ValueError(f"Could not infer case ID from {path}")
    return match.group(0)


def infer_slice_idx(path: Path) -> int:
    match = SLICE_RE.search(path.stem)
    if not match:
        raise ValueError(f"Could not infer slice index from {path}")
    return int(match.group(1))


def read_split_map(path: Path | None, case_ids: list[str], seed: int = 42) -> dict[str, str]:
    if path is not None and path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise RuntimeError(f"Empty split CSV: {path}")
        split_col = "split" if "split" in rows[0] else "split_group"
        mapping = {row["case_id"]: row[split_col].lower() for row in rows}
        # Preserve the established 100/20 split; cases absent from the file are test.
        return {case_id: mapping.get(case_id, "test") for case_id in case_ids}

    rng = np.random.default_rng(seed)
    shuffled = list(case_ids)
    rng.shuffle(shuffled)
    mapping = {}
    for i, case_id in enumerate(shuffled):
        mapping[case_id] = "train" if i < 100 else "val" if i < 120 else "test"
    return mapping


def area_bin(value: float) -> str:
    if value <= 0:
        return "negative"
    if value < 0.10:
        return "00_10"
    if value < 0.50:
        return "10_50"
    if value < 0.90:
        return "50_90"
    return "90_100"


def component_fields(prefix: str, component, image_shape: tuple[int, int]) -> dict:
    height, width = image_shape
    if component is None:
        return {
            f"{prefix}_presence": 0,
            f"{prefix}_area": 0,
            f"{prefix}_x1": 0.0,
            f"{prefix}_y1": 0.0,
            f"{prefix}_x2": 0.0,
            f"{prefix}_y2": 0.0,
            f"{prefix}_center_x_norm": 0.0,
            f"{prefix}_center_y_norm": 0.0,
            f"{prefix}_box_width_norm": 0.0,
            f"{prefix}_box_height_norm": 0.0,
        }
    return {
        f"{prefix}_presence": 1,
        f"{prefix}_area": int(component.area),
        f"{prefix}_x1": float(component.x1),
        f"{prefix}_y1": float(component.y1),
        f"{prefix}_x2": float(component.x2),
        f"{prefix}_y2": float(component.y2),
        f"{prefix}_center_x_norm": float(component.cx / width),
        f"{prefix}_center_y_norm": float(component.cy / height),
        f"{prefix}_box_width_norm": float((component.x2 - component.x1) / width),
        f"{prefix}_box_height_norm": float((component.y2 - component.y1) / height),
    }



def validate_case_layout(processed_root: Path, case_ids: list[str]) -> None:
    errors = []
    for case_id in case_ids:
        metadata_path = processed_root / case_id / "case_metadata.json"
        if not metadata_path.exists():
            errors.append(f"{case_id}: missing case_metadata.json")
            continue
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{case_id}: unreadable metadata: {exc}")
            continue
        if payload.get("processed_format_version") != "v1a1_dual_yx_v2":
            errors.append(
                f"{case_id}: incompatible processed format "
                f"{payload.get('processed_format_version')}; rebuild processed data"
            )
        if payload.get("volume_array_layout") != "ZYX":
            errors.append(f"{case_id}: expected ZYX layout")
        if payload.get("anatomical_lr_axis_in_slice") != "columns":
            errors.append(f"{case_id}: anatomical left/right is not mapped to columns")
    if errors:
        preview = "\n".join(errors[:10])
        raise RuntimeError(
            "Processed data uses an incompatible in-plane axis convention. "
            "Rebuild it with the axis-fixed prepare_kits23_2d.py.\n" + preview
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical dual-kidney manifest and z-consistent tracks")
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--existing-split-csv", default=None)
    parser.add_argument("--target-labels", default="1")
    parser.add_argument("--min-component-area", type=int, default=5)
    parser.add_argument("--max-track-gap", type=int, default=3)
    parser.add_argument("--near-negative-distance", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    processed_root = Path(args.processed_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_labels = parse_int_list(args.target_labels)

    case_paths: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for path in sorted(processed_root.rglob("*.npz")):
        case_paths[infer_case_id(path)].append((infer_slice_idx(path), path.resolve()))
    if not case_paths:
        raise RuntimeError(f"No NPZ files under {processed_root}")
    for paths in case_paths.values():
        paths.sort(key=lambda item: item[0])
    case_ids = sorted(case_paths)
    validate_case_layout(processed_root, case_ids)
    split_map = read_split_map(
        Path(args.existing_split_csv) if args.existing_split_csv else None,
        case_ids,
        seed=args.seed,
    )

    raw_rows: list[dict] = []
    audit_rows: list[dict] = []
    case_summaries: dict[str, dict] = {}
    for case_id in tqdm(case_ids, desc="Building dual-kidney tracks"):
        entries = case_paths[case_id]
        masks: list[np.ndarray] = []
        shapes: list[tuple[int, int]] = []
        for _, path in entries:
            try:
                image, label = load_npz(path)
            except Exception as exc:
                raise RuntimeError(f"Failed to read processed NPZ: {path}: {exc}") from exc
            masks.append(np.isin(label, np.asarray(target_labels)).astype(np.uint8))
            shapes.append(image.shape)
        if len(set(shapes)) != 1:
            raise ValueError(f"Inconsistent slice shapes in {case_id}: {set(shapes)}")
        left_track, right_track, audit = build_dual_tracks(
            masks,
            min_area=args.min_component_area,
            max_gap=args.max_track_gap,
        )
        audit["case_id"] = case_id
        audit_rows.append({
            "case_id": case_id,
            "left_present_slices": audit["left"]["present_slices"],
            "right_present_slices": audit["right"]["present_slices"],
            "left_crossing_count": audit["left"]["crossing_count"],
            "right_crossing_count": audit["right"]["crossing_count"],
            "left_large_jump_count_20pct": audit["left"]["large_jump_count_20pct"],
            "right_large_jump_count_20pct": audit["right"]["large_jump_count_20pct"],
            "left_adjacent_jump_max": audit["left"]["adjacent_jump_max"],
            "right_adjacent_jump_max": audit["right"]["adjacent_jump_max"],
            "slices_with_more_than_two_components": audit["slices_with_more_than_two_components"],
        })
        max_area = {
            "left": max((x.area for x in left_track if x is not None), default=0),
            "right": max((x.area for x in right_track if x is not None), default=0),
        }
        max_combined_area = max(
            (
                (left.area if left is not None else 0)
                + (right.area if right is not None else 0)
                for left, right in zip(left_track, right_track)
            ),
            default=0,
        )
        positive_indices = [
            entries[i][0]
            for i, (left, right) in enumerate(zip(left_track, right_track))
            if left is not None or right is not None
        ]
        case_summaries[case_id] = {
            "max_left_area": max_area["left"],
            "max_right_area": max_area["right"],
            "max_combined_area": max_combined_area,
        }

        for i, ((slice_idx, path), left, right) in enumerate(zip(entries, left_track, right_track)):
            height, width = shapes[i]
            row = {
                "case_id": case_id,
                "slice_idx": slice_idx,
                "npz_path": str(path),
                "split": split_map[case_id],
                "image_height": height,
                "image_width": width,
            }
            row.update(component_fields("left", left, (height, width)))
            row.update(component_fields("right", right, (height, width)))
            row["has_any"] = int(left is not None or right is not None)
            combined_area = int(row["left_area"] + row["right_area"])
            row["combined_area"] = combined_area
            row["combined_relative_area"] = (
                combined_area / max(max_combined_area, 1) if combined_area > 0 else 0.0
            )
            row["combined_area_bin"] = area_bin(row["combined_relative_area"])
            for side in ("left", "right"):
                area = int(row[f"{side}_area"])
                relative = area / max(max_area[side], 1) if area > 0 else 0.0
                row[f"{side}_relative_area"] = relative
                row[f"{side}_area_bin"] = area_bin(relative)
            if row["has_any"]:
                distance = 0
            elif positive_indices:
                distance = min(abs(slice_idx - z) for z in positive_indices)
            else:
                distance = 999999
            row["distance_to_positive"] = int(distance)
            row["is_near_negative"] = int(
                not row["has_any"] and distance <= args.near_negative_distance
            )
            raw_rows.append(row)

    frame = pd.DataFrame(raw_rows).sort_values(["case_id", "slice_idx"])
    manifest_path = output_dir / "v1a1_dual_manifest.csv"
    frame.to_csv(manifest_path, index=False)
    audit_frame = pd.DataFrame(audit_rows).sort_values("case_id")
    audit_frame.to_csv(output_dir / "dual_track_audit.csv", index=False)

    split_summary = {}
    for split, group in frame.groupby("split"):
        split_summary[split] = {
            "cases": int(group["case_id"].nunique()),
            "slices": int(len(group)),
            "positive_slices": int(group["has_any"].sum()),
            "left_positive_sides": int(group["left_presence"].sum()),
            "right_positive_sides": int(group["right_presence"].sum()),
        }
    case_split_counts = frame.groupby("case_id")["split"].nunique()
    leakage_pass = bool(case_split_counts.max() == 1)
    summary = {
        "manifest": str(manifest_path),
        "total_cases": int(frame["case_id"].nunique()),
        "total_slices": int(len(frame)),
        "split_summary": split_summary,
        "case_level_leakage_check": leakage_pass,
        "left_crossings": int(audit_frame["left_crossing_count"].sum()),
        "right_crossings": int(audit_frame["right_crossing_count"].sum()),
        "cases_with_large_left_jumps": int((audit_frame["left_large_jump_count_20pct"] > 0).sum()),
        "cases_with_large_right_jumps": int((audit_frame["right_large_jump_count_20pct"] > 0).sum()),
        "target_labels": list(target_labels),
        "canonical_orientation": "RAS",
        "anatomical_side_rule": "low x = left kidney; high x = right kidney",
    }
    (output_dir / "dual_manifest_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not leakage_pass:
        raise RuntimeError("Case-level leakage detected")
    if summary["left_crossings"] or summary["right_crossings"]:
        raise RuntimeError("Anatomical side crossing detected in generated tracks")


if __name__ == "__main__":
    main()
