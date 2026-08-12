from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

V1_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V1_ROOT / "v1a1"))

from v1a1.components import extract_components, side_candidates


TARGETS = {
    1: "kidney",
    2: "tumor",
    3: "cyst",
}
SIDES = ("left", "right")


def parse_splits(text: str) -> list[str]:
    values = [x.strip().lower() for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("No allowed splits supplied")
    if "test" in values:
        raise ValueError("V1b.0 is development-only; test split is forbidden")
    return values


def component_row(component, image_height: int, image_width: int) -> dict:
    if component is None:
        return {
            "presence": 0,
            "selected_area": 0,
            "x1": 0.0,
            "y1": 0.0,
            "x2": 0.0,
            "y2": 0.0,
            "center_x_norm": 0.0,
            "center_y_norm": 0.0,
            "box_width_norm": 0.0,
            "box_height_norm": 0.0,
        }
    return {
        "presence": 1,
        "selected_area": int(component.area),
        "x1": float(component.x1),
        "y1": float(component.y1),
        "x2": float(component.x2),
        "y2": float(component.y2),
        "center_x_norm": float(component.cx / image_width),
        "center_y_norm": float(component.cy / image_height),
        "box_width_norm": float((component.x2 - component.x1) / image_width),
        "box_height_norm": float((component.y2 - component.y1) / image_height),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit KiTS23 semantic targets for V1b without training or test access. "
            "Build one row per case/slice/target/side using the largest connected "
            "component on that side as single-box supervision."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allowed-splits", default="train,val")
    parser.add_argument("--min-component-area", type=int, default=5)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed_splits = parse_splits(args.allowed_splits)

    frame = pd.read_csv(manifest_path, dtype={"case_id": str})
    required = {"case_id", "slice_idx", "npz_path", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"Manifest missing required columns: {sorted(missing)}")

    frame["split"] = frame["split"].astype(str).str.lower()
    development = frame[frame["split"].isin(allowed_splits)].copy()
    if development.empty:
        raise RuntimeError(f"No rows found for allowed splits: {allowed_splits}")
    if (development["split"] == "test").any():
        raise RuntimeError("Test rows entered the V1b.0 development audit")

    case_split_counts = development.groupby("case_id")["split"].nunique()
    if int(case_split_counts.max()) != 1:
        raise RuntimeError("Case-level split leakage detected")

    # The inherited V1a manifest is one row per case/slice. Deduplicate defensively.
    development = (
        development.sort_values(["case_id", "slice_idx"])
        .drop_duplicates(["case_id", "slice_idx"], keep="first")
        .reset_index(drop=True)
    )

    missing_npz = [p for p in development["npz_path"].astype(str) if not Path(p).is_file()]
    if missing_npz:
        raise RuntimeError(
            f"{len(missing_npz)} development NPZ files are missing; "
            f"examples={missing_npz[:5]}"
        )

    semantic_rows: list[dict] = []
    contrast_rows: list[dict] = []
    observed_labels: set[int] = set()

    for row in tqdm(development.itertuples(index=False), total=len(development), desc="V1b.0 semantic audit"):
        npz_path = Path(str(row.npz_path))
        with np.load(npz_path, allow_pickle=False) as payload:
            if "label" not in payload:
                raise KeyError(f"label missing from {npz_path}")
            label = np.asarray(payload["label"], dtype=np.uint8)

        if label.ndim != 2:
            raise ValueError(f"Expected 2D label in {npz_path}, got {label.shape}")

        image_height, image_width = map(int, label.shape)
        observed_labels.update(int(x) for x in np.unique(label))

        target_present_names: list[str] = []
        target_component_counts: dict[str, int] = {}

        for target_label, target_name in TARGETS.items():
            mask = label == target_label
            components = extract_components(mask, min_area=args.min_component_area)
            target_component_counts[target_name] = int(len(components))
            if components:
                target_present_names.append(target_name)

            sides = side_candidates(components, image_width)
            for side in SIDES:
                side_components = sides[side]
                selected = max(side_components, key=lambda c: c.area) if side_components else None
                side_total_area = int(sum(c.area for c in side_components))
                fields = component_row(selected, image_height, image_width)
                selected_fraction = (
                    float(fields["selected_area"] / side_total_area)
                    if side_total_area > 0
                    else 0.0
                )
                semantic_rows.append(
                    {
                        "case_id": str(row.case_id),
                        "slice_idx": int(row.slice_idx),
                        "npz_path": str(npz_path),
                        "split": str(row.split),
                        "target_label": int(target_label),
                        "target_name": target_name,
                        "side": side,
                        "image_height": image_height,
                        "image_width": image_width,
                        "side_component_count": int(len(side_components)),
                        "side_total_component_area": side_total_area,
                        "selected_component_fraction": selected_fraction,
                        **fields,
                    }
                )

        contrast_rows.append(
            {
                "case_id": str(row.case_id),
                "slice_idx": int(row.slice_idx),
                "npz_path": str(npz_path),
                "split": str(row.split),
                "num_semantic_targets_present": int(len(target_present_names)),
                "semantic_targets_present": ";".join(target_present_names),
                "kidney_component_count": target_component_counts["kidney"],
                "tumor_component_count": target_component_counts["tumor"],
                "cyst_component_count": target_component_counts["cyst"],
                "is_semantic_contrast_slice": int(len(target_present_names) >= 2),
            }
        )

    semantic = pd.DataFrame(semantic_rows).sort_values(
        ["case_id", "slice_idx", "target_label", "side"]
    )
    contrast = pd.DataFrame(contrast_rows).sort_values(["case_id", "slice_idx"])

    semantic_path = output_dir / "v1b_semantic_target_manifest.csv"
    contrast_path = output_dir / "v1b_semantic_contrast_slices.csv"
    semantic.to_csv(semantic_path, index=False)
    contrast.to_csv(contrast_path, index=False)

    count_rows: list[dict] = []
    for split in allowed_splits:
        split_rows = semantic[semantic["split"] == split]
        for target_label, target_name in TARGETS.items():
            target_rows = split_rows[split_rows["target_label"] == target_label]
            for side in SIDES:
                subset = target_rows[target_rows["side"] == side]
                positive = subset[subset["presence"] == 1]
                ambiguous = positive[positive["side_component_count"] > 1]
                count_rows.append(
                    {
                        "split": split,
                        "target_label": target_label,
                        "target_name": target_name,
                        "side": side,
                        "total_samples": int(len(subset)),
                        "positive_samples": int(len(positive)),
                        "positive_cases": int(positive["case_id"].nunique()),
                        "multi_component_positive_samples": int(len(ambiguous)),
                        "median_selected_component_fraction": (
                            float(positive["selected_component_fraction"].median())
                            if len(positive)
                            else 0.0
                        ),
                    }
                )

    counts = pd.DataFrame(count_rows)
    counts_path = output_dir / "v1b_semantic_target_counts.csv"
    counts.to_csv(counts_path, index=False)

    target_split_gate = {}
    for split in allowed_splits:
        target_split_gate[split] = {}
        for target_name in TARGETS.values():
            subset = counts[(counts["split"] == split) & (counts["target_name"] == target_name)]
            target_split_gate[split][target_name] = {
                "positive_side_samples": int(subset["positive_samples"].sum()),
                "positive_cases_across_sides": int(
                    semantic[
                        (semantic["split"] == split)
                        & (semantic["target_name"] == target_name)
                        & (semantic["presence"] == 1)
                    ]["case_id"].nunique()
                ),
            }

    contrast_summary = {}
    for split in allowed_splits:
        subset = contrast[contrast["split"] == split]
        contrast_summary[split] = {
            "slices": int(len(subset)),
            "semantic_contrast_slices": int(subset["is_semantic_contrast_slice"].sum()),
            "cases_with_semantic_contrast": int(
                subset.loc[subset["is_semantic_contrast_slice"] == 1, "case_id"].nunique()
            ),
        }

    required_labels_present = all(label in observed_labels for label in TARGETS)
    every_target_in_every_split = all(
        target_split_gate[split][target]["positive_side_samples"] > 0
        for split in allowed_splits
        for target in TARGETS.values()
    )
    validation_contrast_exists = (
        contrast_summary.get("val", {}).get("semantic_contrast_slices", 0) > 0
        if "val" in allowed_splits
        else True
    )

    summary = {
        "status": "v1b0_semantic_target_audit",
        "source_manifest": str(manifest_path),
        "allowed_splits": allowed_splits,
        "test_access": False,
        "development_cases": int(development["case_id"].nunique()),
        "development_slices": int(len(development)),
        "target_labels": TARGETS,
        "observed_label_values": sorted(observed_labels),
        "min_component_area": int(args.min_component_area),
        "target_split_support": target_split_gate,
        "semantic_contrast": contrast_summary,
        "gate": {
            "all_kits23_semantic_labels_observed": bool(required_labels_present),
            "every_target_has_positive_support_in_each_allowed_split": bool(every_target_in_every_split),
            "validation_contains_same_image_multi_target_contrast": bool(validation_contrast_exists),
            "case_level_split_leakage": False,
            "test_access": False,
            "advance_to_v1b1": bool(
                required_labels_present
                and every_target_in_every_split
                and validation_contrast_exists
            ),
        },
        "outputs": {
            "semantic_target_manifest": str(semantic_path),
            "semantic_target_counts": str(counts_path),
            "semantic_contrast_slices": str(contrast_path),
        },
    }

    summary_path = output_dir / "v1b_semantic_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nSemantic target counts:")
    print(counts.to_string(index=False))
    print("\nAudit summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if not summary["gate"]["advance_to_v1b1"]:
        raise RuntimeError(
            "V1b.0 gate did not pass. Inspect the generated audit files before designing V1b.1."
        )


if __name__ == "__main__":
    main()
