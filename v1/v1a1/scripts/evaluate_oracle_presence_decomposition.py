from __future__ import annotations

from oracle_presence_decomposition_core import *  # noqa: F401,F403


def main() -> None:
    args = parse_args()
    paths = [
        Path(args.predictions), Path(args.medsam_details),
        Path(args.frozen_test_summary), Path(args.calibration_json), Path(args.checkpoint),
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    medsam_repo = Path(args.medsam_repo)
    if not medsam_repo.is_dir():
        raise FileNotFoundError(medsam_repo)
    frozen_summary = validate_lock(paths[2], paths[3], args.expansion_scale)

    predictions = pd.read_csv(paths[0])
    predictions["case_id"] = predictions["case_id"].astype(str)
    predictions["slice_idx"] = predictions["slice_idx"].astype(int)
    required = {"case_id", "slice_idx", "npz_path", "image_height", "image_width",
                "combined_area_bin", "combined_relative_area"}
    for side in SIDES:
        required.update({f"{side}_gt_presence", f"{side}_smooth_presence"})
        required.update({f"{side}_smooth_{key}" for key in ("x1", "y1", "x2", "y2")})
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Predictions missing columns: {missing}")
    positive = predictions[
        predictions["left_gt_presence"].astype(int).eq(1)
        | predictions["right_gt_presence"].astype(int).eq(1)
    ].copy()
    if len(positive) != args.expected_positive_slices:
        raise ValueError(f"Expected {args.expected_positive_slices} positive slices, found {len(positive)}")
    missing_npz = [p for p in positive["npz_path"].astype(str) if not Path(p).is_file()]
    if missing_npz:
        raise FileNotFoundError(f"{len(missing_npz)} NPZ paths missing; first: {missing_npz[:5]}")

    scores = existing_scores(pd.read_csv(paths[1]), args.expected_positive_slices)
    sys.path.insert(0, str(medsam_repo))
    from segment_anything import sam_model_registry

    device = resolve_device(args.device)
    model = sam_model_registry["vit_b"](checkpoint=str(paths[4])).to(device)
    model.eval()
    b_details = run_b(
        model, device, positive, scores,
        parse_int_list(args.target_labels), args.expansion_scale,
    )
    decomposition = scores.merge(
        b_details, on=["case_id", "slice_idx"], validate="one_to_one"
    )
    presence = add_presence_status(positive)[
        ["case_id", "slice_idx", "false_positive_sides", "false_negative_sides", "side_status"]
    ]
    decomposition = decomposition.merge(
        presence, on=["case_id", "slice_idx"], suffixes=("", "_check"), validate="one_to_one"
    )
    if "side_status_check" in decomposition:
        if (decomposition["side_status"] != decomposition["side_status_check"]).any():
            raise RuntimeError("Side-status merge mismatch")
        decomposition = decomposition.drop(columns="side_status_check")
    decomposition["presence_decision_gain_dice"] = (
        decomposition["B_oracle_presence_auto_box_dice"] - decomposition["A_automatic_dice"]
    )
    decomposition["box_localization_gap_dice"] = (
        decomposition["C_oracle_box_dice"] - decomposition["B_oracle_presence_auto_box_dice"]
    )
    decomposition["total_oracle_gap_dice"] = (
        decomposition["C_oracle_box_dice"] - decomposition["A_automatic_dice"]
    )
    residual = (
        decomposition["total_oracle_gap_dice"]
        - decomposition["presence_decision_gain_dice"]
        - decomposition["box_localization_gap_dice"]
    ).abs().max()
    if float(residual) > 1e-12:
        raise RuntimeError(f"Decomposition identity failed: {residual}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    b_details.to_csv(output / "oracle_presence_auto_box_details.csv", index=False)
    decomposition.to_csv(output / "gap_decomposition_per_slice.csv", index=False)
    by_area = grouped(decomposition, "combined_area_bin")
    by_area["_order"] = by_area["combined_area_bin"].map({v: i for i, v in enumerate(AREA_ORDER)})
    by_area = by_area.sort_values("_order").drop(columns="_order")
    by_status = grouped(decomposition, "side_status")
    by_case = grouped(decomposition, "case_id")
    by_case["extra_side_rate"] = by_case["case_id"].map(
        decomposition.groupby("case_id")["false_positive_sides"].apply(lambda s: float((s > 0).mean()))
    )
    by_case["missing_side_rate"] = by_case["case_id"].map(
        decomposition.groupby("case_id")["false_negative_sides"].apply(lambda s: float((s > 0).mean()))
    )
    by_area.to_csv(output / "gap_decomposition_by_area.csv", index=False)
    by_status.to_csv(output / "gap_decomposition_by_side_status.csv", index=False)
    by_case.sort_values("presence_decision_gain_mean", ascending=False).to_csv(
        output / "gap_decomposition_per_case.csv", index=False
    )

    summary = {
        "status": "frozen_test_oracle_presence_gap_decomposition",
        "analysis_only_not_for_test_tuning": True,
        "modes": {
            "A": "automatic presence + automatic smooth boxes",
            "B": "ground-truth presence + automatic smooth boxes",
            "C": "ground-truth presence + oracle boxes",
        },
        "locked_configuration": frozen_summary["locked_configuration"],
        "data": {
            "positive_slices": int(len(decomposition)),
            "presence_exact_slices_reused_from_A": int((b_details["B_result_source"] == "reused_A_exact_presence").sum()),
            "presence_mismatch_slices_recomputed": int((b_details["B_result_source"] == "recomputed_presence_mismatch").sum()),
        },
        "overall": aggregate(decomposition),
        "by_area": by_area.to_dict(orient="records"),
        "by_side_status": by_status.to_dict(orient="records"),
        "artifact_sha256": {
            "predictions": sha256(paths[0]),
            "original_medsam_details": sha256(paths[1]),
            "frozen_test_summary": sha256(paths[2]),
            "calibration_json": sha256(paths[3]),
            "medsam_checkpoint": sha256(paths[4]),
            "B_details": sha256(output / "oracle_presence_auto_box_details.csv"),
            "decomposition_per_slice": sha256(output / "gap_decomposition_per_slice.csv"),
        },
        "identity": "For every slice and in aggregate: (C - A) = (B - A) + (C - B).",
        "research_guardrail": (
            "This oracle decomposition diagnoses the frozen test result. Do not use it "
            "to select test thresholds, expansion, checkpoints, loss weights, or architecture variants."
        ),
    }
    (output / "gap_decomposition_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved gap decomposition to: {output}")


if __name__ == "__main__":
    main()
