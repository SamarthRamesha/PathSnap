"""Reproducible smoke test for the exported PathSnap model artifacts."""

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

# Ensure repository root is importable when invoked from any working directory
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data_exploration.code.generate_candidate_features import (  # noqa: E402
    INFERENCE_FEATURE_COLUMNS,
    generate_feature_rows,
)

ARTIFACTS_DIR = REPO_ROOT / "model_artifacts"
SAMPLE_TRIP_DIR = REPO_ROOT / "data/student_test/00000000"
IDENTIFIER_COLS = {"trip_id", "point_idx", "arc_id", "label"}


def main() -> None:
    model_path = ARTIFACTS_DIR / "final_model.joblib"
    feat_path = ARTIFACTS_DIR / "feature_columns.json"
    cfg_path = ARTIFACTS_DIR / "config.json"

    # 1) Load artifacts
    model = joblib.load(model_path)
    feature_columns = json.loads(feat_path.read_text(encoding="utf-8"))
    config = json.loads(cfg_path.read_text(encoding="utf-8"))

    assert hasattr(model, "predict_proba"), "Loaded model lacks predict_proba"
    assert hasattr(model, "classes_"), "Loaded model lacks classes_"
    assert feature_columns == config["feature_columns"], "feature_columns.json != config.json"
    assert len(feature_columns) == config["feature_count"], "feature_count mismatch"
    assert IDENTIFIER_COLS.isdisjoint(feature_columns), "Identifier/label leaked into feature_columns"
    if hasattr(model, "feature_names_in_"):
        assert list(model.feature_names_in_) == feature_columns, "Pipeline feature_names_in_ mismatch"

    # 2) Generate real inference-safe candidate rows (no .route read, no label column)
    rows, summary = generate_feature_rows(
        SAMPLE_TRIP_DIR,
        candidate_limit=config["candidate_limit"],
        include_label=False,
    )
    assert len(rows) > 0, "Feature generator returned 0 rows"
    df = pd.DataFrame(rows)
    assert list(df.columns) == INFERENCE_FEATURE_COLUMNS, "Unexpected generator inference columns"
    assert "label" not in df.columns, "label must not be present during inference"

    # 3) Build exact feature matrix and run predict_proba
    X = df[feature_columns].astype(np.float32)
    probs_matrix = model.predict_proba(X)

    pos_label = config["positive_class_label"]
    classes = list(model.classes_)
    assert pos_label in classes, f"Positive label {pos_label} not in classes_={classes}"
    pos_idx = classes.index(pos_label)
    assert pos_idx == config["positive_class_index"], "positive_class_index mismatch"

    pos_probs = probs_matrix[:, pos_idx]

    # 4) Validate probability invariants and output ordering
    assert len(pos_probs) == len(df), "Probability count != candidate row count"
    assert np.all(np.isfinite(pos_probs)), "Non-finite probabilities detected"
    assert np.all((pos_probs >= 0.0) & (pos_probs <= 1.0)), "Probabilities outside [0, 1]"

    out = df[["trip_id", "point_idx", "arc_id"]].copy()
    out["prob"] = pos_probs
    assert out[["trip_id", "point_idx", "arc_id"]].equals(df[["trip_id", "point_idx", "arc_id"]]), (
        "Output row ordering was not preserved"
    )

    # Confirm per-point top-1 candidate selection produces one valid arc per GPS point
    top1 = out.loc[out.groupby(["trip_id", "point_idx"], sort=False)["prob"].idxmax()]
    assert len(top1) == summary["gps_points_valid"], "Top-1 count != valid GPS point count"

    print("SMOKE TEST PASSED")
    print(f"  Model:           {config['model_name']} ({type(model).__name__})")
    print(f"  Classes:         {classes} (positive_class={pos_label} at index {pos_idx})")
    print(f"  Features ({len(feature_columns)}):   {feature_columns}")
    print(f"  Trip tested:     {SAMPLE_TRIP_DIR.name} ({summary['gps_points_valid']} GPS pts, {len(df)} candidate rows)")
    print(f"  Prob range:      min={pos_probs.min():.6f}, mean={pos_probs.mean():.6f}, max={pos_probs.max():.6f}")
    print(f"  Top-1 mean prob: {top1['prob'].mean():.6f} across {len(top1)} GPS observations")


if __name__ == "__main__":
    main()
