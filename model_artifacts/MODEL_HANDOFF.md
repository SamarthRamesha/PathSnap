# PathSnap Model Handoff

## Selected Model
Optuna-tuned LightGBM classifier.

## Training
- Training trips: 80
- Candidate limit: 8
- Training rows: 1,667,336
- Cross-validation: 5-fold GroupKFold by trip_id
- Optuna trials: 25
- Random seed: 42

## Feature Schema
13 ordered features:

1. timestamp_gap_s
2. step_distance_m
3. point_speed_mps
4. heading_deg
5. turn_angle_deg
6. perpendicular_distance_m
7. along_arc_fraction
8. arc_length_m
9. arc_bearing_deg
10. heading_delta_deg
11. arc_out_degree
12. arc_in_degree
13. candidate_rank

Identifiers and labels are excluded.

## Inference
- Load final_model.joblib.
- Read feature_columns.json.
- Select columns in the exact documented order.
- Call predict_proba(X)[:, 1].
- Positive class: label 1.
- Preserve trip_id, point_idx and arc_id separately for downstream mapping.

## Artifacts
- final_model.joblib
- feature_columns.json
- config.json
- requirements-model.txt
- environment_versions.txt

## Validation
Run:

python model_experiments/smoke_test_model.py

The smoke test must verify feature ordering, probability validity,
inference-only candidate generation and row-order preservation.

## Important
Student-test labels were not used for model fitting, tuning or selection.
Reference routes must never be accessed during inference.