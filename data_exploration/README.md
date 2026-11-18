# PathSnap Candidate Features

This folder prepares model-ready candidate features. It does not generate plots or report files.

## Files

```text
data_exploration/
  README.md
  generate_candidate_features.py
  candidate_features/
    train_sample.csv
    student_test_sample.csv
    miss_summary.csv
  note.md
```

## Generate the table

```bash
python3 data_exploration/generate_candidate_features.py \
  --data-dir data \
  --output-dir data_exploration/candidate_features \
  --candidate-limit 8 \
  --sample-per-partition 5
```

The generator is dependency-free and returns one row per GPS point and candidate arc.
The train and student-test files have the same columns.

## Fixed feature order

`trip_id`, `point_idx`, `arc_id`, `longitude`, `latitude`, `timestamp`,
`timestamp_gap_s`, `step_distance_m`, `point_speed_mps`, `heading_deg`,
`turn_angle_deg`, `perpendicular_distance_m`, `along_arc_fraction`,
`arc_length_m`, `arc_bearing_deg`, `heading_delta_deg`, `arc_out_degree`,
`arc_in_degree`, `candidate_rank`, `label`.

`arc_id` identifies the candidate and is not a numeric model feature.

## Data rules

- Use `.route` only to create the training label.
- Use `label = 1` for the nearest route arc to a GPS point and `0` for other candidates.
- If no route arc matches, do not invent a label.
- Do not train or tune on student-test route labels.
- Split cross-validation by trip ID.
- Use the same candidate limit and feature function for API inference.
