# Candidate Feature Tables

This directory contains an immediate small-sample candidate feature table and a reusable
one-trip feature-generation function.

## Reusable function

```python
from data_exploration.code.generate_candidate_features import generate_feature_rows

rows, summary = generate_feature_rows(
    Path("data/train/00000001"),
    candidate_limit=8,
)
```

`generate_feature_rows()`:

- reads one trip's `.track`, `.nodes`, `.arcs`, and `.route` files;
- skips malformed or invalid-coordinate GPS rows;
- generates up to eight nearest candidate arcs;
- returns fixed-order candidate rows and a miss-rate summary.

The fixed feature order is `FEATURE_COLUMNS`. `arc_id` is included only as an identifier,
not as a numeric model feature.

## Evaluator and inference trips

Use inference mode when `.route` is absent:

```python
rows, summary = generate_feature_rows(
    Path("data/evaluator_test/00000000"),
    candidate_limit=8,
    include_label=False,
)
```

Inference mode does not read `.route`, and it does not return a label column.
It returns the same 19 model feature columns used by training, after separating the label:

```text
trip_id, point_idx, arc_id, longitude, latitude, timestamp,
timestamp_gap_s, step_distance_m, point_speed_mps, heading_deg,
turn_angle_deg, perpendicular_distance_m, along_arc_fraction,
arc_length_m, arc_bearing_deg, heading_delta_deg, arc_out_degree,
arc_in_degree, candidate_rank
```

## Generate the sample

```bash
python3 data_exploration/code/generate_candidate_features.py \
  --data-dir data \
  --output-dir data_exploration/model_development/candidate_features \
  --candidate-limit 8 \
  --sample-per-partition 5
```

The sample script writes:

- `train_candidate_features_sample.csv`
- `student_test_candidate_features_sample.csv`
- `train_candidate_miss_sample.csv`
- `student_test_candidate_miss_sample.csv`
- `candidate_miss_summary.csv`

## Coordinate and distance handling

Features use a local equirectangular meter projection based on the mean latitude of the valid
GPS points in a trip. Longitude and latitude remain explicit in the table for inspection.
Segment and trajectory calculations use meters. Arc and GPS headings use degrees.
Arc lengths use haversine meters.

This is not a cataloged EPSG CRS. It is a local approximate projected coordinate system:

```text
x = longitude * 111320 * cos(mean_trip_latitude)
y = latitude * 111320
```

For final API consistency, use the same formulas and candidate limit in training and inference.
The existing road-network coordinates are in longitude/latitude.

## Candidate generation

Arcs are indexed in a 0.01-degree spatial grid using segment endpoints and midpoint. A point
starts in its grid cell and expands search rings until it finds at least the requested candidate
count or reaches the search limit. Candidates are ranked by exact local perpendicular segment
distance and retain the smallest 5-10 arcs.

The grid can miss a candidate for a very long or unusual segment outside its midpoint/endpoint
cells. Candidate miss rate should therefore be treated as approximate for this fast sample.
Before final production training, compare it against a brute-force or spatial-indexed implementation.

## Labels and route ambiguity

The label comes only from `.route`.

For each valid GPS point, the script finds the nearest arc in the route arc set. That route arc
is the temporary true arc for the point. It gets label `1`; all other candidates get `0`.
Candidate miss means the nearest route arc was not among the generated candidates.

This is an explicit route-based approximation because the raw `.track` and `.route` files do not
provide a direct point-index-to-arc-index mapping. There is no independent point-level ground
truth.

If a trip has no valid route arc IDs, no true arc can be identified:

- all labels are `0`;
- candidate miss rate is `null`;
- route-related diagnostics remain separate.

This avoids inventing labels from GPS proximity alone.

## Bad-point handling

- Malformed `.track` rows are skipped.
- Invalid longitudes or latitudes are skipped.
- Missing timestamps cannot be reordered reliably and are skipped if the row is malformed.
- Timestamp gaps are measured against the previous valid point.
- Duplicate or backward timestamps keep a zero/negative gap value and produce a flag-like feature;
  they are not silently converted into positive timing.
- `step_distance_m`, `heading_deg`, `turn_angle_deg`, and `speed` are set to zero or normal values
  for the first valid point.
- Invalid timestamp gaps give `point_speed_mps = 0`.

Trip-level gap warnings still come from the exploration report and per-trip CSV.

## Feature order

The feature order is fixed in `FEATURE_COLUMNS` and both partition CSVs use it exactly.
Downstream model code should consume the documented order or explicit column names, never rely
on a separate inference-time CSV schema.

`arc_id` is an identifier and is not included in numeric feature scaling or model input.
