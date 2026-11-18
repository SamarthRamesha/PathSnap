# PathSnap Data Exploration Report

This report is generated from the raw `.track`, `.nodes`, `.arcs`, and `.route` files.
It does not distribute observations across trips or use evaluator labels for model selection.

## Dataset status

Trips found: **90**.

| Partition | Trips | GPS observations (median) | Nodes (median) | Arcs (median) |
|---|---:|---:|---:|---:|
| train | 80 | 1529.0 | 9608.0 | 19526.5 |
| student_test | 10 | 1420.0 | 6596.5 | 13145.0 |
| evaluator_test | 0 | n/a | n/a | n/a |

## File meanings

- `.track`: three whitespace-separated columns: longitude, latitude, timestamp (seconds).
- `.nodes`: two whitespace-separated columns: longitude, latitude.
- `.arcs`: two integer columns: source node ID, target node ID.
- `.route`: one integer column per row: chronological ground-truth arc ID.

Row numbers in `.nodes` are 0-based node IDs. Arc row numbers in `.arcs` are 0-based arc IDs.
GPS row order is a time series and must not be assumed to correspond to road-network row order.

## Coordinate and timestamp findings

Coordinate distance is calculated with the haversine formula because ordinary degree-based
Euclidean distance is not a physical distance. Timestamp gaps are calculated from consecutive
observations; a positive gap greater than 2 seconds is flagged.

See `per_trip_summary.csv` for bounds, observations, gaps, trajectory length, and data-quality flags.

## Road-network findings

Each trip is analyzed separately. Undirected connectivity is used to identify isolated nodes
and disconnected components. Route IDs are checked against the local arc table only.

## Reference-route limitations

- Reference routes are supplied for investigation and should not be treated as error-free ground truth.
- Do not use student-test or evaluator-test routes for training, feature selection, or hyperparameter choices.
- Keep route use isolated to training-trip validation and permitted model development.

## Recommended map-matching assumptions

1. Project coordinates with a local UTM CRS or use geodesic/haversine calculations on angular coordinates.
2. Preserve timestamp order and treat gaps explicitly.
3. Match observations to arcs or arc geometry with a measurement plus transition model.
4. Separate preprocessing fitting from student-test/evaluator-test partitions.
5. Validate route outputs with trip-level cross-validation rather than single global row splits.

## Reproducing the analysis

From the repository root:

```bash
python3 data_exploration/explore_data.py --data-dir data --output-dir data_exploration/output
```

Generated evidence: `per_trip_summary.csv`, `partition_summary.json`, and the visualizations in `plots/`.

## Data warnings

- train: 40 trip(s) with timestamp, invalid-coordinate, or >10-second-gap warnings.
- student_test: 2 trip(s) with timestamp, invalid-coordinate, or >10-second-gap warnings.
- evaluator_test: 0 trip(s) with timestamp, invalid-coordinate, or >10-second-gap warnings.
