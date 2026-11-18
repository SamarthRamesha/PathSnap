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

## Data types and file structure

| File | Rows | Columns | Types | Header |
|---|---|---:|---|---|
| `.track` | GPS observations | 3 | float longitude, float latitude, float timestamp | None |
| `.nodes` | Road-network nodes | 2 | float longitude, float latitude | None |
| `.arcs` | Road-network arcs | 2 | integer source node ID, integer target node ID | None |
| `.route` | Reference arc sequence | 1 | integer arc ID | None |

Whitespace is the field delimiter. The script reports malformed rows and rejects invalid numeric
values before spatial calculations.

## File availability and sizes

- `train`: 80 trip folder(s); track 80, nodes 80, arcs 80, route 80. Total sizes: track 6.36 MiB, nodes 36.65 MiB, arcs 42.22 MiB, route 0.16 MiB.
- `student_test`: 10 trip folder(s); track 10, nodes 10, arcs 10, route 10. Total sizes: track 0.58 MiB, nodes 1.74 MiB, arcs 1.81 MiB, route 0.02 MiB.
- `evaluator_test`: 0 trip folder(s); track 0, nodes 0, arcs 0, route 0. Total sizes: track 0.00 MiB, nodes 0.00 MiB, arcs 0.00 MiB, route 0.00 MiB.

## Coordinate and timestamp findings

Coordinate distance is calculated with the haversine formula because ordinary degree-based
Euclidean distance is not a physical distance. Timestamp gaps are calculated from consecutive
observations; a positive gap greater than 2 seconds is flagged.

See `per_trip_summary.csv` for bounds, observations, gaps, trajectory length, and data-quality flags.
The `lon_min`, `lon_max`, `lat_min`, and `lat_max` columns give the geographical extent
of every trip. The combined-coordinate and gap histograms provide aggregate visual evidence.
A local projected CRS or a geodesic/haversine calculation is preferable to Euclidean degree distance.

## Road-network findings

Each trip is analyzed separately. Undirected connectivity identifies isolated nodes and weak
components. Route IDs are checked against the local arc table only.

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
python3 data_exploration/code/explore_data.py --data-dir data --output-dir data_exploration/output
```

Generated evidence: `per_trip_summary.csv`, `partition_summary.json`, and the visualizations in `plots/`.

## Data warnings

- train: 40 trip(s) with timestamp, invalid-coordinate, or >10-second-gap warnings.
  - Trip IDs: `00000001, 00000002, 00000004, 00000006, 00000011, 00000014, 00000021, 00000022, 00000024, 00000026, 00000028, 00000032, 00000034, 00000037, 00000038, 00000041, 00000042, 00000043, 00000044, 00000046, 00000054, 00000055, 00000057, 00000058, 00000060, 00000062, 00000068, 00000069, 00000070, 00000074, 00000078, 00000080, 00000081, 00000084, 00000085, 00000088, 00000089, 00000090, 00000098, 00000099`
- student_test: 2 trip(s) with timestamp, invalid-coordinate, or >10-second-gap warnings.
  - Trip IDs: `00000065, 00000076`
- evaluator_test: 0 trip(s) with timestamp, invalid-coordinate, or >10-second-gap warnings.
