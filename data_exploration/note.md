# PathSnap Data Exploration Note

## What we did

We inspected the GPS and road-network data and created a simple candidate-feature table for
model training.

The dataset has 80 training trips and 10 student-test trips. Each trip contains:

- `.track`: longitude, latitude, timestamp
- `.nodes`: longitude, latitude
- `.arcs`: from-node ID, to-node ID
- `.route`: reference arc IDs

## How the table is created

For every valid GPS point:

1. Validate longitude and latitude.
2. Read the timestamp and calculate the gap from the previous valid point.
3. Calculate the GPS heading, step distance, and speed.
4. Find up to 8 nearest candidate road arcs.
5. Calculate distance, angle, arc length, and degree features.
6. Set `label = 1` for the route-derived true arc and `0` for other candidates.

One output row represents one `(GPS point, candidate arc)` pair.

## Files

- `generate_candidate_features.py`: reusable feature generator
- `candidate_features/train_sample.csv`: five training trips
- `candidate_features/student_test_sample.csv`: five student-test trips
- `candidate_features/miss_summary.csv`: candidate miss rates

## Coordinate handling

Coordinates are longitude and latitude. Physical distances use Haversine meters. Candidate
segment geometry uses a local approximate meter projection with the mean trip latitude:

```text
x = longitude * 111320 * cos(mean latitude)
y = latitude * 111320
```

Heading and bearing values are in degrees.

## Bad points and gaps

- Malformed track rows are skipped.
- Coordinates outside valid longitude/latitude ranges are skipped.
- The first valid point uses zero for its initial step and gap features.
- Duplicate or backward timestamps remain visible through a zero or negative gap.
- Invalid timestamp gaps do not produce a meaningful speed.

## Labels

Labels come only from `.route`. The raw track and route files do not provide a direct
point-index-to-arc-index mapping, so the nearest route arc to each GPS point is used as an
approximate true arc. This is reference information, not guaranteed ground truth.

If no route arc is available, no label is invented and all candidate labels remain `0`.

## Candidate miss rate

Candidate miss rate means the true route arc was not among the 8 candidates.

- Training sample range: 0.0% to 7.35%.
- Student-test sample range: 0.15% to 2.67%.

## Fixed schema

Both partition files use the same column order. `arc_id` is an identifier, not a numeric feature.

## Leakage rules

- Use training trips to fit preprocessing and cross-validation.
- Use student-test only for development comparison.
- Do not use student-test route labels for training or tuning.
- Split by trip, not by GPS row.
- Use the same feature function and candidate limit for API inference.
