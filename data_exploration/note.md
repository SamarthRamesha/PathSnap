# PathSnap Data Exploration - Simple Explanation

## 1. What we started with

The raw dataset contained vehicle GPS trips and their road networks.
It was not already a table for machine learning. It was several separate text files.

There are 80 training trips and 10 student-test trips.

Every trip has an ID and four file types:

| File | Before processing | Meaning |
|---|---|---|
| `.track` | Long table of raw GPS points | Longitude, latitude, timestamp |
| `.nodes` | Road-network coordinate list | Longitude and latitude for road nodes |
| `.arcs` | Road-network connection list | Source-node ID and target-node ID |
| `.route` | List of reference arc IDs | The route recorded as reference information |

The original data was not directly arranged as:

```text
trip, GPS point, candidate road arc, label
```

We converted the raw files into candidate feature tables and quality summaries.

### Concrete example: Trip 00000001

Before processing, this trip existed as four separate files:

```text
data/train/00000001/00000001.track  -> 1,315 GPS observations
data/train/00000001/00000001.nodes  -> 88,809 road nodes
data/train/00000001/00000001.arcs   -> 165,790 road arcs
data/train/00000001/00000001.route  -> 525 reference arc IDs
```

The raw files looked different:

```text
.track:
8.595203 49.844053 1.000000
8.595294 49.844441 2.000000

.nodes:
8.581679 49.857880
8.584637 49.858295

.arcs:
0 281
0 1
0 5

.route:
2641
64973
133948
```

Before processing, `.track` was 1,315 separate observations, the road network was
88,809 node coordinates, and the route was 525 arc IDs. There was no row saying:

```text
GPS point 100 belongs to arc 64973 and should be label 1
```

## Raw value to processed value

The data changed in the following way:

| Raw input | Processed result |
|---|---|
| Longitude and latitude | Validated coordinates plus GPS heading and movement features |
| Timestamp | Timestamp gap, speed, and time-order checks |
| Two road-node IDs in `.arcs` | Arc length and bearing calculated from node coordinates |
| 165,790 trip arcs | Only the nearest 8 arcs kept for each valid GPS point |
| 525 route arc IDs | Approximate true-candidate label for each GPS point |
| `.track` row numbers | `point_idx` identifiers |
| `.route` arc order | `label = 1/0` candidate labels |

For Trip 00000001, 1,315 valid GPS points and 8 candidates per point produced
10,520 candidate rows. The route-derived true arc was missing from those candidates
only 10 times, giving a miss rate of about 0.76%.

## 2. How to understand the data

The `.nodes` row number is the node ID.
The `.arcs` row number is the arc ID.
The `.route` file gives those arc IDs in sequence.
The `.track` file gives the GPS observations in time order.

A GPS row does not directly correspond to an arc row.
For example, GPS point 1 does not automatically mean arc row 1.
That is why candidate generation and map matching are needed.

## 3. What data exploration did

We did not start training a model. We first understood and checked the data.

We:

- counted all trips in each partition;
- counted GPS observations;
- checked file availability and sizes;
- checked malformed rows;
- checked longitude and latitude ranges;
- checked timestamp gaps and backward timestamps;
- calculated trajectory distance;
- counted nodes and arcs;
- checked arc endpoints;
- checked isolated nodes and disconnected road-network components;
- checked that route arc IDs exist in the local arc table;
- made a report and plots from the results.

The scripts and outputs are organized:

- `code/explore_data.py`: creates the EDA summary;
- `code/generate_candidate_features.py`: creates candidate feature tables;
- `output/`: reports and plots;
- `model_development/`: files passed to the model team.

## 4. What we found

The median training trip has about 1,529 GPS observations.
The median student-test trip has about 1,420 GPS observations.

The median training road network has about 9,608 nodes and 19,527 arcs.
The median student-test road network has about 6,597 nodes and 13,145 arcs.

Training GPS observations range from 361 to 19,313.
Student-test GPS observations range from 262 to 3,958.

We found:

- 40 training trips and 2 student-test trips with timestamp or large-gap warnings;
- no isolated nodes in the available networks;
- disconnected components in some road networks;
- no invalid route arc IDs in the route files examined;
- timestamp gaps larger than 2 seconds in many trips;
- no direct GPS-point-to-route-arc mapping in the raw files.

Exact trip-by-trip values are in `output/per_trip_summary.csv`.

### Partition summary: raw size and measured size

| Item | Training partition | Student-test partition |
|---|---:|---:|
| Trips | 80 | 10 |
| GPS observations, minimum | 361 | 262 |
| GPS observations, median | 1,529 | 1,420 |
| GPS observations, maximum | 19,313 | 3,958 |
| Nodes, minimum | 299 | 1,770 |
| Nodes, median | 9,608 | 6,597 |
| Nodes, maximum | 155,003 | 26,635 |
| Arcs, minimum | 649 | 3,577 |
| Arcs, median | 19,527 | 13,145 |
| Arcs, maximum | 307,392 | 56,027 |
| Median trajectory length | 24.07 km | 20.94 km |
| Median timestamp gap | about 1 second | about 1 second |
| Trips with warning flags | 40 | 2 |

This shows why the raw data needs processing. A trip-level model could not simply use
the 20,000 to 300,000 arc network for every point. It would also not be safe to assume
that every 1-second interval was continuous, because several trips contain much larger gaps.

## 5. How bad GPS points were handled

The `.track` file is expected to contain longitude, latitude, and timestamp.

Our handling is:

- skip malformed rows;
- skip longitude outside `-180` to `180`;
- skip latitude outside `-90` to `90`;
- keep timestamp order;
- calculate the gap from the previous valid point;
- mark duplicate or backward timestamps instead of silently changing them;
- set initial movement features to zero for the first valid point;
- set speed to zero when timestamp gap is not usable.

Large gaps are not deleted automatically because they may contain useful evidence about missing
GPS observations. They are visible in the summary and feature table.

## 6. Why candidate features were needed

The original road networks can contain more than 300,000 arcs.
If we made a row for every GPS point and every arc, the table would become extremely large.

Candidate generation narrows the problem.

For each GPS point:

1. Find the road arcs near the point.
2. Keep the nearest 8 arcs.
3. Calculate features for those 8 arcs.
4. Create one table row for each GPS-point-and-candidate-arc pair.

The final table now looks like:

```text
trip_id, point_idx, arc_id, feature 1, feature 2, ..., label
```

This format is `one candidate arc per row`.

### What one candidate row represents

For a noisy GPS point, the processing program finds nearby road arcs and creates rows similar
to this:

```text
trip_id=00000001
point_idx=100
candidate arc A -> perpendicular_distance=12 m, heading_delta=8 degrees, label=0
candidate arc B -> perpendicular_distance=21 m, heading_delta=3 degrees, label=1
candidate arc C -> perpendicular_distance=35 m, heading_delta=20 degrees, label=0
```

The exact arcs and values are candidate-specific. The labels are derived from `.route`:
the nearest route-reference arc receives `1`, and the other candidates receive `0`.

The model is not learning from raw longitude and latitude alone. It learns from the relationship
between each noisy point and each possible road segment.

## 7. Candidate feature examples

The table includes:

- longitude and latitude;
- timestamp and timestamp gap;
- distance traveled between GPS points;
- GPS speed and heading;
- turn angle;
- perpendicular distance from the GPS point to the road arc;
- position along the arc;
- arc length and bearing;
- heading difference;
- incoming and outgoing arc counts;
- candidate rank.

`arc_id` is kept as an identifier so the model can trace a prediction.
It is not used as a numeric model feature because IDs are local to each trip.

The fixed feature order is stored in `FEATURE_TABLE.md` and `feature_catalog.csv`.
The model pipeline should use the same order for training and API inference.

## 8. How labels were created

The label comes only from `.route`.

For a GPS point:

- find the route arcs from that trip;
- compare each route arc with the GPS point;
- choose the nearest route arc as the approximate true arc;
- set label `1` for that candidate;
- set label `0` for the other candidates.

This is an approximate label. The raw files do not give a direct mapping from every GPS point to
its true arc. `.route` is reference information, not guaranteed perfect ground truth.

If no valid route arc exists, no label is invented.
Candidate miss rate becomes `null` instead of assigning a guessed answer.

## 9. Candidate miss rate

Candidate miss rate tells us how often the approximate true route arc was not among the nearest
8 candidate arcs.

Lower miss rate is better.

For the 5 training trips in the sample:

- miss rate ranged from 0.0% to 7.35%.

For the 5 student-test trips:

- miss rate ranged from 0.15% to 2.67%.

Exact counts are in `model_development/candidate_features/candidate_miss_summary.csv`.

### Per-trip candidate results

| Partition | Trip | Valid GPS points | Missed true arc | Miss rate |
|---|---:|---:|---:|---:|
| Training | 00000001 | 1,315 | 10 | 0.76% |
| Training | 00000002 | 7,243 | 532 | 7.35% |
| Training | 00000003 | 821 | 21 | 2.56% |
| Training | 00000004 | 1,064 | 30 | 2.82% |
| Training | 00000005 | 1,045 | 0 | 0.00% |
| Student-test | 00000000 | 2,503 | 24 | 0.96% |
| Student-test | 00000007 | 1,354 | 2 | 0.15% |
| Student-test | 00000009 | 1,049 | 6 | 0.57% |
| Student-test | 00000040 | 2,995 | 5 | 0.17% |
| Student-test | 00000052 | 262 | 7 | 2.67% |

## 10. Before and after

Before candidate preparation:

- one folder per trip;
- separate GPS, node, arc, and route files;
- coordinates and timestamps in raw text;
- road networks containing many unrelated arcs;
- no fixed model table;
- no direct point-to-arc labels.

After data exploration and candidate preparation:

- trips and road networks have been inspected;
- invalid coordinates and malformed rows are identified;
- GPS distances and gaps are measured;
- road networks are checked for connectivity;
- routes are checked against local arc IDs;
- one fixed candidate table is available for each partition;
- one row is produced per `(GPS point, candidate arc)`;
- route-derived labels are separated from model features;
- candidate miss rate shows whether candidate generation is good enough.

## 10A. Complete processing pipeline

The complete processing flow is:

```text
Raw .track
  -> validate coordinates
  -> validate and order timestamps
  -> calculate point motion

Raw .nodes + .arcs
  -> map node IDs to coordinates
  -> rebuild directed road arcs
  -> calculate arc length and bearing
  -> index nearby arcs

Raw .route
  -> keep reference arc IDs
  -> verify IDs exist in local .arcs
  -> create approximate candidate labels

GPS point + candidate arc
  -> distance, heading, arc, and graph features
  -> one row with label 1 or 0

Candidate-table rows
  -> train a candidate scorer
  -> choose a sequence of connected arcs
  -> output predicted route arc IDs
```

## 11. Where to find everything

- `code/explore_data.py`: EDA and trip summaries;
- `code/generate_candidate_features.py`: reusable one-trip candidate generator;
- `output/`: report, CSV summary, partition JSON, and plots;
- `model_development/FEATURE_TABLE.md`: feature definitions;
- `model_development/feature_catalog.csv`: machine-readable feature list;
- `model_development/candidate_features/`: model-ready sample tables and miss-rate summaries;
- `model_development/README.md`: detailed candidate-table instructions.

The main evidence files are:

- `output/DATA_EXPLORATION_REPORT.md`: plain-language partition summary;
- `output/per_trip_summary.csv`: one row for every explored trip;
- `output/partition_summary.json`: minimum, median, and maximum values by partition;
- `output/plots/gps_observations_histogram.svg`: GPS observation count distribution;
- `output/plots/timestamp_gaps_histogram.svg`: timestamp-gap distribution;
- `model_development/candidate_features/train_candidate_features_sample.csv`: sample training candidates;
- `model_development/candidate_features/student_test_candidate_features_sample.csv`: sample development candidates;
- `model_development/candidate_features/candidate_miss_summary.csv`: candidate miss-rate evidence.

## How to rerun

From the repository root:

```bash
python3 data_exploration/code/explore_data.py \
  --data-dir data \
  --output-dir data_exploration/output

python3 data_exploration/code/generate_candidate_features.py \
  --data-dir data \
  --output-dir data_exploration/model_development/candidate_features \
  --candidate-limit 8 \
  --sample-per-partition 5
```

## 12. Important limitations

- GPS points can be noisy or away from roads.
- Timestamp gaps can affect speed and movement features.
- Road networks contain disconnected components in some trips.
- Route reference arcs are not guaranteed to be perfectly correct.
- The candidate sample uses a local approximate meter projection, not a formal cataloged CRS.
- Candidate miss rate is approximate with the current spatial grid.

## 13. Rules for model training

- Train learned parameters only on permitted training data.
- Split by trip ID, not by individual GPS row.
- Do not use student-test routes for training or tuning.
- Do not use evaluator-test routes or labels for feature selection.
- Fit scalers inside each training fold.
- Keep the feature order fixed.
- Use the same candidate generator and feature formulas at API inference time.

## Viva: quick project explanation

### One-minute summary

We developed a dataset exploration and candidate-feature preparation pipeline for PathSnap.
The raw dataset contains vehicle GPS trajectories, road-network nodes and arcs, and reference
routes. We investigated 80 training trips and 10 student-test trips. The pipeline checks data
quality, timestamps, coordinates, road-network connectivity, route validity, and produces
candidate features for map matching.

### What each file represents

- `.track`: longitude, latitude, and timestamp for GPS observations.
- `.nodes`: longitude and latitude for every road-network node.
- `.arcs`: source node and target node for every directed road segment.
- `.route`: ordered reference arc IDs traversed by the vehicle.

The `.nodes` row position is the 0-based node ID. The `.arcs` row position is the 0-based arc ID.
GPS rows are a time series and do not have a direct one-to-one correspondence with arc rows.

### How the system works

1. Load one trip's `.track`, `.nodes`, `.arcs`, and `.route` files.
2. Validate coordinates and timestamps.
3. Build directed road arcs from node pairs.
4. Generate the nearest 8 candidate arcs around each valid GPS point.
5. Calculate GPS, arc, measurement, and graph-related features.
6. Add label `1` only for the route-derived true candidate and `0` otherwise.
7. Keep the feature order fixed for training and API inference.

### Key implementation details

- GPS distance is calculated using the Haversine formula in meters.
- Segment geometry currently uses a local equirectangular approximation:
  `x = longitude * 111320 * cos(mean latitude)` and
  `y = latitude * 111320`.
- Candidate arcs are spatially indexed using a 0.01-degree grid.
- Candidates are ranked by perpendicular distance from the GPS point to the arc segment.
- The fixed candidate limit is 8 in the published sample.
- `arc_id` is kept only as an identifier and is not used as a numeric model feature.

### Dataset findings

- Training: 80 trips.
- Student-test: 10 trips.
- Training median GPS observations: 1,529.
- Student-test median GPS observations: 1,420.
- Training median road-network nodes: 9,608.
- Training median road-network arcs: 19,527.
- 40 training trips and 2 student-test trips had timestamp or larger-gap warnings.
- No route arc IDs outside the local `.arcs` range were found.
- No isolated nodes were found in the available networks.
- Some road networks contain disconnected components.

### Label methodology

The label comes only from `.route`. It does not come from GPS coordinates alone.

For each point, the implementation:

- takes valid arc IDs from the trip's route;
- measures each route arc against the GPS point;
- identifies the nearest route arc as the approximate true arc;
- labels that candidate `1`;
- labels all other candidates `0`.

This is a route-based approximation because the raw files do not give a direct GPS-point-to-arc
mapping. If no route arc is available for a point or trip, a true arc is not invented.

### Bad-point handling

- Malformed GPS rows are skipped.
- Longitude outside [-180, 180] or latitude outside [-90, 90] is skipped.
- The first valid point gets a gap and motion feature of zero.
- Negative or duplicate timestamp gaps remain visible as data-quality features rather than being
  silently corrected.
- Invalid timestamp gaps prevent calculating a meaningful speed and set speed to zero.
- Trip-level warnings are reported separately in the EDA files.

### Candidate miss rate

Candidate miss rate is the percentage of GPS points for which the route-derived true arc is not
among the generated candidates.

For the 5-trip sample:

- Training miss rate range: 0.0% to 7.35%.
- Student-test miss rate range: 0.15% to 2.67%.
- The exact per-trip miss counts are in `candidate_miss_summary.csv`.

The miss rate is a recall-like check for candidate generation. A lower value is better.

### Partition and leakage prevention

- Training data is used for fitting and trip-level cross-validation.
- Student-test data is used only for development-stage comparison.
- Evaluator-test data must remain separate.
- `.route` labels must not be used as model inputs or used to tune student/evaluator predictions.
- Preprocessing, scaling, and model tuning must be fit within permitted training folds.
- Splitting must be by `trip_id`, not individual GPS rows.

### API and inference consistency

Samarth's API pipeline will call the same feature-generation logic used in training.
The feature order is fixed by `FEATURE_COLUMNS`; `arc_id` is an identifier only.

For inference, the model receives the same columns in the same order, the same candidate limit,
the same coordinate calculations, and the same bad-point rules. The route-based label is not
available during inference.

Confirmed behavior for evaluator trips:

- `generate_feature_rows(trip_dir, candidate_limit=8, include_label=False)` works when `.route`
  is missing;
- inference mode does not read `.route`;
- it returns the same 19 model feature columns in the same order;
- it does not create labels or guess ground truth;
- it uses the same local meter projection and candidate-limit logic as training;
- `trip_id`, `point_idx`, and `arc_id` remain identifiers and are not label-derived features.

### Likely viva questions

**Why not use Euclidean distance in degrees?**
Longitude and latitude are angular coordinates. Degree differences do not directly represent
equal physical distances.

**What is map matching?**
It is the process of matching GPS observations to the road arc or sequence of arcs that most
likely represents the vehicle's movement.

**Why candidate generation instead of every arc?**
Some trips have over 300,000 arcs. Generating every possible arc for every GPS point creates an
extremely large table and is inefficient.

**Why do not use `arc_id` as a feature?**
Arc IDs are local to each trip and have no shared numeric meaning across trips.

**Why split by trip rather than GPS row?**
GPS observations from the same trip are strongly correlated. Row-level splitting can leak
information and produce overly optimistic results.

**How is candidate miss rate calculated?**
It compares route-derived true arc IDs with the nearest generated candidate arcs.

**What are the main limitations?**

- Route labels are reference information, not guaranteed error-free ground truth.
- No direct GPS-point-to-arc mapping is provided.
- Candidate selection is currently an approximate grid implementation.
- Timestamp gaps and GPS noise remain difficult cases.
- The local equirectangular CRS is suitable for this prototype, but a formal projected CRS or
  geodesic method should be considered for final implementation.
