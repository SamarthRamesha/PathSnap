# PathSnap Data Exploration - Simple Explanation

## 1. What we started with

The raw dataset contained vehicle GPS trips and their road networks.
It was not already a table for machine learning. It was several separate text files.

There are 80 training trips and 10 student-test trips.
The evaluator-test partition was not in the working data.

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

## 11. Where to find everything

- `code/explore_data.py`: EDA and trip summaries;
- `code/generate_candidate_features.py`: reusable one-trip candidate generator;
- `output/`: report, CSV summary, partition JSON, and plots;
- `model_development/FEATURE_TABLE.md`: feature definitions;
- `model_development/feature_catalog.csv`: machine-readable feature list;
- `model_development/candidate_features/`: model-ready sample tables and miss-rate summaries;
- `model_development/README.md`: detailed candidate-table instructions.

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
