# PathSnap Model Feature Table

This table separates information that is available in raw data from features that must be
computed during preprocessing or candidate generation. The primary modeling unit is one
GPS observation matched to one candidate road arc.

The machine-readable copy is
[`feature_catalog.csv`](./feature_catalog.csv).

## 1. Feature levels

| Level | One row represents | Main purpose |
|---|---|---|
| Trip | One trajectory | Data quality, partition, and trip diagnostics |
| GPS observation | One row in a `.track` file | Motion and timestamp features |
| Candidate | One GPS observation plus one candidate arc | Measurement and transition features |
| Arc | One row in a `.arcs` file | Road geometry and connectivity features |
| Sequence | One or more consecutive GPS/candidate observations | Temporal and route-smoothness features |

Model training should use candidate rows. Trip-level summaries must not be used as
observation-level predictors unless they are recomputed from training trips only and justified.

## 2. Raw GPS features

| Feature | Source | Type | Definition | Missing/invalid handling | Leakage risk |
|---|---|---|---|---|---|
| `gps_index` | `.track` row order | integer | 0-based observation position | Preserve ordering; do not fill | None |
| `longitude` | `.track` column 1 | float | GPS longitude | Drop or flag if outside [-180, 180] | None |
| `latitude` | `.track` column 2 | float | GPS latitude | Drop or flag if outside [-90, 90] | None |
| `timestamp` | `.track` column 3 | float | Observation time in seconds | Flag duplicate/backward values | None |
| `timestamp_gap` | consecutive `.track` rows | float | Current timestamp minus previous timestamp | Set no-previous to 0; flag <=0 or >10 seconds | None |
| `gps_step_m` | consecutive GPS coordinates | float | Haversine distance from previous point | Set no-previous to 0 | None |
| `gps_speed_mps` | GPS coordinates and timestamps | float | `gps_step_m / timestamp_gap` | Mask <=0 gaps; cap only as preprocessing experiment | None |
| `gps_heading_deg` | consecutive GPS coordinates | float | Direction of travel in degrees | Set no-previous to 0 or missing | None |
| `turn_angle_deg` | consecutive GPS headings | float | Difference between current and previous heading | Circular angle: [-180, 180] | None |
| `projected_x` | longitude, latitude | float | Local projected easting | Fit/select projection once from dataset location | None |
| `projected_y` | longitude, latitude | float | Local projected northing | Fit/select projection once from dataset location | None |

## 3. Road-network and arc features

| Feature | Source | Type | Definition | Missing/invalid handling | Leakage risk |
|---|---|---|---|---|---|
| `arc_id` | `.arcs` row index | integer | 0-based road arc identifier | Candidate key; must be valid | None |
| `from_node_id` | `.arcs` column 1 | integer | Arc source node | Validate against node count | None |
| `to_node_id` | `.arcs` column 2 | integer | Arc target node | Validate against node count | None |
| `from_lon` | `.nodes[from_node_id]` | float | Source-node longitude | Drop candidate if endpoint invalid | None |
| `from_lat` | `.nodes[from_node_id]` | float | Source-node latitude | Drop candidate if endpoint invalid | None |
| `to_lon` | `.nodes[to_node_id]` | float | Target-node longitude | Drop candidate if endpoint invalid | None |
| `to_lat` | `.nodes[to_node_id]` | float | Target-node latitude | Drop candidate if endpoint invalid | None |
| `arc_length_m` | node coordinates | float | Haversine distance between arc endpoints | Recompute; reject 0 or invalid geometry | None |
| `arc_bearing_deg` | node coordinates | float | Direction from source to target | Circular feature | None |
| `arc_out_degree` | network adjacency | integer | Number of outgoing arcs at source node | 0 only for malformed networks | None |
| `arc_in_degree` | network adjacency | integer | Number of incoming arcs at target node | 0 if no predecessor | None |
| `candidate_rank_by_distance` | candidate set | integer | Rank candidate arcs by perpendicular distance | 1 is nearest | None |

## 4. GPS-to-candidate features

| Feature | Source | Type | Definition | Missing/invalid handling | Leakage risk |
|---|---|---|---|---|---|
| `candidate_id` | candidate generation | integer | Candidate arc ID | Must exist in `.arcs` | None |
| `perpendicular_distance_m` | GPS point and arc segment | float | Minimum distance to candidate arc | Use projected coordinates or local geodesic geometry | None |
| `along_arc_distance_m` | GPS point and arc segment | float | Projected position along candidate arc | Clamp to [0, arc length] | None |
| `forward_along_arc_fraction` | along_arc and arc length | float | `along_arc_distance_m / arc_length_m` | Clamp to [0, 1] | None |
| `distance_log1p_m` | perpendicular distance | float | `log1p(perpendicular_distance_m)` | Keeps large distances from dominating | None |
| `heading_delta_deg` | GPS heading and arc bearing | float | Absolute circular difference, optionally signed | Report 0-180 absolute; signed for transition context | None |
| `speed_arc_ratio` | GPS speed and arc length/time | float | GPS speed relative to plausible arc travel | Missing if timestamp gap is invalid | None |
| `candidate_is_origin` | candidate and GPS index | boolean | Candidate matches expected source region | Derived without route labels | None |
| `candidate_is_destination` | candidate and GPS index | boolean | Candidate matches expected destination region | Derived without route labels | None |

## 5. Sequence and transition features

| Feature | Source | Type | Definition | Missing/invalid handling | Leakage risk |
|---|---|---|---|---|---|
| `previous_arc_id` | previous GPS/candidate | integer | Arc ID selected in previous step | Missing for first observation | None before model output |
| `arc_transition_valid` | candidate arcs | boolean | Directed path existence from previous to candidate | 1-hop or short graph lookup | None before model output |
| `shortest_path_hops` | road graph | integer | Hop count from previous candidate to current candidate | Set high/capped for no path | None before model output |
| `shortest_path_m` | road graph | float | Path distance from previous candidate to current candidate | Set high/capped for no path | None before model output |
| `turn_transition_deg` | prior arc and candidate bearing | float | Circular bearing change | Define for two selected bearings | None before model output |
| `distance_from_previous_arc_m` | previous candidate and GPS point | float | Candidate measurement relation over time | Useful for HMM/sequence models | None before model output |

## 6. Reference-only features

These features must be computed only for labeled training routes or permitted development
validation. They must not be used as model inputs in student-test or evaluator-test inference.

| Feature | Source | Type | Definition | Allowed use |
|---|---|---|---|---|
| `ground_truth_arc_id` | `.route` | integer | Chronological reference arc at trip level | Training label only |
| `route_arc_index` | `.route` row position | integer | Position in reference sequence | Labeled analysis only |
| `route_is_valid` | `.route` vs `.arcs` | boolean | All route IDs map to valid arc IDs | Data-quality check only |
| `is_ground_truth_candidate` | route and candidate | boolean | Candidate equals the reference arc | Positive/negative training label only |
| `route_arc_count` | `.route` | integer | Number of reference arc entries | Trip diagnostics only |

## 7. Candidate generation rules

Recommended baseline:

1. Project all GPS points and road nodes using a local metric CRS.
2. Build road-arc line geometry from the two endpoint nodes.
3. For each GPS point, find a candidate set of arcs within a starting radius such as 100-250 m.
4. Keep a fixed maximum candidate count, for example 5-15 arcs, for consistent training.
5. Add `distance_log1p_m`, heading, arc length, degree, and transition features.
6. Create one row per `(trip_id, gps_index, candidate_id)`.
7. Mark each candidate using the route label only when building the training set.

Candidate radius and candidate count are training-model hyperparameters. Tune them only by
trip-level cross-validation on permitted training data.

## 8. Preprocessing plan

| Data class | Recommendation |
|---|---|
| Coordinates | Validate; reject invalid points; reproject to a local metric CRS |
| Timestamp | Sort only after preserving original order; flag duplicate/backward gaps |
| Missing values | Report before filling; use explicit missingness flags where possible |
| Distance features | Clip or scale within training fold |
| Heading/bearing | Encode cyclic features (`sin`, `cos`) in addition to raw degree if needed |
| Arc ID | Never treat as an ordinal numeric feature |
| Node degree | Standardize or use trees that are insensitive to scale |
| Long-tail distances | Use both raw and log-scaled values |
| Categorical graph features | Use one-hot, embeddings, or graph/model-specific representations |
| Scaling | Fit scalers per training fold only; never fit on student-test or evaluator-test data |

## 9. Split and leakage policy

- Split by `trip_id`, never by GPS row or candidate row.
- Use 80 training trips for development cross-validation.
- Reserve the 10 student-test trips for final development-stage comparison only.
- Do not fit preprocessing parameters on student-test or evaluator-test data.
- Do not use student-test `.route` labels to tune or train the model.
- Keep all route labels out of candidate feature generation at inference time.
- Report trip-level and candidate-level metrics separately.

## 10. Suggested model baseline

A baseline map-matcher can combine:

- emission score: perpendicular distance plus heading difference;
- transition score: shortest-path distance and bearing change;
- sequence model: Viterbi over per-GPS candidate states;
- evaluation label: whether the predicted candidate equals the supplied reference arc;
- robustness check: predicted route continuity and invalid transitions.
