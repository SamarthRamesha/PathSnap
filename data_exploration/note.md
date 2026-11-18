# PathSnap Data Exploration Note

## What was done

The raw PathSnap dataset was investigated and organized into a reproducible exploration workflow.
A dependency-free Python script was created to read the `.track`, `.nodes`, `.arcs`, and `.route`
files for every trip. It calculates a per-trip summary and writes team-readable reports and plots.

The main files created are:

- [`explore_data.py`](./explore_data.py): scans every trip and calculates dataset statistics.
- [`README.md`](./README.md): explains how to run the exploration.
- [`DATA_EXPLORATION_REPORT.md`](./output/DATA_EXPLORATION_REPORT.md): summarizes findings.
- [`per_trip_summary.csv`](./output/per_trip_summary.csv): contains one row of measurements per trip.
- [`partition_summary.json`](./output/partition_summary.json): contains combined statistics for each partition.
- [`plots/`](./output/plots): contains histogram SVG files.
- [`FEATURE_TABLE.md`](./FEATURE_TABLE.md): defines GPS, road-network, candidate, and sequence features for model training.
- [`feature_catalog.csv`](./feature_catalog.csv): provides the feature list in machine-readable form.

The script was run for the real uploaded data. It found 80 training trips and 10 student-test trips.
The evaluator-test partition was not present in the working dataset.

## What the data is

The dataset contains vehicle GPS trajectories and the road network associated with each trip.
A trip is stored in a folder named with an eight-digit ID, and the files inside use the same ID.

There are four relevant file types:

| File | Meaning | Format |
|---|---|---|
| `.track` | GPS observations along a vehicle trajectory | Longitude, latitude, timestamp |
| `.nodes` | Coordinates of road-network nodes | Longitude, latitude |
| `.arcs` | Directed connections between road-network nodes | From-node ID, to-node ID |
| `.route` | Reference sequence of traversed road arcs | One arc ID per row |

Example:

```text
data/train/00000001/
  00000001.track
  00000001.nodes
  00000001.arcs
  00000001.route
```

The same trip ID connects these four files. However, a row in the track file is not necessarily
associated with a row having the same index in the road-network files.

## How the files relate

1. The `.nodes` file contains road-network coordinates.
2. Each row of `.nodes` has a 0-based node ID equal to its line position.
3. The `.arcs` file contains a directed road segment as a source-node ID and target-node ID pair.
4. Each row of `.arcs` has a 0-based arc ID equal to its line position.
5. The `.route` file lists the arc IDs believed to form the vehicle's reference route.
6. The `.track` file provides the sequence of observed GPS points and timestamps.

The map-matching problem is to determine which road arc or sequence of arcs best explains each
GPS trajectory. Arc geometry must be reconstructed from the arc endpoints and node coordinates.

## How data exploration was performed

The exploration followed these steps.

### 1. Inventory the partitions

The script searched `data/train`, `data/student_test`, and `data/evaluator_test`.
It counted trip folders and checked whether track, node, arc, and route files were available.
It also recorded file sizes.

### 2. Analyze GPS observations

For every `.track` file, the script:

- reads longitude, latitude, and timestamp columns;
- counts observations and malformed rows;
- checks that coordinates fall within longitude and latitude ranges;
- calculates gaps between consecutive timestamps;
- flags backward or duplicate timestamps;
- calculates the geographic extent of each trajectory;
- calculates an approximate trajectory length with the haversine distance formula.

Timestamp gaps greater than 2 seconds are flagged. Gaps greater than 10 seconds are highlighted
as stronger anomalies because they may indicate missing observations or travel interruptions.

### 3. Analyze the road network

For every `.nodes` and `.arcs` pair, the script:

- counts nodes and valid arcs;
- validates arc endpoints against the node range;
- detects malformed rows;
- builds an undirected connectivity view;
- counts isolated nodes;
- counts disconnected components;
- reports component sizes and average degree.

The connected-component result helps identify whether a trip's road network is a single usable
network or contains unrelated fragments.

### 4. Validate reference routes

For every `.route` file, the script:

- counts the route arc sequence;
- counts duplicate arc entries;
- counts unique arc IDs;
- verifies that every route arc ID exists in the local `.arcs` file.

It does not assume that route IDs are correct physical ground truth. The route is reference
information that may contain limitations or errors.

### 5. Generate evidence

The script produces:

- a summary table in CSV format;
- JSON statistics grouped by partition;
- a Markdown report with team-readable findings;
- SVG histograms for GPS counts and timestamp gaps;
- optional HTML maps through Folium when Folium is installed.

## What the results show

The actual dataset results are:

| Partition | Trips | Median GPS observations | Median nodes | Median arcs |
|---|---:|---:|---:|---:|
| Training | 80 | 1,529 | 9,608 | 19,527 |
| Student testing | 10 | 1,420 | 6,597 | 13,145 |

Additional observations:

- 40 training trips and 2 student-test trips have timestamp, invalid-coordinate, or very-large-gap warnings.
- Training GPS counts range from 361 to 19,313 observations.
- Student-test GPS counts range from 262 to 3,958 observations.
- No route arc ID outside the local arc range was found in the available route files.
- No isolated nodes were found in the available networks.
- The road networks contain disconnected components in some trips.

Exact values for every trip are in
[`per_trip_summary.csv`](./output/per_trip_summary.csv).

## Geographic considerations

Longitude and latitude are angular coordinates, so ordinary Euclidean distance in degrees should
not be used as a physical distance. This exploration uses the haversine formula to estimate
distance in meters.

For a map-matching implementation, teams should consider:

- a locally projected coordinate system such as UTM;
- geodesic or great-circle distance;
- the local size and location of the trajectory;
- GPS measurement noise;
- irregular sampling;
- large timestamp gaps;
- observations that do not lie exactly on a road arc.

## Data-quality assumptions and limitations

The following assumptions are used for interpretation:

- `.track` rows are ordered in time.
- Consecutive timestamps define sampling gaps.
- `.nodes` row numbers are 0-based node IDs.
- `.arcs` row numbers are 0-based arc IDs.
- `.route` values are 0-based arc IDs.
- A route is useful reference information but is not guaranteed to be error-free.

The supplied data may also have:

- GPS noise;
- missing observations;
- irregular timestamps;
- road-network fragments;
- GPS points slightly away from roads;
- routes that cannot be independently verified.

None of these issues should be interpreted as part of a complete ground-truth guarantee.

## Partition and leakage rules

The teams must preserve the separation between partitions:

- use `data/train` for training and permitted development;
- use `data/student_test` for development-stage evaluation and model comparison;
- keep evaluator-test data and labels separate;
- do not use evaluator-test labels for feature selection;
- do not use evaluator-test results to choose algorithms or hyperparameters;
- do not manually alter predictions based on known evaluation routes;
- do not hardcode answers for particular trips;
- fit preprocessing and learned transformations only with permitted training data;
- validate models using trip-level splits rather than random mixing of rows from the same trip.

## How to rerun the exploration

Run this command from the repository root:

```bash
python3 data_exploration/explore_data.py \
  --data-dir data \
  --output-dir data_exploration/output
```

The script uses only the Python standard library, so no third-party installation is required for
the current CSV and SVG outputs. Folium maps are generated only when Folium is installed.

## Recommended next steps

1. Examine trips with more than 10-second gaps.
2. Inspect coordinate bounds and identify outliers.
3. Build road-arc geometries from node endpoints.
4. Project coordinates into an appropriate local system.
5. Select candidate arcs near each GPS observation.
6. Use both measurement distance and transition probability for map matching.
7. Validate with trip-level cross-validation.
8. Use the route reference only for permitted training and validation work.
