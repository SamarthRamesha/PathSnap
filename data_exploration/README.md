# PathSnap Data Exploration

This folder contains the reproducible dataset inventory and exploration report for the PathSnap project.

## Dataset layout

```text
data/
  train/<trip-id>/<trip-id>.{track,nodes,arcs,route}
  student_test/<trip-id>/<trip-id>.{track,nodes,arcs,route}
  evaluator_test/<trip-id>/<trip-id>.{track,nodes,arcs}
```

Trip folders should be named with the 8-digit trip ID and contain files with the same stem.
The expected data is not currently present in this working tree. Restore or download it into `data/`,
then run the analysis.

## Run the exploration

```bash
python3 data_exploration/code/explore_data.py \
  --data-dir data \
  --output-dir data_exploration/output
```

The script is dependency-free and reports:

- trip and file counts;
- GPS row counts and malformed/invalid observations;
- timestamp ordering and sampling gaps;
- longitude/latitude bounds and geodesic trajectory length;
- road-network node and arc counts;
- isolated nodes and disconnected components;
- route ID validity and route length;
- SVG histograms;
- optional per-trip Folium maps when `folium` is installed.

The complete generated report is written to `data_exploration/output/DATA_EXPLORATION_REPORT.md`.

For a simple explanation, read [note.md](./note.md).

For model-training work, use [FEATURE_TABLE.md](./model_development/FEATURE_TABLE.md) and
[feature_catalog.csv](./model_development/feature_catalog.csv). The table documents raw GPS features,
road-arc features, candidate-generation features, sequence features, preprocessing,
and leakage-safe usage for the 80-trip training and 10-trip development partitions.

## Partition and leakage rules

- Train only on `data/train`.
- Use `data/student_test` only for development-stage model comparison.
- Never use evaluator-test labels for training, feature selection, or hyperparameter selection.
- Fit preprocessing and learned transformations using permitted training data only.
- Do not manually adjust predictions from known evaluation routes or hardcode trip answers.

## Interpretation notes

The supplied route is reference information, not independently verified ground truth.
GPS points can be noisy or off-road, timestamps can contain gaps, and road arcs are indexed
independently from GPS rows. Coordinate calculations use haversine distances for a
physical-distance approximation; a local projected CRS is appropriate for detailed map matching.

## Folder layout

```text
data_exploration/
  README.md
  note.md
  code/
    explore_data.py
    generate_candidate_features.py
  output/
    DATA_EXPLORATION_REPORT.md
    per_trip_summary.csv
    partition_summary.json
    plots/
  model_development/
    README.md
    FEATURE_TABLE.md
    feature_catalog.csv
    candidate_features/
      train_candidate_features_sample.csv
      student_test_candidate_features_sample.csv
      candidate_miss_summary.csv
      train_candidate_miss_sample.csv
      student_test_candidate_miss_sample.csv
```

`code/` contains the reusable scripts.
`output/` contains data-exploration reports and plots.
`model_development/` is the organized handoff for the next model-development step.
