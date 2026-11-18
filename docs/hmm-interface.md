# PathSnap HMM interface proposal

This is a provisional Python boundary for one trip. Pruthviraj and Reyansh can adapt their candidate tables and model outputs to this boundary once they agree on their formats.

## Dataset inspection

Training trip `00000001` on `main` contains these whitespace-separated files. Row identifiers are zero-based and local to the trip.

| File | Sample row | Meaning | Rows |
| --- | --- | --- | ---: |
| .track | `8.595203 49.844053 1.000000` | Longitude, latitude, timestamp in seconds | 1,315 |
| .nodes | `8.581679 49.857880` | Longitude and latitude, keyed by row index | 88,809 |
| .arcs | `0 281` | Directed edge from node 0 to node 281, keyed by row index | 165,790 |
| .route | `2641` | Ordered reference arc ID | 525 |

The raw files supply the network and observations. Candidate generation produces a small set of projected positions on nearby arcs for each GPS observation. Reference routes are reserved for evaluation and are a different sequence from the per-observation matches.

## Proposed inputs and outputs

- Pass ordered observations with stable GPS row indices, timestamps in seconds, and GPS coordinates in a common metric projection. Use an empty candidate list for a known missing observation and retain timestamp gaps when observations are absent from the source.
- Pass candidates grouped by observation. Each candidate has a trip-local directed arc ID, offset in meters from the source node, and exactly one of perpendicular distance in meters or a finite model log score. Candidate scores attach to the same GPS row and arc projection that the model scored. The adapter will decide how to transform model outputs after Reyansh confirms their semantics.
- Pass directed arcs with source and target node IDs and lengths in meters. Use the same length convention for arc lengths and offsets.
- Return one selected candidate or null per observation, contiguous decoded arc sequences with intermediate network arcs, and break reasons at the corresponding GPS indices. The API can map arc IDs to node geometry for display.

## Scoring and gaps

A hidden state is a candidate directed arc and its projected offset at one observation. The bootstrap emission is the Gaussian distance log score `-0.5 * (distance / sigma)^2`. A supplied model log score replaces it. Transitions use `-abs(network_distance - gps_distance) / beta`, the log of an exponential compatibility factor. These are relative scores for Viterbi. Directed shortest-path distance includes the residual length of the previous arc and the prefix of the next arc. Forward movement on the same arc uses the difference in offsets. Unreachable transitions have zero weight.

Use uniform starting weights per segment. Split on empty candidate sets, non-increasing timestamps, gaps above a configurable threshold, or when all feasible predecessors are lost. Preserve those breaks in the returned result. Fit scoring scales and any gap threshold on training-trip validation after the data and feature interfaces are confirmed.

## Open integration decisions

- Pruthviraj and Raksha should confirm projection, candidate identity, and offset units.
- Reyansh and Raksha should confirm how scores map back to candidates, their meaning and calibration, and the held-out comparison protocol.
- Samarth and Raksha should confirm the evaluator request and response formats and obtain the organiser brief or endpoint documentation to establish who provides the evaluator UI.
