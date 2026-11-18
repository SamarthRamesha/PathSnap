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

All 525 reference arc IDs are in range, and all 524 consecutive pairs connect in the direction of travel. Eight arc records have coincident endpoints. The smoke test omitted those zero-length arcs while retaining the original row IDs for the remaining arcs. The preprocessing adapter still needs to agree on a policy for these records.

The raw files supply the network and observations. Candidate generation produces a small set of projected positions on nearby arcs for each GPS observation. Reference routes are reserved for evaluation and are a different sequence from the per-observation matches.

## Proposed inputs and outputs

- Pass ordered observations with stable GPS row indices, finite timestamps in seconds, and exactly two finite GPS coordinates in a common metric projection. Use an empty candidate list for an observed point with no candidates. Set the GPS position to null for an explicitly missing coordinate. Retain timestamp gaps when observations are absent from the source.
- Pass candidates grouped by observation. Each candidate has a trip-local directed arc ID, offset in meters from the source node, and exactly one of perpendicular distance in meters or a finite model log score. Candidate scores attach to the same GPS row and arc projection that the model scored. The probability adapter converts positive candidate probabilities to natural log scores.
- Pass directed arcs with source and target node IDs and lengths in meters. Use the same length convention for arc lengths and offsets.
- Return one selected candidate or null per observation, contiguous decoded arc sequences with intermediate network arcs, and break reasons at the corresponding GPS indices. The API can map arc IDs to node geometry for display.

## Scoring and gaps

A hidden state is a candidate directed arc and its projected offset at one observation. The bootstrap emission is the Gaussian distance log score `-0.5 * (distance / sigma)^2`. A supplied model log score replaces it. Transitions use `-abs(network_distance - gps_distance) / beta`, the log of an exponential compatibility factor. These are relative scores for Viterbi. Directed shortest-path distance includes the residual length of the previous arc and the prefix of the next arc. Forward movement on the same arc uses the difference in offsets. Unreachable transitions have zero weight.

Use uniform starting weights per segment. A missing GPS position or empty candidate set closes the active segment and yields a null match. A non-increasing timestamp, an excessive gap, or an unreachable transition between candidate-bearing rows closes the active segment and starts another at the current row. Exact candidate-score ties prefer the first candidate in input order. Dijkstra tries lower-ID nodes first at equal distances and retains the first equally short bridge it discovers. When log scores approach the numeric limits, the decoder shifts emissions and previous path scores by common offsets to preserve relative weights. Scoring scales and the gap threshold need training-trip validation.

## Open integration decisions

- Pruthviraj and Raksha should confirm projection, candidate identity, and offset units.
- Reyansh and Raksha should confirm how scores map back to candidates, their meaning and calibration, and the held-out comparison protocol.
- Samarth and Raksha should confirm the evaluator request and response formats and obtain the organiser brief or endpoint documentation to establish who provides the evaluator UI.

## Running the prototype

The decoder requires Python 3.10 or newer and uses the standard library. From the repository root, run:

```sh
python -m unittest discover -s tests -v
python -m pytest -q -rs
```

Install pytest to use the second command. The benchmark input regression uses NumPy and Folium and is skipped when either optional dependency is missing.

`hmm.viterbi.decode(observations, arcs)` accepts `Observation`, `Candidate`, and `Arc` instances from the same module. Use `xy_m=None` for an explicitly missing GPS sample. `DecodeResult.matches` preserves input order and uses `None` for unmatched observations. `routes` lists each contiguous sequence in order, and `breaks` gives the GPS index and reason (`missing_gps`, `no_candidates`, `gps_gap`, `non_increasing_time`, or `disconnected`).

The prototype defaults to `sigma_m=15`, `beta_m=30`, and `max_gap_s=30`. Setting `max_gap_s=None` disables the upper time-gap split. These defaults are uncalibrated. For each step, the decoder batches directed shortest paths from each required source. It caches up to 128 exact source-and-target-set requests for one decode. Cache misses can still explore a large component. Benchmark representative candidate counts and networks before using it in the evaluator.

## Evaluator UI status

The inspected repository branches and public search did not identify an organiser UI, submission spec, or confirmed endpoint. The supplied task assigns Raksha an evaluator-facing view. Confirm with Samarth or the organisers whether this view is a deliverable and which request and response format it should use.

## FastAPI integration with Reyansh's model

Use the probability adapter for the confirmed model format. Keep all three
per-observation inputs in the same GPS row order, including rows with empty
lists, and supply one trip's full directed road network.

```python
from hmm.viterbi import Arc, GpsSample, decode_probabilities

result = decode_probabilities(
    gps_samples=[GpsSample(gps_index=42, timestamp_s=4.0, xy_m=(23.0, 0.0))],
    probabilities=[[(17, 1.0)]],
    offsets_m=[{17: 23.0}],
    arcs=[Arc(arc_id=17, source=11, target=12, length_m=40.0)],
)
```

- Use at most eight unique arc IDs per score list and probabilities in [0, 1].
  The adapter takes the natural log of positive scores as supplied, with no
  normalization requirement. Zero-probability candidates are skipped.
- For each positive-score candidate with a GPS position, provide its offset
  in meters from the directed arc's source, in [0, length_m]. The available
  FastAPI adapter uses the generator's directed along-arc fraction times the
  network arc length in meters, and projects GPS observations with the same local
  scale as candidate generation, translated to the first sample. Set
  `xy_m=None` for missing GPS and use an empty offset map when none are needed.
- Include all valid directed arcs, including potential connecting arcs outside
  the candidate lists. Arc IDs, endpoint IDs, and GPS indices must be nonnegative
  integers. Each arc needs a positive finite length in meters. Keep the arc ID
  when filtering invalid geometry. Keep a separate arc-to-geometry lookup
  for rendering routes. Node coordinates are only needed upstream for
  computing lengths and offsets, or downstream for rendering.
- Invalid input, such as mismatched row counts, duplicate scores, malformed GPS
  coordinates, missing offsets, or out-of-range numeric scores, raises `ValueError`.

The result is a `DecodeResult` dataclass. For a FastAPI response, use
`fastapi.encoders.jsonable_encoder(result)`. The example gives:

```json
{
  "matches": [{"arc_id": 17, "offset_m": 23.0, "distance_m": null, "log_score": 0.0}],
  "routes": [[17]],
  "breaks": []
}
```

`matches` has one entry per input GPS row, with `null` for an unmatched row.
`log_score` is the model's log probability for that candidate. Each array in
`routes` represents one contiguous decoded segment. A break entry has the
shape `{"gps_index": 43, "reason": "disconnected"}`. See the prototype section
above for the full set of reasons.

## Independent map demo

From the repository root, install Folium and render the comparison:

```sh
python -m pip install "folium>=0.16"
python -m demos.hmm_map --output /tmp/hmm-demo.html
```

Open the HTML in a browser. The basemap and Folium assets load over the internet.
The map uses a five-observation synthetic fixture and hand-assigned model
probabilities. At GPS row 1, the classifier picks disconnected arc 3 with
probability 0.75. Viterbi picks arc 2 with probability 0.25 and follows the
network through intermediate arc 1. The 50-second GPS gap appears as a pause
marker and starts a new route segment.

Both methods use the same candidates and GPS readings. The classifier-only
baseline selects the highest probability per observation, then applies the
same routing and gap rules with those choices fixed. The disconnected step
count excludes the intentional GPS gap. This fixture demonstrates behavior.
A comparison on held-out trips still needs real model scores.

For the full training-trip map and timings, see [the real-trip benchmark](hmm-benchmark.md).
