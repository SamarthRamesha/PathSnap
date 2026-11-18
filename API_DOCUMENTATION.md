# PathSnap API

Start the API from the repository root with the backend environment activated.

```sh
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Swagger UI is available at [`/docs`](http://127.0.0.1:8000/docs). The generated schema is at `/openapi.json`. The team-owned browser client at `/ui` implements the organiser's trip-selection workflow with the routes below. Confirm any deployment-specific rules during evaluation.

## Health and readiness

| Method and path | Status | Response |
| --- | --- | --- |
| `GET /health` | 200 | `status: "ok"`, `app: "PathSnap"`, and `components` |
| `GET /ready` | 200 or 503 | `status: "ready"` or `"unavailable"`, and `components` |

Each report contains `road_network`, `feature_generator`, `model`, and `hmm_decoder` in `components`. Each component has an `available` boolean and a `detail` string or null. With the default network and the integrated teammate sources, the model is the only unavailable component until the bundle is supplied. Road network readiness refers to the default network validated at startup. Model readiness reflects bundle loading and metadata validation. Accuracy requires separate evaluation.

## Prediction request

`POST /predict` accepts this JSON request.

```json
{
  "network_id": "00000001",
  "observations": [
    {"longitude": 8.595203, "latitude": 49.844053, "timestamp_s": 1},
    {"longitude": 8.595294, "latitude": 49.844441, "timestamp_s": 2}
  ]
}
```

The eight-digit network ID selects `<root>/<network_id>/<network_id>.nodes` and `.arcs` on the server. Network IDs and order of observations are preserved. Longitude and latitude are finite numbers in degrees within ±180 and ±90. Timestamps are finite numbers in seconds on a shared time reference and must be strictly increasing. At least one observation is required. Repeated coordinates at different times are valid. Large time gaps are passed to the decoder. Unknown fields are rejected.

## Prediction response

HTTP 200 is reserved for a completed inference:

| Field | Meaning |
| --- | --- |
| `network_id` | The registered network used |
| `matches` | One slot per input observation. A match has `gps_index`, `arc_id`, `offset_m`, and the model's `probability`. An unmatched slot is null. |
| `routes` | Ordered arrays of directed arc IDs, one per contiguous segment. May include connecting arcs between observation matches. |
| `breaks` | Objects with `gps_index` and decoder `reason`, such as `no_candidates`, `disconnected`, or `gps_gap` |
| `unmatched_gps_indices` | Zero-based input indices whose match is null |
| `diagnostics` | Counts for GPS observations, candidate rows, model predictions, matched and unmatched observations, route segments, excluded zero-length arcs, and elapsed milliseconds |

### Error responses

| Status | Condition | Body |
| --- | --- | --- |
| 404 | Network ID is absent from the configured network directory | `{"detail": "..."}` |
| 422 | Request JSON or values fail validation | `detail` array with `loc`, `msg`, and `type` for each error |
| 500 | Runtime prediction failure or invalid model output | `{"detail": "..."}` |
| 503 | Required component is unavailable or the selected network is malformed | `{"detail": "..."}` |

With the model bundle absent, the example request returns HTTP 503.

```json
{"detail":"Missing model artifacts: final_model.joblib, feature_columns.json, config.json"}
```

A malformed metadata file or missing model runtime dependency is reported through the model's `detail` in `/ready`. Supply the trusted bundle under `PATHSNAP_MODEL_DIR` and restart the service after installing its training dependency versions.


## Browser evaluator

| Method and path | Behavior |
| --- | --- |
| `GET /ui` | Serves the browser client from Laptop 1 for use on Laptop 2 |
| `GET /evaluation/trips` | Returns sorted IDs with an available track and network under `PATHSNAP_EVALUATOR_DIR` |
| `POST /evaluation/{trip_id}` | Runs prediction on the registered server-side track. The request has the trip ID in its path. |

The evaluation result holds the standard `prediction` object, `gps_geometry` as longitude/latitude pairs, one polyline per segment in `route_geometry`, and `scores`. Score fields are point agreement with the nearest routable reference arc, and precision, recall and F1 of unique directed-arc overlap. Point agreement uses all observed rows, including unmatched rows in the denominator. Reference-derived scores describe this implementation's scoring method. Confirm any organiser-specific scoring rule.

Private reference data is read after prediction and only aggregate scores leave the server. An unlabeled trip returns null scores. An unconfigured evaluator directory or unavailable model returns 503. Unknown trip IDs return 404.
