"""Trip selection and server-side evaluation for the project's browser client."""

from collections.abc import Callable
from pathlib import Path
from re import fullmatch

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .network import DatasetError, RoadNetwork, load_network, load_nodes, load_track
from .prediction import run_prediction
from .schemas import GpsObservation, NetworkId, PredictionRequest, PredictionResponse

router = APIRouter()
_UI = Path(__file__).resolve().parents[2] / "frontend" / "evaluator.html"
Point = tuple[float, float]


class EvaluationScores(BaseModel):
    point_reference_agreement: float
    route_arc_precision: float
    route_arc_recall: float
    route_arc_f1: float


class EvaluationResult(BaseModel):
    prediction: PredictionResponse
    gps_geometry: list[Point]
    route_geometry: list[list[Point]]
    scores: EvaluationScores | None


def _evaluator_root(request: Request) -> Path:
    root = request.app.state.evaluator_directory
    if root is None or not root.is_dir():
        raise HTTPException(status_code=503, detail="Configure PATHSNAP_EVALUATOR_DIR on the API host")
    return root.resolve()


def _trip_file(root: Path, trip_id: str, extension: str) -> Path:
    name = f"{trip_id}.{extension}"
    path = (root / trip_id / name).resolve()
    if not path.is_relative_to(root) or path.name != name:
        raise HTTPException(status_code=404, detail="Unknown evaluation trip")
    return path


@router.get("/ui", include_in_schema=False)
def evaluator_ui():
    return FileResponse(_UI, media_type="text/html")


def registered_trip_ids(root: Path) -> list[str]:
    registered = []
    for directory in sorted(root.iterdir()):
        trip_id = directory.name
        if not fullmatch(r"[0-9]{8}", trip_id) or not directory.resolve().is_relative_to(root):
            continue
        try:
            files = [_trip_file(root, trip_id, suffix) for suffix in ("track", "nodes", "arcs")]
        except HTTPException:
            continue
        if all(path.is_file() for path in files):
            registered.append(trip_id)
    return registered


@router.get("/evaluation/trips")
def evaluation_trips(request: Request) -> list[str]:
    return registered_trip_ids(_evaluator_root(request))


def _read_reference(path: Path, network: RoadNetwork) -> set[int]:
    raw_arc_count = len(network.arcs) + network.zero_length_arcs
    try:
        with path.open(encoding="utf-8") as source:
            arc_ids = {int(line) for line in source}
    except (ValueError, OSError) as exc:
        raise DatasetError("The private reference has invalid arc records") from exc
    if not arc_ids or min(arc_ids) < 0 or max(arc_ids) >= raw_arc_count:
        raise DatasetError("The private reference contains no route or an unknown arc")
    routable = arc_ids.intersection(network.arcs)
    if not routable:
        raise DatasetError("The private reference contains no routable arcs")
    return routable


def _score(
    reference_ids: set[int],
    network: RoadNetwork,
    nodes: list[Point],
    gps: list[GpsObservation],
    prediction: PredictionResponse,
    segment_geometry: Callable[[Point, Point, Point, float], tuple[float, float]],
) -> EvaluationScores:
    reference = [
        (arc_id, nodes[network.arcs[arc_id].source], nodes[network.arcs[arc_id].target])
        for arc_id in reference_ids
    ]
    origin_lat = sum(row.latitude for row in gps) / len(gps)
    correct = 0
    for point, match in zip(gps, prediction.matches, strict=True):
        if match is None:
            continue
        position = (point.longitude, point.latitude)
        _, nearest = min(
            (segment_geometry(position, start, end, origin_lat)[0], arc_id)
            for arc_id, start, end in reference
        )
        correct += match.arc_id == nearest

    predicted = {arc_id for segment in prediction.routes for arc_id in segment}
    overlap = len(predicted & reference_ids)
    return EvaluationScores(
        point_reference_agreement=correct / len(gps),
        route_arc_precision=overlap / len(predicted) if predicted else 0.0,
        route_arc_recall=overlap / len(reference_ids),
        route_arc_f1=2 * overlap / (len(predicted) + len(reference_ids)),
    )


@router.post("/evaluation/{trip_id}")
def evaluate_trip(trip_id: NetworkId, request: Request) -> EvaluationResult:
    root = _evaluator_root(request)
    track = _trip_file(root, trip_id, "track")
    if not track.is_file():
        raise HTTPException(status_code=404, detail="Unknown evaluation trip")
    try:
        observations = load_track(track)
        pipeline = request.app.state.pipeline
        prediction = run_prediction(
            pipeline,
            PredictionRequest(network_id=trip_id, observations=observations),
            network_root=root,
        )
        network = load_network(root, trip_id)
        nodes = load_nodes(network.nodes_path)
        geometry = [
            [nodes[network.arcs[segment[0]].source], *(
                nodes[network.arcs[arc_id].target] for arc_id in segment
            )]
            for segment in prediction.routes
        ]
        reference = _trip_file(root, trip_id, "route")
        scores = (
            _score(
                _read_reference(reference, network), network, nodes, observations,
                prediction, pipeline.features.segment_geometry,
            )
            if reference.is_file() else None
        )
    except DatasetError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return EvaluationResult(
        prediction=prediction,
        gps_geometry=[(point.longitude, point.latitude) for point in observations],
        route_geometry=geometry,
        scores=scores,
    )
