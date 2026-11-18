"""Connect the independently developed feature, model and decoder components."""

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from math import cos, isclose, radians
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from threading import Lock
from time import perf_counter
from types import ModuleType

from .model import (
    CANDIDATE_LIMIT,
    IDENTIFIERS,
    MODEL_FIELDS,
    CandidateModel,
    ModelContractError,
)
from .network import DatasetError, RoadNetwork, load_network
from .schemas import (
    Components,
    ComponentStatus,
    GpsObservation,
    Match,
    PredictionDiagnostics,
    PredictionRequest,
    PredictionResponse,
    RouteBreak,
)

logger = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parents[2]
_FEATURE_MODULE = "data_exploration.code.generate_candidate_features"
_HMM_MODULE = "hmm.viterbi"


@dataclass(frozen=True)
class Settings:
    network_root: Path = _ROOT / "data" / "train"
    readiness_network_id: str = "00000001"
    model_directory: Path = _ROOT / "model_artifacts"
    evaluator_directory: Path | None = None

    @classmethod
    def from_environment(cls) -> "Settings":
        defaults = cls()
        evaluator = os.environ.get("PATHSNAP_EVALUATOR_DIR")
        return cls(
            network_root=(_ROOT / Path(os.environ.get("PATHSNAP_NETWORK_DIR", defaults.network_root)).expanduser()),
            readiness_network_id=os.environ.get(
                "PATHSNAP_READINESS_NETWORK_ID", defaults.readiness_network_id
            ),
            model_directory=(_ROOT / Path(os.environ.get("PATHSNAP_MODEL_DIR", defaults.model_directory)).expanduser()),
            evaluator_directory=(_ROOT / Path(evaluator).expanduser()) if evaluator else None,
        )


class ComponentError(RuntimeError):
    pass


def _load_component(name: str, verify: Callable[[ModuleType], bool]):
    try:
        module = import_module(name)
        if not verify(module):
            raise ComponentError(f"{name} has an incompatible interface")
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == name or name.startswith(f"{exc.name}.")):
            return None, ComponentStatus(available=False, detail=f"Install the team module {name}")
        logger.exception("Missing dependency for PathSnap component %s", name)
        return None, ComponentStatus(available=False, detail=f"{name} has a missing dependency")
    except Exception:
        logger.exception("Unable to load PathSnap component %s", name)
        return None, ComponentStatus(available=False, detail=f"{name} has an incompatible interface")
    return module, ComponentStatus(available=True)


def generate_candidates(
    feature_module: ModuleType, payload: PredictionRequest, network: RoadNetwork
) -> list[dict]:
    # The upstream generator takes a trip directory. Provide the three inference files.
    with TemporaryDirectory(prefix="pathsnap-") as directory:
        trip = Path(directory) / payload.network_id
        trip.mkdir()
        copyfile(network.nodes_path, trip / f"{payload.network_id}.nodes")
        copyfile(network.arcs_path, trip / f"{payload.network_id}.arcs")
        with (trip / f"{payload.network_id}.track").open("w", encoding="utf-8") as track:
            for observation in payload.observations:
                track.write(
                    f"{observation.longitude}\t{observation.latitude}\t{observation.timestamp_s}\n"
                )
        rows, summary = feature_module.generate_feature_rows(
            trip, candidate_limit=CANDIDATE_LIMIT, include_label=False
        )
    if (
        summary["gps_points_valid"] != len(payload.observations)
        or summary["route_arc_count"] != 0
        or summary["candidate_limit"] != CANDIDATE_LIMIT
    ):
        raise ComponentError("Feature generation did not preserve the unlabeled trip")
    return rows


def _project_observations(decoder: ModuleType, observations: list[GpsObservation]):
    origin_lat = sum(row.latitude for row in observations) / len(observations)
    lon_scale = 111_320.0 * cos(radians(origin_lat))
    first = observations[0]
    return [
        decoder.GpsSample(
            index,
            row.timestamp_s,
            ((row.longitude - first.longitude) * lon_scale,
             (row.latitude - first.latitude) * 111_320.0),
        )
        for index, row in enumerate(observations)
    ]


def _candidate_offsets(rows: list[dict], network: RoadNetwork, payload: PredictionRequest):
    offsets: list[dict[int, float]] = [{} for _ in payload.observations]
    last_index = -1
    for row in rows:
        if tuple(row) != (*IDENTIFIERS, *MODEL_FIELDS) or row["trip_id"] != payload.network_id:
            raise ComponentError("Unexpected candidate row schema or trip identifier")
        index, arc_id = row["point_idx"], row["arc_id"]
        if type(index) is not int or not 0 <= index < len(offsets) or index < last_index:
            raise ComponentError("Candidate rows are out of GPS observation order")
        if type(arc_id) is not int or arc_id not in network.arcs or arc_id in offsets[index]:
            raise ComponentError("A candidate references an unknown or duplicate arc")
        arc = network.arcs[arc_id]
        rank = len(offsets[index]) + 1
        if rank > CANDIDATE_LIMIT or row["candidate_rank"] != rank:
            raise ComponentError("Candidate ranks must be ordered and respect the candidate limit")
        fraction = row["along_arc_fraction"]
        if not 0 <= fraction <= 1 or not isclose(row["arc_length_m"], arc.length_m):
            raise ComponentError("Candidate geometry differs from the directed road network")
        offsets[index][arc_id] = fraction * arc.length_m
        last_index = index
    return offsets


def decode_candidates(
    decoder: ModuleType,
    payload: PredictionRequest,
    network: RoadNetwork,
    rows: list[dict],
    probabilities: list[float],
    offsets: list[dict[int, float]],
):
    """Preserve candidate order and an empty slot for GPS rows without candidates."""
    grouped: list[list[tuple[int, float]]] = [[] for _ in payload.observations]
    for row, score in zip(rows, probabilities, strict=True):
        grouped[row["point_idx"]].append((row["arc_id"], score))
    result = decoder.decode_probabilities(
        _project_observations(decoder, payload.observations),
        grouped,
        offsets,
        network.arcs.values(),
    )
    if len(result.matches) != len(payload.observations):
        raise ComponentError("The decoder returned a misaligned match sequence")
    matches = []
    for index, candidate in enumerate(result.matches):
        if candidate is None:
            matches.append(None)
        else:
            matches.append(Match(
                gps_index=index,
                arc_id=candidate.arc_id,
                offset_m=candidate.offset_m,
                probability=dict(grouped[index])[candidate.arc_id],
            ))
    breaks = [RouteBreak(gps_index=item.gps_index, reason=item.reason) for item in result.breaks]
    return matches, result.routes, breaks


class InferencePipeline:
    def __init__(self, settings: Settings):
        self.network_root = settings.network_root
        self._model_lock = Lock()
        try:
            load_network(self.network_root, settings.readiness_network_id)
        except DatasetError as exc:
            road_network = ComponentStatus(available=False, detail=str(exc))
        else:
            road_network = ComponentStatus(available=True)

        self.features, feature_status = _load_component(
            _FEATURE_MODULE,
            lambda module: callable(module.generate_feature_rows)
            and callable(module.segment_geometry)
            and tuple(module.INFERENCE_FEATURE_COLUMNS) == (*IDENTIFIERS, *MODEL_FIELDS),
        )
        self.decoder, decoder_status = _load_component(
            _HMM_MODULE,
            lambda module: callable(module.decode_probabilities) and callable(module.GpsSample),
        )
        self.model = None
        try:
            self.model = CandidateModel.load(settings.model_directory)
        except ModelContractError as exc:
            model_status = ComponentStatus(available=False, detail=str(exc))
        except ModuleNotFoundError as exc:
            logger.exception("Missing runtime dependency for the PathSnap model bundle")
            model_status = ComponentStatus(
                available=False,
                detail=f"Install the model's training version of the missing dependency: {exc.name}",
            )
        except Exception:
            logger.exception("Unable to load the PathSnap model bundle")
            model_status = ComponentStatus(
                available=False,
                detail="Check the model artifact metadata and install the matching model dependencies",
            )
        else:
            model_status = ComponentStatus(available=True)

        self.components = Components(
            road_network=road_network,
            feature_generator=feature_status,
            model=model_status,
            hmm_decoder=decoder_status,
        )

    @property
    def ready(self) -> bool:
        return all(component.available for _, component in self.components)

    def predict(
        self, payload: PredictionRequest, *, network_root: Path | None = None
    ) -> PredictionResponse:
        start = perf_counter()
        network = load_network(network_root or self.network_root, payload.network_id)
        blockers = [
            component.detail
            for name, component in self.components
            if name != "road_network" and not component.available
        ]
        if blockers:
            raise ComponentError("; ".join(detail for detail in blockers if detail))
        assert self.model is not None and self.features is not None and self.decoder is not None

        rows = generate_candidates(self.features, payload, network)
        offsets = _candidate_offsets(rows, network, payload)
        with self._model_lock:
            probabilities = self.model.predict_probabilities(rows)
        matches, routes, breaks = decode_candidates(
            self.decoder, payload, network, rows, probabilities, offsets
        )
        unmatched = [index for index, match in enumerate(matches) if match is None]
        return PredictionResponse(
            network_id=payload.network_id,
            matches=matches,
            routes=routes,
            breaks=breaks,
            unmatched_gps_indices=unmatched,
            diagnostics=PredictionDiagnostics(
                gps_observations=len(matches),
                candidate_rows=len(rows),
                model_predictions=len(probabilities),
                matched_observations=len(matches) - len(unmatched),
                unmatched_observations=len(unmatched),
                route_segments=len(routes),
                zero_length_arcs_excluded=network.zero_length_arcs,
                elapsed_ms=round((perf_counter() - start) * 1000, 3),
            ),
        )
