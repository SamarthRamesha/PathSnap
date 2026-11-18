"""Real teammate components with controlled probabilities, never a trained model."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient

from data_exploration.code import generate_candidate_features as feature_module
from src.api.main import create_app
from src.api.model import IDENTIFIERS, MODEL_FIELDS, CandidateModel
from src.api.network import load_network
from src.api.pipeline import Settings, generate_candidates
from src.api.schemas import PredictionRequest

NETWORK_ID = "00000001"


@pytest.fixture
def synthetic_trip(tmp_path):
    trip = tmp_path / NETWORK_ID
    trip.mkdir()
    (trip / f"{NETWORK_ID}.nodes").write_text(
        "0 0\n0.0001 0\n0.0002 0\n0.25 0\n0.2501 0\n", encoding="utf-8"
    )
    (trip / f"{NETWORK_ID}.arcs").write_text("0 1\n1 2\n3 4\n", encoding="utf-8")
    return trip


def trajectory(*pairs):
    return {
        "network_id": NETWORK_ID,
        "observations": [
            {"longitude": longitude, "latitude": 0, "timestamp_s": timestamp}
            for longitude, timestamp in pairs
        ],
    }


def test_unlabeled_feature_generation_matches_training_geometry(synthetic_trip, monkeypatch):
    payload = PredictionRequest.model_validate(trajectory((0.00002, 1), (0.00017, 2)))
    network = load_network(synthetic_trip.parent, NETWORK_ID)

    reads = []
    read_rows = feature_module.read_rows

    def track_file_read(path):
        reads.append(path.suffix)
        return read_rows(path)

    monkeypatch.setattr(feature_module, "read_rows", track_file_read)
    inference = generate_candidates(feature_module, payload, network)
    assert reads == [".track", ".nodes", ".arcs"]
    assert len(inference) == 4
    assert [row["point_idx"] for row in inference] == [0, 0, 1, 1]
    assert [row["arc_id"] for row in inference] == [0, 1, 1, 0]
    assert [row["candidate_rank"] for row in inference] == [1, 2, 1, 2]
    assert all(tuple(row) == (*IDENTIFIERS, *MODEL_FIELDS) for row in inference)
    assert not list(synthetic_trip.glob("*.route"))

    track = synthetic_trip / f"{NETWORK_ID}.track"
    route = synthetic_trip / f"{NETWORK_ID}.route"
    track.write_text("0.00002 0 1\n0.00017 0 2\n", encoding="utf-8")
    route.write_text("0\n1\n", encoding="utf-8")  # Synthetic training labels only.
    training, _ = feature_module.generate_feature_rows(synthetic_trip, include_label=True)
    assert ".route" in reads
    assert [row.pop("label") for row in training] == [1, 0, 1, 0]
    assert training == inference


def test_candidate_limit_and_order_on_a_dense_fixture(tmp_path):
    trip = tmp_path / NETWORK_ID
    trip.mkdir()
    (trip / f"{NETWORK_ID}.nodes").write_text("0 0\n0.001 0\n", encoding="utf-8")
    (trip / f"{NETWORK_ID}.arcs").write_text("0 1\n" * 10, encoding="utf-8")
    payload = PredictionRequest.model_validate(trajectory((0.0005, 1)))
    rows = generate_candidates(feature_module, payload, load_network(tmp_path, NETWORK_ID))

    assert [row["arc_id"] for row in rows] == list(range(8))
    assert [row["candidate_rank"] for row in rows] == list(range(1, 9))


class NearestRankClassifierDouble:
    """Controlled scores from candidate rank, using the experiment's 13-column input."""

    classes_ = np.array([1, 0])
    feature_names_in_ = [
        "timestamp_gap_s",
        "step_distance_m",
        "point_speed_mps",
        "heading_deg",
        "turn_angle_deg",
        "perpendicular_distance_m",
        "along_arc_fraction",
        "arc_length_m",
        "arc_bearing_deg",
        "heading_delta_deg",
        "arc_out_degree",
        "arc_in_degree",
        "candidate_rank",
    ]

    def predict_proba(self, features):
        assert list(features.columns) == self.feature_names_in_
        chosen = np.where(features["candidate_rank"].to_numpy() == 1, 0.9, 0.0)
        return np.column_stack((chosen, 1 - chosen))


@pytest.fixture
def team_client(synthetic_trip, monkeypatch):
    classifier = NearestRankClassifierDouble()
    model_double = CandidateModel.from_pipeline(classifier, classifier.feature_names_in_)
    monkeypatch.setattr(CandidateModel, "load", lambda _directory: model_double)
    application = create_app(Settings(
        network_root=synthetic_trip.parent, evaluator_directory=synthetic_trip.parent
    ))
    with TestClient(application) as client:
        yield client


def test_real_decoder_serializes_segments_and_unmatched_rows(team_client):
    readiness = team_client.get("/ready")
    assert readiness.status_code == 200
    assert all(value["available"] for value in readiness.json()["components"].values())

    response = team_client.post("/predict", json=trajectory(
        (0.00004, 1), (0.00016, 2), (0.25005, 3), (1.0, 4), (0.25008, 100), (0.25009, 140),
    ))

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["routes"] == [[0, 1], [2], [2], [2]]
    assert result["breaks"] == [
        {"gps_index": 2, "reason": "disconnected"},
        {"gps_index": 3, "reason": "no_candidates"},
        {"gps_index": 5, "reason": "gps_gap"},
    ]
    assert result["unmatched_gps_indices"] == [3]
    assert result["matches"][3] is None
    assert [match["arc_id"] for match in result["matches"] if match is not None] == [0, 1, 2, 2, 2]
    assert result["matches"][0]["offset_m"] == pytest.approx(4.4478, abs=0.001)
    assert result["matches"][0]["probability"] == 0.9
    assert result["diagnostics"]["candidate_rows"] == result["diagnostics"]["model_predictions"] == 7
    assert result["diagnostics"]["gps_observations"] == 6
    assert result["diagnostics"]["matched_observations"] == 5
    assert result["diagnostics"]["route_segments"] == 4


def test_real_decoder_preserves_all_unmatched_observations(team_client):
    response = team_client.post("/predict", json=trajectory((1.0, 1), (1.1, 2)))

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["matches"] == [None, None]
    assert result["routes"] == []
    assert result["unmatched_gps_indices"] == [0, 1]
    assert result["breaks"] == [
        {"gps_index": 0, "reason": "no_candidates"},
        {"gps_index": 1, "reason": "no_candidates"},
    ]
    assert result["diagnostics"]["candidate_rows"] == result["diagnostics"]["model_predictions"] == 0


def test_real_decoder_respects_directed_arcs_and_concurrent_requests(team_client):
    forward = trajectory((0.00003, 1), (0.00008, 2))
    backward = trajectory((0.00008, 1), (0.00003, 2))

    def score(payload):
        response = team_client.post("/predict", json=payload)
        assert response.status_code == 200, response.text
        return response.json()

    with ThreadPoolExecutor(max_workers=2) as executor:
        follows_arc, wrong_way = executor.map(score, (forward, backward))

    assert follows_arc["routes"] == [[0]]
    assert follows_arc["breaks"] == []
    assert wrong_way["routes"] == [[0], [0]]
    assert wrong_way["breaks"] == [{"gps_index": 1, "reason": "disconnected"}]



def test_evaluator_scores_privately_after_real_feature_and_decoder_pipeline(
    team_client, synthetic_trip, monkeypatch
):
    track = synthetic_trip / f"{NETWORK_ID}.track"
    route = synthetic_trip / f"{NETWORK_ID}.route"
    track.write_text("0.00004 0 1\n0.00016 0 2\n", encoding="utf-8")
    read_rows = feature_module.read_rows
    reads = []

    def track_file_read(path):
        reads.append(path.suffix)
        return read_rows(path)

    monkeypatch.setattr(feature_module, "read_rows", track_file_read)
    assert team_client.get("/evaluation/trips").json() == [NETWORK_ID]
    unlabeled = team_client.post(f"/evaluation/{NETWORK_ID}")
    assert unlabeled.status_code == 200 and unlabeled.json()["scores"] is None

    route.write_text("0\n1\n", encoding="utf-8")
    scored = team_client.post(f"/evaluation/{NETWORK_ID}")
    assert scored.status_code == 200, scored.text
    result = scored.json()
    assert reads == [".track", ".nodes", ".arcs"] * 2
    assert result["prediction"]["matches"] == unlabeled.json()["prediction"]["matches"]
    assert result["prediction"]["routes"] == [[0, 1]]
    assert result["route_geometry"] == [[[0, 0], [0.0001, 0], [0.0002, 0]]]
    assert result["scores"] == dict(
        point_reference_agreement=1.0, route_arc_precision=1.0,
        route_arc_recall=1.0, route_arc_f1=1.0,
    )
    assert set(result) == {"prediction", "gps_geometry", "route_geometry", "scores"}
    assert team_client.post("/evaluation/99999999").status_code == 404
