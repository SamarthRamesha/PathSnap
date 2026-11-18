"""HTTP integration tests covering startup, validation, and unavailable inference."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.model import CandidateModel
from src.api.pipeline import Settings

NETWORK_ID = "00000001"
OBSERVATION = {"longitude": 0, "latitude": 0, "timestamp_s": 1}


@pytest.fixture
def network_root(tmp_path):
    network = tmp_path / NETWORK_ID
    network.mkdir()
    (network / f"{NETWORK_ID}.nodes").write_text("0 0\n0.001 0\n", encoding="utf-8")
    (network / f"{NETWORK_ID}.arcs").write_text("0 1\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def client(network_root):
    app = create_app(Settings(network_root=network_root, model_directory=network_root / "missing"))
    with TestClient(app) as test_client:
        yield test_client


def test_health_starts_without_model_and_reports_component_state(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app"] == "PathSnap"
    assert all(
        body["components"][component]["available"]
        for component in ("road_network", "feature_generator", "hmm_decoder")
    )
    assert not body["components"]["model"]["available"]


def test_readiness_reports_missing_artifacts(client):
    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"
    model = response.json()["components"]["model"]
    for name in ("final_model.joblib", "feature_columns.json", "config.json"):
        assert name in model["detail"]


def test_readiness_names_a_missing_model_runtime_dependency(network_root, monkeypatch):
    def raise_missing_dependency(_directory):
        raise ModuleNotFoundError(name="lightgbm")

    monkeypatch.setattr(CandidateModel, "load", raise_missing_dependency)
    app = create_app(Settings(network_root=network_root))
    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    model_status = response.json()["components"]["model"]
    assert not model_status["available"]
    assert "lightgbm" in model_status["detail"]


def test_prediction_returns_unavailable_without_artifacts(client):
    response = client.post(
        "/predict", json={"network_id": NETWORK_ID, "observations": [OBSERVATION]}
    )

    assert response.status_code == 503
    assert "Missing model artifacts" in response.json()["detail"]
    assert "matches" not in response.json()


@pytest.mark.parametrize("observations", [
    [],
    [{"latitude": 0, "timestamp_s": 1}],
    [{**OBSERVATION, "longitude": 181}],
    [{**OBSERVATION, "latitude": -91}],
    [{**OBSERVATION, "longitude": True}],
    [{**OBSERVATION, "timestamp_s": "wrong"}],
    [{**OBSERVATION, "timestamp_s": "12"}],
    [OBSERVATION, OBSERVATION],
    [OBSERVATION, {**OBSERVATION, "timestamp_s": 0}],
])
def test_invalid_prediction_requests_return_422(client, observations):
    response = client.post(
        "/predict", json={"network_id": NETWORK_ID, "observations": observations}
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"]


@pytest.mark.parametrize("invalid_number", ["NaN", "Infinity", "-Infinity", "1e309"])
def test_nonfinite_payload_returns_serializable_422(client, invalid_number):
    body = (
        '{"network_id":"00000001","observations":'
        '[{"longitude":0,"latitude":0,"timestamp_s":' + invalid_number + '}]}'
    )
    response = client.post("/predict", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == "timestamp_s"


def test_invalid_json_returns_422(client):
    response = client.post("/predict", content="{", headers={"content-type": "application/json"})
    assert response.status_code == 422


def test_unknown_network_and_path_input(client):
    for name, status in (("99999999", 404), ("../private", 422)):
        response = client.post("/predict", json={"network_id": name, "observations": [OBSERVATION]})
        assert response.status_code == status


def test_invalid_network_is_reported_without_exposing_server_path(client, tmp_path):
    path = tmp_path / "00000002"
    path.mkdir()
    (path / "00000002.nodes").write_text("0 0\n", encoding="utf-8")
    (path / "00000002.arcs").write_text("0 1\n", encoding="utf-8")

    response = client.post(
        "/predict", json={"network_id": "00000002", "observations": [OBSERVATION]}
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Arc 0 references a missing node"}


def test_predict_openapi_documents_schema_and_status_codes(client):
    assert client.get("/docs").status_code == 200
    operation = client.get("/openapi.json").json()["paths"]["/predict"]["post"]
    assert set(operation["responses"]) == {"200", "404", "422", "500", "503"}
    assert operation["requestBody"]["required"]



def test_evaluator_client_loads_without_private_evaluation_data(client):
    response = client.get("/ui")
    assert response.status_code == 200
    assert "PathSnap evaluator" in response.text
    assert client.get("/evaluation/trips").status_code == 503


def test_relative_settings_are_based_on_project_root(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATHSNAP_EVALUATOR_DIR", "data/student_test")
    monkeypatch.setenv("PATHSNAP_NETWORK_DIR", "data/train")
    monkeypatch.setenv("PATHSNAP_MODEL_DIR", "model_artifacts")
    root = Path(__file__).resolve().parents[1]
    settings = Settings.from_environment()
    assert settings.evaluator_directory == root / "data" / "student_test"
    assert settings.network_root == root / "data" / "train"
    assert settings.model_directory == root / "model_artifacts"
