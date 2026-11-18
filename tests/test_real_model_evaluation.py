"""Exercise the shipped LightGBM model, generator and decoder on a real unlabeled trip."""

from pathlib import Path
from shutil import copyfile

from fastapi.testclient import TestClient

from hmm import viterbi
from src.api.main import create_app
from src.api.model import CandidateModel
from src.api.network import load_network
from src.api.pipeline import Settings


ROOT = Path(__file__).resolve().parents[1]
TRIP_ID = "00000052"


def test_model_probabilities_reach_hmm_before_any_reference_read(tmp_path, monkeypatch):
    source = ROOT / "data" / "student_test" / TRIP_ID
    trip = tmp_path / TRIP_ID
    trip.mkdir()
    for suffix in ("track", "nodes", "arcs"):
        name = f"{TRIP_ID}.{suffix}"
        copyfile(source / name, trip / name)
    reference = trip / f"{TRIP_ID}.route"
    assert not reference.exists()

    measured = []
    events = []
    predict_probabilities = CandidateModel.predict_probabilities
    decode_probabilities = viterbi.decode_probabilities
    open_file = Path.open

    def record_model_output(self, rows):
        probabilities = predict_probabilities(self, rows)
        assert len(self.feature_columns) == 13 and self.feature_dtype == "float32"
        assert self.pipeline.classes_[self.positive_index] == 1
        assert all(0 <= value <= 1 for value in probabilities)
        measured[:] = [
            (row["point_idx"], row["arc_id"], probability)
            for row, probability in zip(rows, probabilities, strict=True)
        ]
        events.append("model")
        return probabilities

    def check_decoder_input(gps_samples, groups, offsets, arcs):
        assert [
            (sample.gps_index, arc_id, probability)
            for sample, group in zip(gps_samples, groups, strict=True)
            for arc_id, probability in group
        ] == measured
        result = decode_probabilities(gps_samples, groups, offsets, arcs)
        events.append("decoder")
        return result

    def watch_reference(path, *args, **kwargs):
        if path.suffix == ".route":
            assert path == reference
            assert events[-2:] == ["model", "decoder"]
            events.append("reference")
        return open_file(path, *args, **kwargs)

    monkeypatch.setattr(CandidateModel, "predict_probabilities", record_model_output)
    monkeypatch.setattr(viterbi, "decode_probabilities", check_decoder_input)
    monkeypatch.setattr(Path, "open", watch_reference)
    settings = Settings(
        network_root=tmp_path,
        readiness_network_id=TRIP_ID,
        model_directory=ROOT / "model_artifacts",
        evaluator_directory=tmp_path,
    )

    with TestClient(create_app(settings, demo=True)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
        assert client.get("/evaluation/trips").json() == [TRIP_ID]
        assert 'fetch(path, options)' in client.get("/ui").text
        unlabeled = client.post(f"/evaluation/{TRIP_ID}")
        assert unlabeled.status_code == 200, unlabeled.text
        assert unlabeled.json()["scores"] is None
        assert events == ["model", "decoder"]

        copyfile(source / reference.name, reference)
        labeled = client.post(f"/evaluation/{TRIP_ID}")
        assert labeled.status_code == 200, labeled.text
        assert events == ["model", "decoder", "model", "decoder", "reference"]

    result = labeled.json()
    prediction = result["prediction"]
    network = load_network(tmp_path, TRIP_ID)
    assert prediction["matches"] == unlabeled.json()["prediction"]["matches"]
    assert prediction["routes"] == unlabeled.json()["prediction"]["routes"]
    assert len(prediction["matches"]) == len(result["gps_geometry"]) == 262
    assert prediction["diagnostics"]["model_predictions"] == 2096
    assert result["route_geometry"]
    for route, geometry in zip(prediction["routes"], result["route_geometry"], strict=True):
        assert len(geometry) == len(route) + 1
        assert all(network.arcs[a].target == network.arcs[b].source for a, b in zip(route, route[1:]))
    assert all(0 <= score <= 1 for score in result["scores"].values())
