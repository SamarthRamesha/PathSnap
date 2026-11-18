"""Model adapter unit tests using only explicit test doubles."""

import json

import numpy as np
import pytest

from src.api.model import CandidateModel, ModelContractError


class CandidateClassifierDouble:
    classes_ = np.array([1, 0])
    feature_names_in_ = ["heading_deg", "candidate_rank"]
    n_features_in_ = 2

    def predict_proba(self, features):
        assert list(features.columns) == self.feature_names_in_
        assert list(features.index) == [0, 1]
        positive = features["heading_deg"].to_numpy() / 100
        return np.column_stack((positive, 1 - positive))


ROWS = [
    {"trip_id": "sample", "point_idx": 2, "arc_id": 9, "candidate_rank": 1, "heading_deg": 80},
    {"trip_id": "sample", "point_idx": 2, "arc_id": 4, "candidate_rank": 2, "heading_deg": 20},
]
COLUMNS = ["heading_deg", "candidate_rank"]


def test_adapter_orders_features_excludes_identifiers_and_preserves_candidate_order():
    model = CandidateModel.from_pipeline(CandidateClassifierDouble(), COLUMNS)
    assert model.positive_index == 0
    assert model.predict_probabilities(ROWS) == [0.8, 0.2]
    assert model.predict_probabilities([]) == []


@pytest.mark.parametrize("columns", [
    [], ["arc_id"], ["trip_id"], ["point_idx"], ["label"],
    ["candidate_rank", "candidate_rank"], ["candidate_rank", "heading_deg"], ["missing"],
])
def test_adapter_rejects_unsupported_metadata(columns):
    with pytest.raises(ModelContractError):
        CandidateModel.from_pipeline(CandidateClassifierDouble(), columns)


@pytest.mark.parametrize("classes", [[0, 2], [1, 1], [1], [0, 1, 2], [1, np.nan]])
def test_adapter_rejects_invalid_classes(classes):
    pipeline_double = CandidateClassifierDouble()
    pipeline_double.classes_ = classes
    with pytest.raises(ModelContractError, match="class"):
        CandidateModel.from_pipeline(pipeline_double, COLUMNS)


@pytest.mark.parametrize(("heading", "message"), [
    ("80", "must be numeric"),
    (np.inf, "must be finite"),
])
def test_adapter_rejects_invalid_feature_values(heading, message):
    model = CandidateModel.from_pipeline(CandidateClassifierDouble(), COLUMNS)
    rows = [{**ROWS[0], "heading_deg": heading}, ROWS[1]]
    with pytest.raises(ModelContractError, match=message):
        model.predict_probabilities(rows)


@pytest.mark.parametrize("scores", [
    [[0.4, 0.6]], [[0.4], [0.1]], [[0.4, 0.6], [0.1]], [[0.4, 0.6], [0.1, np.nan]],
    [[0.4, 0.6], [0.1, np.inf]], [[0.4, 0.6], [0.1, 1.1]], [["0.4", "0.6"], ["0.1", "0.9"]],
])
def test_adapter_rejects_invalid_prediction_output(scores):
    pipeline_double = CandidateClassifierDouble()
    pipeline_double.predict_proba = lambda _frame: scores
    model = CandidateModel.from_pipeline(pipeline_double, COLUMNS)
    with pytest.raises(ModelContractError, match="predict_proba|probabilities"):
        model.predict_probabilities(ROWS)


def test_bundle_checks_metadata_and_reports_missing_files(tmp_path, monkeypatch):
    with pytest.raises(ModelContractError, match="final_model.joblib, feature_columns.json, config.json"):
        CandidateModel.load(tmp_path)
    model_file = tmp_path / "final_model.joblib"
    model_file.write_text("Test sentinel. This is not a serialized model.", encoding="utf-8")
    columns_file = tmp_path / "feature_columns.json"
    columns_file.write_text(json.dumps(COLUMNS), encoding="utf-8")
    config = tmp_path / "config.json"

    def load_test_double(path):
        assert path == model_file
        return CandidateClassifierDouble()

    monkeypatch.setattr("src.api.model.joblib.load", load_test_double)
    config.write_text('{"candidate_limit": 7}', encoding="utf-8")
    with pytest.raises(ModelContractError, match="candidate_limit"):
        CandidateModel.load(tmp_path)
    config.write_text('{"candidate_limit": 8, "positive_class": 1}', encoding="utf-8")
    assert CandidateModel.load(tmp_path).predict_probabilities(ROWS) == [0.8, 0.2]
    columns_file.write_text("[", encoding="utf-8")
    with pytest.raises(ModelContractError, match=r"valid JSON from feature_columns\.json"):
        CandidateModel.load(tmp_path)


def test_exported_label_and_dtype_are_used_instead_of_defaults(tmp_path, monkeypatch):
    (tmp_path / "final_model.joblib").touch()
    (tmp_path / "feature_columns.json").write_text(json.dumps(COLUMNS), encoding="utf-8")
    config = {
        "feature_columns": COLUMNS,
        "feature_count": 2,
        "candidate_limit": 8,
        "positive_class_label": 0,
        "positive_class_index": 1,
        "classes": [1, 0],
        "training_configuration": {"feature_dtype": "float32"},
    }
    pipeline = CandidateClassifierDouble()
    predict = pipeline.predict_proba

    def require_dtype(features):
        assert all(dtype == np.dtype("float32") for dtype in features.dtypes)
        return predict(features)

    monkeypatch.setattr(pipeline, "predict_proba", require_dtype)
    monkeypatch.setattr("src.api.model.joblib.load", lambda _: pipeline)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    model = CandidateModel.load(tmp_path)
    assert model.positive_index == 1
    assert model.predict_probabilities(ROWS) == pytest.approx([0.2, 0.8])
    config["positive_class_index"] = 0
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ModelContractError, match="fitted positive class index"):
        CandidateModel.load(tmp_path)
