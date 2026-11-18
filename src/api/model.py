"""Load a trusted candidate-classification artifact and validate its outputs."""

import json
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

CANDIDATE_LIMIT = 8
IDENTIFIERS = ("trip_id", "point_idx", "arc_id")
MODEL_FIELDS = (
    "longitude",
    "latitude",
    "timestamp",
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
)
ARTIFACT_FILES = ("final_model.joblib", "feature_columns.json", "config.json")


class ModelContractError(ValueError):
    pass


def _valid_class_label(value: object) -> bool:
    return isinstance(value, (str, int)) or (isinstance(value, float) and isfinite(value))


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelContractError(f"Could not read valid JSON from {path.name}") from exc


@dataclass(frozen=True)
class CandidateModel:
    pipeline: Any
    feature_columns: tuple[str, ...]
    positive_index: int
    feature_dtype: str | None = None

    @classmethod
    def load(cls, directory: Path) -> "CandidateModel":
        missing = [name for name in ARTIFACT_FILES if not (directory / name).is_file()]
        if missing:
            raise ModelContractError(f"Missing model artifacts: {', '.join(missing)}")
        model_path, columns_path, config_path = (directory / name for name in ARTIFACT_FILES)
        columns = _read_json(columns_path)
        config = _read_json(config_path)
        if not isinstance(config, dict):
            raise ModelContractError("config.json must contain an object")
        limit = config.get("candidate_limit", CANDIDATE_LIMIT)
        if type(limit) is not int or limit != CANDIDATE_LIMIT:
            raise ModelContractError(f"Model candidate_limit must be {CANDIDATE_LIMIT}")
        if "feature_columns" in config and config["feature_columns"] != columns:
            raise ModelContractError("config.json and feature_columns.json disagree on feature order")
        training = config.get("training_configuration", {})
        if not isinstance(training, dict):
            raise ModelContractError("training_configuration must be an object")
        dtype = training.get("feature_dtype")
        if dtype not in (None, "float32", "float64"):
            raise ModelContractError("Unsupported training feature dtype")
        label = config.get("positive_class_label", config.get("positive_class", 1))
        if "positive_class" in config and label != config["positive_class"]:
            raise ModelContractError("Export metadata has conflicting positive class labels")

        pipeline = joblib.load(model_path)
        model = cls.from_pipeline(pipeline, columns, label)
        if "feature_count" in config and config["feature_count"] != len(model.feature_columns):
            raise ModelContractError("Export metadata disagrees with the fitted feature count")
        if "classes" in config and config["classes"] != list(pipeline.classes_):
            raise ModelContractError("Export metadata disagrees with the fitted class order")
        if "positive_class_index" in config and config["positive_class_index"] != model.positive_index:
            raise ModelContractError("Export metadata disagrees with the fitted positive class index")
        return replace(model, feature_dtype=dtype)

    @classmethod
    def from_pipeline(
        cls, pipeline: Any, columns: object, positive_class: object = 1
    ) -> "CandidateModel":
        if (
            not isinstance(columns, list)
            or not columns
            or any(not isinstance(col, str) or col not in MODEL_FIELDS for col in columns)
            or len(set(columns)) != len(columns)
        ):
            raise ModelContractError(
                "feature_columns.json must be an ordered list of unique supported model features"
            )
        if not callable(getattr(pipeline, "predict_proba", None)):
            raise ModelContractError("Model must provide predict_proba")
        classes = np.asarray(getattr(pipeline, "classes_", None), dtype=object)
        if (
            classes.shape != (2,)
            or not all(_valid_class_label(label) for label in classes)
            or classes[0] == classes[1]
        ):
            raise ModelContractError("Model must provide two distinct classes in classes_")
        if not _valid_class_label(positive_class):
            raise ModelContractError("positive_class must be a scalar class label")
        positive_indices = [index for index, label in enumerate(classes) if label == positive_class]
        if len(positive_indices) != 1:
            raise ModelContractError("Model classes_ must contain the positive class exactly once")
        fitted_columns = getattr(pipeline, "feature_names_in_", None)
        if fitted_columns is not None and list(fitted_columns) != columns:
            raise ModelContractError("Model fitted column order differs from feature_columns.json")
        fitted_count = getattr(pipeline, "n_features_in_", len(columns))
        if fitted_count != len(columns):
            raise ModelContractError("Model fitted feature count differs from feature_columns.json")
        return cls(pipeline, tuple(columns), positive_indices[0])

    def predict_probabilities(self, rows: list[dict]) -> list[float]:
        if not rows:
            return []
        frame = pd.DataFrame.from_records(rows)
        if not set(self.feature_columns).issubset(frame.columns):
            raise ModelContractError("Candidate rows are missing model features")
        features = frame[list(self.feature_columns)]
        values = features.to_numpy()
        if values.dtype.kind not in "iuf":
            raise ModelContractError("Candidate features must be numeric")
        if not np.isfinite(values).all():
            raise ModelContractError("Candidate features must be finite")
        if self.feature_dtype is not None:
            features = features.astype(self.feature_dtype)
            if not np.isfinite(features.to_numpy()).all():
                raise ModelContractError("Candidate features exceed the training dtype range")
        output = self.pipeline.predict_proba(features)
        try:
            probabilities = np.asarray(output)
        except (ValueError, TypeError) as exc:
            raise ModelContractError("predict_proba must return a rectangular probability array") from exc
        if probabilities.shape != (len(rows), 2):
            raise ModelContractError("predict_proba must return a row per candidate and a column per class")
        if probabilities.dtype.kind not in "iuf" or not (
            np.isfinite(probabilities).all()
            and (probabilities >= 0).all()
            and (probabilities <= 1).all()
        ):
            raise ModelContractError("Candidate probabilities must be finite and in [0, 1]")
        return probabilities[:, self.positive_index].tolist()
