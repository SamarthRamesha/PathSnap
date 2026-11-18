from itertools import pairwise
from math import isfinite
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

NetworkId = Annotated[str, Field(pattern=r"^[0-9]{8}$")]
FiniteNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class GpsObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    longitude: Annotated[FiniteNumber, Field(ge=-180, le=180)]
    latitude: Annotated[FiniteNumber, Field(ge=-90, le=90)]
    timestamp_s: FiniteNumber


def validate_timestamps(observations: list[GpsObservation]) -> None:
    for index, (previous, current) in enumerate(pairwise(observations), start=1):
        gap = current.timestamp_s - previous.timestamp_s
        if gap <= 0 or not isfinite(gap):
            raise ValueError(f"GPS row {index} must have a later timestamp with a finite gap")


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    network_id: NetworkId
    observations: list[GpsObservation] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_observation_order(self) -> Self:
        validate_timestamps(self.observations)
        return self


class Match(BaseModel):
    gps_index: int
    arc_id: int
    offset_m: float
    probability: float


class RouteBreak(BaseModel):
    gps_index: int
    reason: str


class PredictionDiagnostics(BaseModel):
    gps_observations: int
    candidate_rows: int
    model_predictions: int
    matched_observations: int
    unmatched_observations: int
    route_segments: int
    zero_length_arcs_excluded: int
    elapsed_ms: float


class PredictionResponse(BaseModel):
    network_id: NetworkId
    matches: list[Match | None]
    routes: list[list[int]]
    breaks: list[RouteBreak]
    unmatched_gps_indices: list[int]
    diagnostics: PredictionDiagnostics


class ComponentStatus(BaseModel):
    available: bool
    detail: str | None = None


class Components(BaseModel):
    road_network: ComponentStatus
    feature_generator: ComponentStatus
    model: ComponentStatus
    hmm_decoder: ComponentStatus


class HealthResponse(BaseModel):
    status: Literal["ok"]
    app: str
    components: Components


class ReadinessResponse(BaseModel):
    status: Literal["ready", "unavailable"]
    components: Components


class ApiError(BaseModel):
    detail: str
