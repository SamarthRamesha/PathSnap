import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from .model import ModelContractError
from .network import DatasetError, NetworkNotFound
from .pipeline import ComponentError, InferencePipeline
from .schemas import ApiError, PredictionRequest, PredictionResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/predict",
    responses={
        404: {"model": ApiError, "description": "Unknown network"},
        500: {"model": ApiError, "description": "Inference failed"},
        503: {"model": ApiError, "description": "A required component or network is unavailable"},
    },
)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    return run_prediction(request.app.state.pipeline, payload)


def run_prediction(
    pipeline: InferencePipeline, payload: PredictionRequest, *, network_root: Path | None = None
) -> PredictionResponse:
    try:
        return pipeline.predict(payload, network_root=network_root)
    except NetworkNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (DatasetError, ComponentError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelContractError as exc:
        logger.exception("PathSnap model violated its inference contract")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("PathSnap prediction failed")
        raise HTTPException(status_code=500, detail="PathSnap prediction failed") from exc
