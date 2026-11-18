from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .evaluation import registered_trip_ids, router as evaluation_router
from .pipeline import InferencePipeline, Settings
from .prediction import router as prediction_router
from .schemas import HealthResponse, ReadinessResponse

APP_NAME = "PathSnap"


def create_app(settings: Settings | None = None, *, demo: bool = False) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        config = settings or Settings.from_environment()
        pipeline = InferencePipeline(config)
        application.state.pipeline = pipeline
        application.state.evaluator_directory = config.evaluator_directory
        if demo:
            logger = logging.getLogger("uvicorn.error")
            for name, state in pipeline.components:
                logger.info("%s: %s", name, "ready" if state.available else state.detail)
            if not pipeline.ready:
                raise RuntimeError("PathSnap inference components are unavailable")
            assert pipeline.model is not None
            logger.info(
                "Trained model loaded: %d ordered features, class %s",
                len(pipeline.model.feature_columns),
                pipeline.model.pipeline.classes_[pipeline.model.positive_index],
            )
            directory = config.evaluator_directory
            if directory is None or not directory.is_dir():
                raise RuntimeError("Set PATHSNAP_EVALUATOR_DIR to a readable evaluation directory")
            trips = registered_trip_ids(directory.resolve())
            if not trips:
                raise RuntimeError("The configured evaluator directory contains no complete trips")
            logger.info("Evaluation dataset ready: %d trips", len(trips))
        yield

    application = FastAPI(title=APP_NAME, lifespan=lifespan)

    @application.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, exc: RequestValidationError) -> JSONResponse:
        # Rejected input can contain nonfinite numbers that cannot be echoed as JSON.
        errors = [
            {key: error[key] for key in ("loc", "msg", "type")}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

    @application.get("/health")
    async def health(request: Request) -> HealthResponse:
        pipeline = request.app.state.pipeline
        return HealthResponse(status="ok", app=APP_NAME, components=pipeline.components)

    @application.get(
        "/ready",
        responses={503: {"model": ReadinessResponse, "description": "A component is unavailable"}},
    )
    async def ready(request: Request, response: Response) -> ReadinessResponse:
        pipeline = request.app.state.pipeline
        if not pipeline.ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(
            status="ready" if pipeline.ready else "unavailable",
            components=pipeline.components,
        )

    application.include_router(prediction_router)
    application.include_router(evaluation_router)
    return application


app = create_app()
