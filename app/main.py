import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from app.config import settings
from app.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)
from app.model import model_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown actions."""
    logger.info("Initializing application...")
    try:
        # Pre-load the pre-trained Hugging Face model on server startup
        model_service.load_model()
        logger.info("Application startup completed successfully.")
    except Exception as e:
        logger.error("Failed to load model on startup: %s", str(e))
    yield
    logger.info("Shutting down application...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.get(
    "/",
    tags=["Root"],
    summary="Root Endpoint",
    description="Returns service information and documentation links."
)
async def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "health_check": "/health",
        "endpoints": {
            "predict": "POST /predict",
            "batch_predict": "POST /predict/batch"
        }
    }

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health Check",
    description="Confirms that the service is alive and model is loaded."
)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_name=model_service.model_name,
        model_loaded=model_service.is_loaded
    )

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Single Text Inference",
    description="Accepts a single string and returns sentiment classification (e.g. POSITIVE/NEGATIVE)."
)
async def predict(request: PredictionRequest):
    try:
        result = model_service.predict(request.text)
        return PredictionResponse(**result)
    except Exception as e:
        logger.exception("Error during prediction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )

@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Inference"],
    summary="Batch Text Inference",
    description="Accepts a list of strings and returns sentiment classification for each."
)
async def predict_batch(request: BatchPredictionRequest):
    try:
        results = model_service.predict_batch(request.texts)
        prediction_responses = [PredictionResponse(**r) for r in results]
        return BatchPredictionResponse(
            predictions=prediction_responses,
            total_processed=len(prediction_responses)
        )
    except Exception as e:
        logger.exception("Error during batch prediction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(e)}"
        )
