from pydantic import BaseModel, Field
from typing import List

class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    model_name: str = Field(..., example="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    model_loaded: bool = Field(..., example=True)

class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Input text string for sentiment analysis",
        example="Deploying pre-trained machine learning models with CI/CD is efficient!"
    )

class PredictionResponse(BaseModel):
    text: str = Field(..., example="Deploying pre-trained machine learning models with CI/CD is efficient!")
    label: str = Field(..., example="POSITIVE")
    score: float = Field(..., example=0.9998)
    model_name: str = Field(..., example="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_items=1,
        max_items=50,
        description="List of text strings for batch analysis",
        example=[
            "This model deployment pipeline works seamlessly.",
            "I am having trouble with broken dependencies."
        ]
    )

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processed: int = Field(..., example=2)
