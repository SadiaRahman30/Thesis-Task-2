
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["healthy"])
    model_name: str = Field(..., examples=["distilbert/distilbert-base-uncased-finetuned-sst-2-english"])
    model_loaded: bool = Field(..., examples=[True])

class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Input text string for sentiment analysis",
        examples=["Deploying pre-trained machine learning models with CI/CD is efficient!"]
    )

class PredictionResponse(BaseModel):
    text: str = Field(..., examples=["Deploying pre-trained machine learning models with CI/CD is efficient!"])
    label: str = Field(..., examples=["POSITIVE"])
    score: float = Field(..., examples=[0.9998])
    model_name: str = Field(..., examples=["distilbert/distilbert-base-uncased-finetuned-sst-2-english"])

class BatchPredictionRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of text strings for batch analysis",
        examples=[[
            "This model deployment pipeline works seamlessly.",
            "I am having trouble with broken dependencies."
        ]]
    )

class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total_processed: int = Field(..., examples=[2])
