import os


class Settings:
    PROJECT_NAME: str = "Hugging Face Model Inference API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = (
        "Production-grade FastAPI service serving a Hugging Face "
        "sentiment analysis model with automated CI/CD."
    )
    # Default lightweight, high-performance sentiment analysis model
    DEFAULT_MODEL_NAME: str = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    MODEL_NAME: str = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

settings = Settings()

