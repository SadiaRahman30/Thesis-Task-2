import logging
from typing import List, Dict, Any, Optional
from transformers import pipeline
from app.config import settings

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.MODEL_NAME
        self._pipeline = None

    def load_model(self):
        """Loads the Hugging Face sentiment analysis pipeline."""
        if self._pipeline is None:
            logger.info("Loading model: %s", self.model_name)
            self._pipeline = pipeline(
                task="sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name
            )
            logger.info("Model loaded successfully.")

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def predict(self, text: str) -> Dict[str, Any]:
        """Runs inference on a single text string."""
        if not self.is_loaded:
            self.load_model()
        
        # Pipeline returns list of dicts: [{'label': 'POSITIVE', 'score': 0.999}]
        result = self._pipeline(text)[0]
        return {
            "text": text,
            "label": result["label"],
            "score": round(float(result["score"]), 4),
            "model_name": self.model_name
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Runs inference on a list of text strings."""
        if not self.is_loaded:
            self.load_model()
            
        results = self._pipeline(texts)
        outputs = []
        for text, result in zip(texts, results):
            outputs.append({
                "text": text,
                "label": result["label"],
                "score": round(float(result["score"]), 4),
                "model_name": self.model_name
            })
        return outputs

# Global singleton instance
model_service = ModelService()
