import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    """Verify root endpoint returns HTTP 200 and API metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "docs_url" in data
    assert data["docs_url"] == "/docs"

def test_health_check_endpoint():
    """Verify health check endpoint returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_name" in data
    assert "model_loaded" in data

@patch("app.main.model_service.predict")
def test_predict_success(mock_predict):
    """Verify single prediction endpoint handles text input and returns sentiment."""
    mock_predict.return_value = {
        "text": "This automated CI/CD pipeline is fantastic!",
        "label": "POSITIVE",
        "score": 0.9998,
        "model_name": "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    }

    payload = {"text": "This automated CI/CD pipeline is fantastic!"}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "POSITIVE"
    assert data["score"] >= 0.5
    assert data["text"] == payload["text"]
    assert "model_name" in data

def test_predict_validation_empty_string():
    """Verify that empty text payload returns 422 Validation Error."""
    payload = {"text": ""}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_missing_payload():
    """Verify that missing body returns 422 Validation Error."""
    response = client.post("/predict", json={})
    assert response.status_code == 422

@patch("app.main.model_service.predict_batch")
def test_predict_batch_success(mock_predict_batch):
    """Verify batch prediction endpoint processes multiple inputs."""
    sample_texts = [
        "The model deployment succeeded.",
        "A deployment failure occurred."
    ]
    mock_predict_batch.return_value = [
        {
            "text": sample_texts[0],
            "label": "POSITIVE",
            "score": 0.9991,
            "model_name": "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
        },
        {
            "text": sample_texts[1],
            "label": "NEGATIVE",
            "score": 0.9985,
            "model_name": "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
        }
    ]

    payload = {"texts": sample_texts}
    response = client.post("/predict/batch", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["label"] == "POSITIVE"
    assert data["predictions"][1]["label"] == "NEGATIVE"

def test_predict_batch_empty_list():
    """Verify that an empty list for batch prediction returns 422 Validation Error."""
    payload = {"texts": []}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422

