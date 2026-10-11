import sys
from pathlib import Path

# Add the project root (/content) to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

VALID_PATIENT = {
    "age": 55,
    "sex": 1,
    "cp": 2,
    "trestbps": 130,
    "chol": 250,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 0,
    "thal": 3
}


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_predict_valid_patient():
    response = client.post("/predict", json=VALID_PATIENT)
    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] in [0, 1]
    assert result["prediction_label"] in [
        "No disease", "Disease present"
    ]
    assert 0 <= result["probability_class_1"] <= 1


def test_predict_rejects_invalid_input():
    invalid_patient = VALID_PATIENT.copy()
    invalid_patient["age"] = 500

    response = client.post("/predict", json=invalid_patient)
    assert response.status_code == 422
