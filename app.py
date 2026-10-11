from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "models" / "heart_disease_pipeline.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Heart Disease Classification API",
    description="Educational ML prediction API",
    version="1.0.0"
)


class PatientInput(BaseModel):
    age: float = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)
    trestbps: float = Field(gt=0)
    chol: float = Field(gt=0)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(gt=0)
    exang: int = Field(ge=0, le=1)
    oldpeak: float
    slope: int = Field(ge=1, le=3)
    ca: int = Field(ge=0, le=3)
    thal: int = Field(ge=3, le=7)


@app.get("/")
def home():
    return {"message": "Heart Disease Classification API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict(patient: PatientInput):
    patient_data = pd.DataFrame(
        [patient.model_dump()]
    )

    prediction = int(model.predict(patient_data)[0])
    probability = float(
        model.predict_proba(patient_data)[0, 1]
    )

    return {
        "prediction": prediction,
        "prediction_label": (
            "Disease present" if prediction == 1
            else "No disease"
        ),
        "probability_class_1": round(probability, 4),
        "note": (
            "Educational model output only; "
            "not a medical diagnosis."
        )
    }
