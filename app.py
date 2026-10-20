from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter


import logging
import time
from datetime import datetime, timezone

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger("heart_disease_api")

predictions_total = Counter(
    "predictions_total",
    "Total successful predictions by predicted label",
    ["label"],
)


class PredictionLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Log only prediction requests, not health probes or Swagger requests.
        if request.url.path != "/predict":
            return await call_next(request)

        start_time = time.perf_counter()
        status_code = 500
        predicted_label = "unavailable"

        try:
            response = await call_next(request)
            status_code = response.status_code

            # The endpoint can set this header with the predicted label.
            predicted_label = response.headers.get(
                "X-Predicted-Label", "not_available"
            )
            return response

        except Exception:
            logger.exception("Prediction request failed")
            raise

        finally:
            latency_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                "event=predict "
                "timestamp=%s status_code=%s latency_ms=%.2f "
                "predicted_label=%s",
                datetime.now(timezone.utc).isoformat(),
                status_code,
                latency_ms,
                predicted_label,
            )


MODEL_PATH = Path(__file__).parent / "models" / "heart_disease_pipeline.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Heart Disease Classification API",
    description="Educational ML prediction API",
    version="1.0.0"
)
Instrumentator().instrument(app).expose(app)

app.add_middleware(PredictionLoggingMiddleware)


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
    patient_data = pd.DataFrame([patient.model_dump()])

    prediction = int(model.predict(patient_data)[0])
    probability = float(model.predict_proba(patient_data)[0, 1])
    predictions_total.labels(label=str(prediction)).inc()

    result = {
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

    return JSONResponse(
        content=result,
        headers={"X-Predicted-Label": str(prediction)},
    )

