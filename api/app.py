from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import json

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Titanic Survival Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "titanic_best_pipeline.joblib")
META_PATH = os.path.join(BASE_DIR, "model", "titanic_metadata.json")

model = joblib.load(MODEL_PATH)

with open(META_PATH) as f:
    metadata = json.load(f)

FEATURES = metadata["features"]


class TitanicInput(BaseModel):
    data: list


@app.get("/")
def health():
    return {"status": "ok", "model": "titanic-logistic-regression"}


@app.post("/predict")
def predict(input_data: TitanicInput):
    df = pd.DataFrame([input_data.data], columns=FEATURES)
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0, 1]

    return {
        "survived": bool(pred),
        "probability_survived": round(float(proba), 4),
    }
