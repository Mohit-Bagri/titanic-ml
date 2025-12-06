from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import json

from fastapi.responses import FileResponse

app = FastAPI(title="Titanic Survival Prediction API")

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
def home():
    return {"message": "Titanic Survival API is running"}


@app.get("/ui")
def ui():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    UI_PATH = os.path.join(PROJECT_ROOT, "docs", "index.html")
    return FileResponse(UI_PATH)


@app.post("/predict")
def predict(input_data: TitanicInput):
    # ✅ CRITICAL FIX: NumPy → DataFrame with column names
    df = pd.DataFrame([input_data.data], columns=FEATURES)

    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0, 1]

    return {
        "survived": bool(pred),
        "probability_survived": round(float(proba), 4),
    }
