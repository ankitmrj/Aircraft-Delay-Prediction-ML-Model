from fastapi import FastAPI
import pandas as pd

from app.schema import FlightInput
from app.model import model

app = FastAPI(title="Aircraft Delay Prediction API")

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict_delay(data: FlightInput):
    input_df = pd.DataFrame([{
        "weather": data.weather,
        "dep_hour": data.dep_hour,
        "route": data.route,
        "aircraft_type": data.aircraft_type
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "delayed": bool(prediction),
        "delay_probability": round(probability, 2)
    }
