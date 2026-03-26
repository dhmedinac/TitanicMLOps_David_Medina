from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import TitanicPredictor
from typing import List

app = FastAPI(title="Titanic Survival API")
predictor = TitanicPredictor()


class Passenger(BaseModel):
    pclass: int
    sex: str
    age: float
    sibsp: int
    parch: int
    fare: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(passenger: Passenger):
    return predictor.predict_passenger(passenger.model_dump())

@app.post("/predict/batch")
def predict_batch(passengers: List[Passenger]):
    """Endpoint for batch predictions."""
    # Convert list of Pydantic models to list of dicts
    data = [p.model_dump() for p in passengers]
    return predictor.predict_batch(data)