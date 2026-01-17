from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_passenger


app = FastAPI(title="Titanic Survival API")


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
    return predict_passenger(passenger.model_dump())