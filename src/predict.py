import joblib
import pandas as pd


MODEL_PATH = 'models/model.pkl'
model = joblib.load(MODEL_PATH)




def predict_passenger(data: dict):
    df = pd.DataFrame([data])
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})


    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]


    return {
    'survived': int(prediction),
    'probability': round(float(probability), 4)
    }