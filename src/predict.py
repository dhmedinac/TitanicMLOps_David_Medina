import joblib
import pandas as pd


MODEL_PATH = 'models/model.pkl'



class TitanicPredictor:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model = joblib.load(model_path)

    def predict_passenger(self, data: dict):
        df = pd.DataFrame([data])
        df['sex'] = df['sex'].map({'male': 0, 'female': 1})

        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0][1]
    
        return {
        'survived': int(prediction),
        'probability': round(float(probability), 4)
        }

    