import joblib
import pandas as pd
import yaml
from typing import List, Dict

class TitanicPredictor:
    def __init__(self, config_path: str = 'config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model = joblib.load(self.config['paths']['model_path'])

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper to apply mappings to any DataFrame."""
        mappings = self.config['data_features']['mappings']
        for col, mapping in mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
        return df

    def predict_passenger(self, data: dict):
        """Single prediction logic (Online)."""
        df = pd.DataFrame([data])
        df = self._prepare_data(df)
        
        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0][1]
    
        return {
            'survived': int(prediction),
            'probability': round(float(probability), 4)
        }

    def predict_batch(self, data_list: List[Dict]):
        """Batch prediction logic (Lotes)."""
        df = pd.DataFrame(data_list)
        df = self._prepare_data(df)
        
        predictions = self.model.predict(df)
        probabilities = self.model.predict_proba(df)[:, 1]
        
        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append({
                'survived': int(pred),
                'probability': round(float(prob), 4)
            })
        return results
    