import joblib
import pandas as pd
import yaml


class TitanicPredictor:
    def __init__(self, config_path: str = 'config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model = joblib.load(self.config['paths']['model_path'])

    def predict_passenger(self, data: dict):
        df = pd.DataFrame([data])
        
        # Aplicar mapeos desde la configuración
        mappings = self.config['data_features']['mappings']
        for col, mapping in mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)

        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0][1]
    
        return {
            'survived': int(prediction),
            'the probability was': round(float(probability), 4)
        }

    