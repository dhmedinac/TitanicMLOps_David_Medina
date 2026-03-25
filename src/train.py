import yaml
import pandas as pd
import seaborn as sns # Para mantener tu lógica de carga actual
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from src.preprocess import TitanicPreprocessor


DATA_PATH = 'data/titanic.csv'
MODEL_PATH = 'models/model.pkl'


class TitanicTrainer:
    def __init__(self, config_path='config/config.yaml'):
        # Cargar configuración
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Inicializar modelo con hiperparámetros de la config
        self.model = RandomForestClassifier(
            **self.config['model_hyperparameters']
        )

    def train(self):
        # En producción podrías usar self.config['paths']['data_path']
        df = sns.load_dataset("titanic") 

        # Pasar parámetros de la config al preprocesador
        preprocessor = TitanicPreprocessor(
            features=self.config['data_features']['selected_columns'],
            mappings=self.config['data_features']['mappings'],
            target=self.config['data_features']['target']
        )
        X, y = preprocessor.preprocess(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['train_params']['test_size'],
            random_state=self.config['train_params']['random_state']
        )

        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")

    def save(self):
        path = self.config['paths']['model_path']
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

if __name__ == '__main__':
    trainer = TitanicTrainer()
    trainer.train()
    trainer.save()