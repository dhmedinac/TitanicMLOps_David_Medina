import pandas as pd

class TitanicPreprocessor:
    def __init__(self, features, mappings, target):
        self.features = features
        self.mappings = mappings
        self.target = target

    def preprocess(self, df: pd.DataFrame):
        df = df.copy()

        # Manejo de nulos
        df['age'] = df['age'].fillna(df['age'].median())
        
        # Uso de mapeos dinámicos desde la configuración
        for col, mapping in self.mappings.items():
            df[col] = df[col].map(mapping)

        X = df[self.features]
        y = df[self.target]

        return X, y       