import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import seaborn as sns
from preprocess import TitanicPreprocessor


DATA_PATH = 'data/titanic.csv'
MODEL_PATH = 'models/model.pkl'


class TitanicTrainer:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self):
        # df = pd.read_csv(DATA_PATH) Original line
        df = sns.load_dataset("titanic")  # Updated line to load dataset using seaborn

        preprocessor = TitanicPreprocessor()
        X, y = preprocessor.preprocess(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy: {acc:.4f}")

    def save(self):
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")


if __name__ == '__main__':
    trainer = TitanicTrainer()
    trainer.train()
    trainer.save()