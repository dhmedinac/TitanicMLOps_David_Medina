import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import seaborn as sns
from preprocess import preprocess


DATA_PATH = 'data/titanic.csv'
MODEL_PATH = 'models/model.pkl'


if __name__ == '__main__':
#df = pd.read_csv(DATA_PATH) Original line
    df = sns.load_dataset("titanic")  # Updated line to load dataset using seaborn

    X, y = preprocess(df)


    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
    )


    model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
    )


    model.fit(X_train, y_train)


    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)


    print(f"Accuracy: {acc:.4f}")


    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")