import pandas as pd


def preprocess(df: pd.DataFrame):
    df = df.copy()


    df['age'] = df['age'].fillna(df['age'].median())
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})


    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']
    X = df[features]
    y = df['survived']


    return X, y