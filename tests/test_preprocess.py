import pandas as pd
from src.preprocess import preprocess


def test_preprocess_outputs_correct_shape_and_values():
    # Arrange: small fake dataset
    df = pd.DataFrame({
        "pclass": [1, 3],
        "sex": ["male", "female"],
        "age": [22, None],
        "sibsp": [1, 0],
        "parch": [0, 0],
        "fare": [7.25, 71.83],
        "survived": [0, 1],
    })

    # Act
    X, y = preprocess(df)

    # Assert: correct columns
    assert list(X.columns) == [
        "pclass", "sex", "age", "sibsp", "parch", "fare"
    ]

    # Assert: no missing ages
    assert X["age"].isnull().sum() == 0

    # Assert: Sex encoded correctly
    assert set(X["sex"].unique()).issubset({0, 1})

    # Assert: target correct
    assert y.tolist() == [0, 1]
