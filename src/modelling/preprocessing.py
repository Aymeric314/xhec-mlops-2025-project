from typing import List

import pandas as pd
from prefect import task
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split


@task(name="split_data")
def split_data(df):
    y = df["Rings"]
    X = df.drop(columns=["Rings"])

    # 3. Perform Train-Test Split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=41
    )
    return X_train, X_test, y_train, y_test


@task(name="preprocessing")
def preprocessing(df):
    # Simply drop Length column, let DictVectorizer handle categorical encoding
    df.drop(columns=["Length"], inplace=True)
    return df


@task(name="extract_x_y")
def extract_x_y(
    df: pd.DataFrame,
    categorical_cols: List[str] = None,
    dv: DictVectorizer = None,
    with_target: bool = True,
) -> tuple:
    """Extract features and target for abalone dataset."""
    # Use ALL columns as features (both numerical and categorical)
    # This matches the notebook approach but uses DictVectorizer
    feature_cols = df.columns.tolist()

    # Remove target column if present
    if "Rings" in feature_cols:
        feature_cols.remove("Rings")

    # Convert ALL features to dictionary format for DictVectorizer
    dicts = df[feature_cols].to_dict(orient="records")

    y = None
    if dv is None:
        dv = DictVectorizer()
        dv.fit(dicts)

    if with_target:
        y = df["Rings"].values

    x = dv.transform(dicts)
    return x, y, dv
