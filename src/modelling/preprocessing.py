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


@task(name="encoding")
def encode_sex_column(df) -> pd.DataFrame:

    # 1. Apply One-Hot Encoding
    df_encoded = pd.get_dummies(df["Sex"], prefix="Sex", drop_first=True, dtype=int)

    # 2. Concatenate the new encoded columns back to the original DataFrame
    df = pd.concat([df.drop("Sex", axis=1), df_encoded], axis=1)
    return df


@task(name="preprocessing")
def preprocessing(df):
    df = encode_sex_column(df)
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
    if categorical_cols is None:
        # Use the one-hot encoded Sex columns instead of the original Sex column
        categorical_cols = [col for col in df.columns if col.startswith("Sex_")]

    # If no categorical columns found, use all columns as features
    if not categorical_cols:
        categorical_cols = df.columns.tolist()

    dicts = df[categorical_cols].to_dict(orient="records")

    y = None
    if dv is None:
        dv = DictVectorizer()
        dv.fit(dicts)

    if with_target:
        y = df["Rings"].values

    x = dv.transform(dicts)
    return x, y, dv
