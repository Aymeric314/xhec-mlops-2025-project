import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(df):
    y = df["Rings"]
    X = df.drop(columns=["Rings"])

    # 3. Perform Train-Test Split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=41
    )
    return X_train, X_test, y_train, y_test


def encode_sex_column(df) -> pd.DataFrame:

    # 1. Apply One-Hot Encoding
    df_encoded = pd.get_dummies(df["Sex"], prefix="Sex", drop_first=True, dtype=int)

    # 2. Concatenate the new encoded columns back to the original DataFrame
    df = pd.concat([df.drop("Sex", axis=1), df_encoded], axis=1)
    return df


def preprocessing(df):
    df = encode_sex_column(df)
    df.drop(columns=["Length"], inplace=True)
    return df
