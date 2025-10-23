import pandas as pd


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
