# data_preprocessing.py

import pandas as pd


def load_data(filepath):
    """
    Load dataset from CSV file
    """

    df = pd.read_csv(filepath)

    return df


def clean_data(df):
    """
    Remove duplicates and handle missing values
    """

    df = df.drop_duplicates()

    df = df.fillna(method='ffill')

    return df


def encode_features(df):
    """
    Convert categorical features into numeric
    """

    df_encoded = pd.get_dummies(
        df,
        columns=['location', 'furnishing'],
        drop_first=True
    )

    return df_encoded


def split_features_target(df):
    """
    Split dataset into X and y
    """

    X = df.drop('price', axis=1)

    y = df['price']

    return X, y