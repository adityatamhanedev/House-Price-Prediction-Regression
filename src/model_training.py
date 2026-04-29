# model_training.py

import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def split_data(X, y):
    """
    Split dataset into train and test sets
    """

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


def train_linear_regression(X_train, y_train):
    """
    Train Linear Regression model
    """

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    return model


def train_random_forest(X_train, y_train):
    """
    Train Random Forest model
    """

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance
    """

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return mae, rmse, r2


def save_model(model, filename):
    """
    Save trained model
    """

    pickle.dump(
        model,
        open(filename, "wb")
    )