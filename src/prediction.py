# prediction.py

import pickle
import pandas as pd


def load_model(model_path):
    """
    Load saved model
    """

    model = pickle.load(
        open(model_path, "rb")
    )

    return model


def predict_price(model, input_data):
    """
    Predict house price
    """

    df = pd.DataFrame(input_data)

    prediction = model.predict(df)

    return prediction[0]