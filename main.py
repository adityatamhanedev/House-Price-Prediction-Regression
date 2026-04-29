# main.py

from src.data_preprocessing import (
    load_data,
    clean_data,
    encode_features,
    split_features_target
)

from src.model_training import (
    split_data,
    train_linear_regression,
    train_random_forest,
    evaluate_model,
    save_model
)

from src.prediction import (
    load_model,
    predict_price
)

# Load data
df = load_data("data/housing_data.csv")

# Clean data
df = clean_data(df)

# Encode data
df = encode_features(df)

# Split features
X, y = split_features_target(df)

# Train-test split
X_train, X_test, y_train, y_test = split_data(X, y)

# Train models
lr_model = train_linear_regression(X_train, y_train)

rf_model = train_random_forest(X_train, y_train)

# Predictions
lr_pred = lr_model.predict(X_test)

rf_pred = rf_model.predict(X_test)

# Evaluate
lr_mae, lr_rmse, lr_r2 = evaluate_model(
    y_test,
    lr_pred
)

rf_mae, rf_rmse, rf_r2 = evaluate_model(
    y_test,
    rf_pred
)

print("Linear Regression R2:", lr_r2)

print("Random Forest R2:", rf_r2)

# Save best model
save_model(
    rf_model,
    "models/random_forest.pkl"
)