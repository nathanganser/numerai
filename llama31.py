import logging
import json
import pandas as pd
import numpy as np
from numerapi import NumerAPI
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import mutual_info_regression, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
import optuna
import cloudpickle
import requests

logging.basicConfig(level=logging.INFO)
model_name = "llama31_improved"

featureset = "small"
n_trials=2 #100
n_estimators=2 #20000

def objective(trial, X, y):
    rf_params = {
        'n_estimators': trial.suggest_int('rf_n_estimators', n_estimators, n_estimators*2),
        'max_depth': trial.suggest_int('rf_max_depth', 3, 30),
        'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10)
    }
    gb_params = {
        'n_estimators': trial.suggest_int('gb_n_estimators', n_estimators, n_estimators*2),
        'max_depth': trial.suggest_int('gb_max_depth', 3, 30),
        'learning_rate': trial.suggest_loguniform('gb_learning_rate', 1e-3, 1.0),
        'subsample': trial.suggest_uniform('gb_subsample', 0.5, 1.0)
    }

    rf = MultiOutputRegressor(RandomForestRegressor(**rf_params, random_state=42))
    gb = MultiOutputRegressor(GradientBoostingRegressor(**gb_params, random_state=42))

    rf.fit(X, y)
    gb.fit(X, y)

    rf_pred = rf.predict(X)
    gb_pred = gb.predict(X)

    stack_input = np.column_stack((rf_pred, gb_pred))
    stack = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    stack.fit(stack_input, y)

    final_pred = stack.predict(stack_input)
    mse = mean_squared_error(y, final_pred)
    return mse


def compute_me():
    napi = NumerAPI()
    DATA_VERSION = "v4.3"

    logging.info("Downloading dataset...")
    # Uncomment these lines if you need to download the datasets
    # napi.download_dataset(f"{DATA_VERSION}/validation_int8.parquet")
    # napi.download_dataset(f"{DATA_VERSION}/live_int8.parquet")
    # napi.download_dataset(f"{DATA_VERSION}/features.json")
    feature_metadata = json.load(open(f"{DATA_VERSION}/features.json"))
    feature_set = feature_metadata["feature_sets"][featureset]
    targets = feature_metadata["targets"]

    # Load and preprocess data
    logging.info("Loading and preprocessing data...")
    data = pd.read_parquet(f"{DATA_VERSION}/validation_int8.parquet", columns=["era"] + targets + feature_set)

    # Use more data for improved model performance
    data = data[data["era"].isin(data["era"].unique()[::72])]

    # Handle missing values
    data = data.dropna()

    # Feature selection and preprocessing pipeline
    logging.info("Feature selection and preprocessing...")
    feature_pipeline = Pipeline([
        ('variance_threshold', VarianceThreshold(threshold=0.01)),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95, whiten=True))  # Keep 95% of variance
    ])

    X = data[feature_set]
    y = data[targets]

    X_processed = feature_pipeline.fit_transform(X)

    # Split data into training and validation sets
    logging.info("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    # Hyperparameter optimization
    logging.info("Performing hyperparameter optimization...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)

    best_params = study.best_params
    logging.info(f"Best parameters: {best_params}")

    # Train final model with best parameters
    logging.info("Training final model...")
    rf_final = MultiOutputRegressor(RandomForestRegressor(
        n_estimators=best_params['rf_n_estimators'],
        max_depth=best_params['rf_max_depth'],
        min_samples_split=best_params['rf_min_samples_split'],
        min_samples_leaf=best_params['rf_min_samples_leaf'],
        random_state=42
    ))
    gb_final = MultiOutputRegressor(GradientBoostingRegressor(
        n_estimators=best_params['gb_n_estimators'],
        max_depth=best_params['gb_max_depth'],
        learning_rate=best_params['gb_learning_rate'],
        subsample=best_params['gb_subsample'],
        random_state=42
    ))

    rf_final.fit(X_train, y_train)
    gb_final.fit(X_train, y_train)

    rf_pred_train = rf_final.predict(X_train)
    gb_pred_train = gb_final.predict(X_train)
    stack_input_train = np.column_stack((rf_pred_train, gb_pred_train))

    stack_final = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    stack_final.fit(stack_input_train, y_train)

    # Model evaluation
    logging.info("Evaluating model...")
    rf_pred_val = rf_final.predict(X_val)
    gb_pred_val = gb_final.predict(X_val)
    stack_input_val = np.column_stack((rf_pred_val, gb_pred_val))
    final_pred_val = stack_final.predict(stack_input_val)

    mse = mean_squared_error(y_val, final_pred_val)
    logging.info(f"Mean Squared Error: {mse:.4f}")

    # Define prediction function
    def predict(live_features: pd.DataFrame) -> pd.DataFrame:
        live_features_processed = feature_pipeline.transform(live_features[feature_set])
        rf_pred = rf_final.predict(live_features_processed)
        gb_pred = gb_final.predict(live_features_processed)
        stack_input = np.column_stack((rf_pred, gb_pred))
        final_pred = stack_final.predict(stack_input)
        return pd.DataFrame(final_pred, index=live_features.index, columns=targets)

    logging.info("Serializing and uploading prediction function...")
    p = cloudpickle.dumps(predict)
    with open(f"{model_name}.pkl", "wb") as f:
        f.write(p)

    url = "https://api.bytescale.com/v2/accounts/12a1yew/uploads/form_data"
    headers = {"Authorization": "Bearer public_12a1yewAHfRPdqAXnHXQDib1RwoJ"}
    files = {"file": open(f"{model_name}.pkl", "rb")}
    response = requests.post(url, headers=headers, files=files)
    if response.status_code == 200:
        logging.info("File uploaded successfully.")
    else:
        logging.error(f"File upload failed with status code: {response.status_code}")


if __name__ == "__main__":
    compute_me()