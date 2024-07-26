import json
import pandas as pd

from numerapi import NumerAPI
import cloudpickle
import requests
import logging
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor

logging.basicConfig(level=logging.INFO)
model_name = "claude_v1"

def compute_me():
    napi = NumerAPI()
    DATA_VERSION = "v4.3"
    featureset = "all"
    logging.info("Downloading dataset...")
    napi.download_dataset(f"{DATA_VERSION}/validation_int8.parquet")
    napi.download_dataset(f"{DATA_VERSION}/live_int8.parquet")
    napi.download_dataset(f"{DATA_VERSION}/features.json")
    feature_metadata = json.load(open(f"{DATA_VERSION}/features.json"))
    feature_set = feature_metadata["feature_sets"][featureset]

    # Load and preprocess data
    logging.info("Loading and preprocessing data...")
    val_data = pd.read_parquet(f"{DATA_VERSION}/validation_int8.parquet", columns=["era", "target"] + feature_set)

    # Count the number of rows with NaN values
    nan_rows = val_data.isnull().any(axis=1).sum()
    print(f"Number of rows with NaN values: {nan_rows}")

    # Delete rows with NaN values
    val_data = val_data.dropna()

    # Print the updated number of rows
    print(f"Number of rows after deleting NaN values: {len(val_data)}")

    features = feature_set
    target = "target"

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 93,
        'max_depth': 9,
        'learning_rate': 0.02765,
        'feature_fraction': 0.951296710128533,
        'bagging_fraction': 0.6565998368130392,
        'bagging_freq': 6,
        'min_child_samples': 24
    }

    # Train LightGBM model with best hyperparameters
    logging.info("Training LightGBM model with best hyperparameters...")
    lgb_model = lgb.LGBMRegressor(**params)
    lgb_model.fit(val_data[features], val_data[target])

    # Train CatBoost model
    logging.info("Training CatBoost model...")
    cat_params = {
        'loss_function': 'RMSE',
        'learning_rate': 0.05,
        'depth': 10,
        'subsample': 0.8,
        'colsample_bylevel': 0.8,
        'random_seed': 42
    }
    cat_model = CatBoostRegressor(**cat_params, verbose=0)
    cat_model.fit(val_data[features], val_data[target])

    # Ensemble models using stacking
    logging.info("Ensembling models using stacking...")
    estimators = [('lgb', lgb_model), ('cat', cat_model)]
    ensemble_model = StackingRegressor(estimators=estimators, final_estimator=lgb.LGBMRegressor())
    ensemble_model.fit(val_data[features], val_data[target])

    # Define prediction function
    def predict(live_features: pd.DataFrame) -> pd.DataFrame:
        live_predictions = ensemble_model.predict(live_features[features])
        submission = pd.Series(live_predictions, index=live_features.index)
        return submission.to_frame("prediction")

    # Test prediction function
    logging.info("Testing prediction function...")
    live_data = pd.read_parquet(f"{DATA_VERSION}/live_int8.parquet", columns=["era"] + feature_set)
    predictions = predict(live_data)
    logging.info(f"Predictions shape: {predictions.shape}")

    # Serialize and save prediction function
    logging.info("Serializing and saving prediction function...")
    with open(f"{model_name}.pkl", "wb") as f:
        cloudpickle.dump(predict, f)

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