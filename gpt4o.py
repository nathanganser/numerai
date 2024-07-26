
import json

import pandas as pd
from numerapi import NumerAPI
import cloudpickle
import requests
import logging
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.INFO)
model_name = "gpt4o"

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

    # Handle missing values
    val_data = val_data.dropna()

    #val_data = val_data[val_data["era"].isin(val_data["era"].unique()[::36])]

    features = feature_set
    target = "target"
    rolling_array = [5, 10, 15]

    # Feature Engineering: Adding rolling mean and volatility features
    logging.info("Adding rolling mean and volatility features...")
    for window in rolling_array:
        val_data[f'rolling_mean_{window}'] = val_data[features].rolling(window=window, axis=1).mean().mean(axis=1)
        val_data[f'rolling_volatility_{window}'] = val_data[features].rolling(window=window, axis=1).std().mean(axis=1)

    # Selecting the most important features using a simple LightGBM model
    logging.info("Selecting the most important features...")
    lgb_temp = lgb.LGBMRegressor()
    lgb_temp.fit(val_data[features], val_data[target])
    feature_importances = pd.Series(lgb_temp.feature_importances_, index=features).sort_values(ascending=False)
    selected_features = feature_importances.head(100).index.tolist()  # Select top 100 features

    # Split data for validation
    logging.info("Splitting data for validation...")
    train_data, test_data = train_test_split(val_data, test_size=0.2, random_state=42)

    # Define parameter grids for LightGBM and CatBoost
    lgb_param_grid = {
        'num_leaves': [31, 50],  # Removed 93, as 31 and 50 are common values
        'max_depth': [6, 9],  # Removed 12, to reduce complexity
        'learning_rate': [0.01, 0.05],  # Kept the most distinct values
        'feature_fraction': [0.8, 0.9],  # Removed 0.951296710128533, to keep it simple
        'bagging_fraction': [0.5, 0.6],  # Removed 0.6565998368130392
        'bagging_freq': [5, 6],  # Removed 7, to reduce complexity
        'min_child_samples': [20, 30]  # Removed 24, keeping distinct low and high values
    }

    cat_param_grid = {
        'learning_rate': [0.03, 0.1],  # Kept the most distinct values
        'depth': [6, 10],  # Removed 12, to reduce complexity
        'subsample': [0.7, 0.8],  # Removed 0.9, to reduce complexity
        'colsample_bylevel': [0.7, 0.9]  # Removed 0.8, keeping distinct low and high values
    }

    # Hyperparameter tuning using GridSearchCV
    logging.info("Hyperparameter tuning for LightGBM...")
    lgb_model = lgb.LGBMRegressor()
    lgb_grid_search = GridSearchCV(estimator=lgb_model, param_grid=lgb_param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
    lgb_grid_search.fit(train_data[selected_features], train_data[target])
    best_lgb_params = lgb_grid_search.best_params_
    logging.info(f"Best LightGBM params: {best_lgb_params}")

    logging.info("Hyperparameter tuning for CatBoost...")
    cat_model = CatBoostRegressor(verbose=0)
    cat_grid_search = GridSearchCV(estimator=cat_model, param_grid=cat_param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
    cat_grid_search.fit(train_data[selected_features], train_data[target])
    best_cat_params = cat_grid_search.best_params_
    logging.info(f"Best CatBoost params: {best_cat_params}")

    # Train models with best hyperparameters
    logging.info("Training LightGBM model with best hyperparameters...")
    lgb_model = lgb.LGBMRegressor(**best_lgb_params)
    lgb_model.fit(train_data[selected_features], train_data[target])

    logging.info("Training CatBoost model with best hyperparameters...")
    cat_model = CatBoostRegressor(**best_cat_params, verbose=0)
    cat_model.fit(train_data[selected_features], train_data[target])

    # Ensemble models using stacking with cross-validation
    logging.info("Ensembling models using stacking with cross-validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    stacking_model = StackingRegressor(
        estimators=[('lgb', lgb_model), ('cat', cat_model)],
        final_estimator=lgb.LGBMRegressor()
    )

    for train_index, val_index in kf.split(train_data):
        X_train, X_val = train_data.iloc[train_index][selected_features], train_data.iloc[val_index][selected_features]
        y_train, y_val = train_data.iloc[train_index][target], train_data.iloc[val_index][target]

        stacking_model.fit(X_train, y_train)
        val_preds = stacking_model.predict(X_val)
        rmse = mean_squared_error(y_val, val_preds, squared=False)
        logging.info(f"Validation RMSE: {rmse}")

    # Define prediction function
    def predict(live_features: pd.DataFrame) -> pd.DataFrame:
        live_predictions = stacking_model.predict(live_features[selected_features])
        submission = pd.Series(live_predictions, index=live_features.index)
        return submission.to_frame("prediction")

    # Test prediction function
    logging.info("Testing prediction function...")
    live_data = pd.read_parquet(f"{DATA_VERSION}/live_int8.parquet", columns=["era"] + feature_set)

    # Adding same rolling features to live data
    for window in rolling_array:
        live_data[f'rolling_mean_{window}'] = live_data[features].rolling(window=window, axis=1).mean().mean(axis=1)
        live_data[f'rolling_volatility_{window}'] = live_data[features].rolling(window=window, axis=1).std().mean(axis=1)

    predictions = predict(live_data)
    logging.info(f"Predictions shape: {predictions.shape}")

    # Serialize and upload prediction function
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

compute_me()
