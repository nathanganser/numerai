import pandas as pd
from sklearn.ensemble import VotingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import RFECV
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from numerapi import NumerAPI
import numpy as np
import json
import cloudpickle
import requests
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf

# Initialize NumerAPI
napi = NumerAPI()

# Set data version and feature set
DATA_VERSION = "v4.3"
featureset = "medium"
napi.download_dataset(f"{DATA_VERSION}/train_int8.parquet")
napi.download_dataset(f"{DATA_VERSION}/validation_int8.parquet")
napi.download_dataset(f"{DATA_VERSION}/features.json")
feature_metadata = json.load(open(f"{DATA_VERSION}/features.json"))
feature_set = feature_metadata["feature_sets"][featureset]

# Load datasets
train_data = pd.read_parquet(f"{DATA_VERSION}/train_int8.parquet", columns=["era", "target"] + feature_set)
validation_data = pd.read_parquet(f"{DATA_VERSION}/validation_int8.parquet", columns=["era", "target"] + feature_set)

#validation_data = validation_data[(validation_data["era"].astype(int) > 200) & (validation_data["era"].isin(validation_data["era"].unique()[::3]))]

#train_data = train_data[(train_data["era"].astype(int) > 200) & (train_data["era"].isin(train_data["era"].unique()[::3]))]

# Feature Selection using Recursive Feature Elimination
selector = RFECV(estimator=RandomForestRegressor(n_estimators=100, random_state=42), step=1, cv=5, scoring='neg_mean_squared_error')
selector.fit(train_data[feature_set], train_data["target"])
selected_features = list(train_data[feature_set].columns[selector.support_])

# Era-based Cross-validation with additional metrics
def era_based_cv(model, train_data, validation_data, features):
    eras = train_data["era"].unique()
    cv_scores = []
    for era in eras:
        train_fold = train_data[train_data["era"] != era]
        val_fold = train_data[train_data["era"] == era]
        model.fit(train_fold[features], train_fold["target"])
        predictions = model.predict(val_fold[features])
        mse = mean_squared_error(val_fold["target"], predictions)
        mae = mean_absolute_error(val_fold["target"], predictions)
        r2 = r2_score(val_fold["target"], predictions)
        corr = np.corrcoef(val_fold["target"], predictions)[0, 1]
        cv_scores.append({"era": era, "mse": mse, "mae": mae, "r2": r2, "corr": corr})
        print(f"Era {era} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, Corr: {corr:.4f}")
    return pd.DataFrame(cv_scores)

# Model Training with hyperparameter tuning
lgb_model = lgb.LGBMRegressor(
    n_estimators=5000,
    learning_rate=0.005,
    max_depth=7,
    num_leaves=2**7-1,
    colsample_bytree=0.8,
    subsample=0.9,
    subsample_freq=1,
    random_state=42
)

hist_gb_model = HistGradientBoostingRegressor(
    max_iter=500,
    max_depth=15,
    learning_rate=0.05,
    l2_regularization=1e-4,
    random_state=42
)

catboost_model = CatBoostRegressor(
    iterations=2000,
    depth=8,
    learning_rate=0.03,
    l2_leaf_reg=3,
    random_seed=42,
    loss_function='RMSE',
    verbose=0
)

lgb_scores = era_based_cv(lgb_model, train_data, validation_data, selected_features)
hist_gb_scores = era_based_cv(hist_gb_model, train_data, validation_data, selected_features)
catboost_scores = era_based_cv(catboost_model, train_data, validation_data, selected_features)

# Ensemble with optimized weights based on cross-validation performance
ensemble_model = VotingRegressor(
    estimators=[("lgb", lgb_model), ("hist_gb", hist_gb_model), ("catboost", catboost_model)],
    weights=[
        lgb_scores["corr"].mean(),
        hist_gb_scores["corr"].mean(),
        catboost_scores["corr"].mean()
    ]
)
ensemble_model.fit(train_data[selected_features], train_data["target"])


# Define your prediction pipeline as a function
def predict(live_features: pd.DataFrame) -> pd.DataFrame:
    live_predictions = ensemble_model.predict(live_features[selected_features])
    submission = pd.Series(live_predictions, index=live_features.index)
    return submission.to_frame("prediction")

# Use the cloudpickle library to serialize your function
p = cloudpickle.dumps(predict)
with open("predict.pkl", "wb") as f:
    f.write(p)

# Upload the predict.pkl file to Bytescale
url = "https://api.bytescale.com/v2/accounts/12a1yew/uploads/form_data"
headers = {"Authorization": "Bearer public_12a1yewAHfRPdqAXnHXQDib1RwoJ"}
files = {"file": open("predict.pkl", "rb")}
response = requests.post(url, headers=headers, files=files)
if response.status_code == 200:
    print("File uploaded successfully.")
else:
    print("File upload failed with status code:", response.status_code)
