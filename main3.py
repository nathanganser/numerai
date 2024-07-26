import json
import numpy as np
import pandas as pd
from numerapi import NumerAPI
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss

# Set the data version to one of the most recent versions
DATA_VERSION = "v4.3"

# Download data
napi = NumerAPI()
napi.download_dataset(f"{DATA_VERSION}/train_int8.parquet")
napi.download_dataset(f"{DATA_VERSION}/validation_int8.parquet")
napi.download_dataset(f"{DATA_VERSION}/features.json")
napi.download_dataset(f"{DATA_VERSION}/live_int8.parquet")

# Load data
feature_metadata = json.load(open(f"{DATA_VERSION}/features.json"))
feature_cols = feature_metadata["feature_sets"]["medium"]
target_cols = feature_metadata["targets"]

train = pd.read_parquet(
    f"{DATA_VERSION}/train_int8.parquet",
    columns=["era"] + feature_cols + target_cols
)
validation = pd.read_parquet(
    f"{DATA_VERSION}/validation_int8.parquet",
    columns=["era"] + feature_cols + target_cols
)

# Filter validation data
validation = validation[validation["data_type"] == "validation"]
del validation["data_type"]

# Fill NaNs in targets with 0.5 (mean target value)
for c in target_cols:
    train[c] = train[c].fillna(0.5)
    validation[c] = validation[c].fillna(0.5)

# Ensemble target
train["ensemble_target"] = train[target_cols].mean(axis=1)
validation["ensemble_target"] = validation[target_cols].mean(axis=1)

# Group K-fold for cross validation
gkf = GroupKFold(n_splits=5)

# LightGBM hyper-parameters optimized for the competition
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'max_depth': -1,
    'num_leaves': 31,
    'min_data_in_leaf': 500,
    'learning_rate': 0.035,
    'subsample': 0.80,
    'subsample_freq': 1,
    'feature_fraction': 0.60,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'seed': 42,
}

# Train and cross-validate model
oof_predictions = np.zeros(len(train))
validation_predictions = np.zeros(len(validation))

for fold, (trn_idx, val_idx) in enumerate(gkf.split(X=train, groups=train['era'])):
    print(f"===== FOLD {fold} =====")

    X_train, y_train = train.iloc[trn_idx][feature_cols], train.iloc[trn_idx]["ensemble_target"]
    X_valid, y_valid = train.iloc[val_idx][feature_cols], train.iloc[val_idx]["ensemble_target"]

    dtrain = lgb.Dataset(X_train, y_train, weight=1 / np.square(0.25))
    dvalid = lgb.Dataset(X_valid, y_valid, weight=1 / np.square(0.25), reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dtrain, dvalid],
        early_stopping_rounds=50,
        verbose_eval=100
    )

    oof_predictions[val_idx] = model.predict(X_valid)
    validation_predictions += model.predict(validation[feature_cols]) / gkf.n_splits

    print(f"FOLD {fold} logloss: {log_loss(y_valid, oof_predictions[val_idx])}")

print(f"----- OVERALL OOF logloss: {log_loss(train['ensemble_target'], oof_predictions)} -----")


def predict(live_features: pd.DataFrame) -> pd.DataFrame:
    predictions = validation_predictions

    # Scale to [0, 1] per era
    predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())

    # Rank predictions per era
    predictions = predictions.groupby(validation['era']).rank(pct=True)

    return predictions.to_frame("prediction")


# Quick test with live features
live_features = pd.read_parquet(f"{DATA_VERSION}/live_int8.parquet", columns=feature_cols)
predict(live_features)

# Use the cloudpickle library to serialize function and dependencies
import cloudpickle

model_pkl = cloudpickle.dumps(predict)
with open("predict.pkl", "wb") as f:
    f.write(model_pkl)