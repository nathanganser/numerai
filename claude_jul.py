import logging
import json
import pandas as pd
import numpy as np
from numerapi import NumerAPI
import cloudpickle
import requests
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor
import optuna

logging.basicConfig(level=logging.INFO)
model_name = "llama31"


def compute_me():
    napi = NumerAPI()
    DATA_VERSION = "v4.3"
    featureset = "small"
    logging.info("Downloading dataset...")
    #napi.download_dataset(f"{DATA_VERSION}/validation_int8.parquet")
    #napi.download_dataset(f"{DATA_VERSION}/live_int8.parquet")
    #napi.download_dataset(f"{DATA_VERSION}/features.json")
    feature_metadata = json.load(open(f"{DATA_VERSION}/features.json"))
    feature_set = feature_metadata["feature_sets"][featureset]
    targets = feature_metadata["targets"]

    # Load and preprocess data
    logging.info("Loading and preprocessing data...")
    data = pd.read_parquet(f"{DATA_VERSION}/validation_int8.parquet", columns=["era"] + targets + feature_set)
    data = data[data["era"].isin(data["era"].unique()[::36])]


    # Handle missing values
    data = data.dropna()

    # Separate features and target
    X = data[feature_set]
    y = data[targets[0]]  # Assuming we're predicting the first target

    # Normalize features
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    # Feature selection
    logging.info("Performing feature selection...")
    correlation_threshold = 0.8
    correlation_matrix = X_scaled.corr().abs()
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]
    X_scaled = X_scaled.drop(to_drop, axis=1)

    mi_scores = mutual_info_regression(X_scaled, y)
    mi_threshold = np.percentile(mi_scores, 70)  # Keep top 30% features
    selected_features = X_scaled.columns[mi_scores > mi_threshold].tolist()
    X_selected = X_scaled[selected_features]

    # Feature engineering
    logging.info("Performing feature engineering...")
    X_engineered = X_selected.copy()

    # Interaction features
    for i in range(min(10, len(selected_features))):
        for j in range(i + 1, min(11, len(selected_features))):
            X_engineered[f"interaction_{i}_{j}"] = X_selected.iloc[:, i] * X_selected.iloc[:, j]

    # Time-based features
    X_engineered['era'] = data['era']
    for feature in selected_features[:5]:  # Use top 5 features for rolling statistics
        X_engineered[f"{feature}_rolling_mean"] = X_engineered.groupby('era')[feature].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
        X_engineered[f"{feature}_rolling_std"] = X_engineered.groupby('era')[feature].transform(
            lambda x: x.rolling(5, min_periods=1).std())

    X_engineered = X_engineered.drop('era', axis=1)

    # Define optimization objective
    def objective(trial):
        lgbm_params = {
            'n_estimators': trial.suggest_int('lgbm_n_estimators', 100, 1000),
            'learning_rate': trial.suggest_loguniform('lgbm_learning_rate', 1e-3, 1e-1),
            'num_leaves': trial.suggest_int('lgbm_num_leaves', 20, 3000),
            'max_depth': trial.suggest_int('lgbm_max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('lgbm_min_child_samples', 1, 300),
            'subsample': trial.suggest_uniform('lgbm_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_uniform('lgbm_colsample_bytree', 0.6, 1.0),
        }

        xgb_params = {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 1000),
            'learning_rate': trial.suggest_loguniform('xgb_learning_rate', 1e-3, 1e-1),
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 300),
            'subsample': trial.suggest_uniform('xgb_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_uniform('xgb_colsample_bytree', 0.6, 1.0),
        }

        cb_params = {
            'iterations': trial.suggest_int('cb_iterations', 100, 1000),
            'learning_rate': trial.suggest_loguniform('cb_learning_rate', 1e-3, 1e-1),
            'depth': trial.suggest_int('cb_depth', 3, 12),
            'min_data_in_leaf': trial.suggest_int('cb_min_data_in_leaf', 1, 300),
            'bagging_temperature': trial.suggest_uniform('cb_bagging_temperature', 0, 1),
        }

        models = [
            ('lgbm', LGBMRegressor(**lgbm_params)),
            ('xgb', XGBRegressor(**xgb_params)),
            ('catboost', CatBoostRegressor(**cb_params, silent=True))
        ]

        final_estimator = LGBMRegressor(n_estimators=100)
        model = StackingRegressor(estimators=models, final_estimator=final_estimator, cv=5)

        tscv = TimeSeriesSplit(n_splits=5)
        scores = []

        for train_index, val_index in tscv.split(X_engineered):
            X_train, X_val = X_engineered.iloc[train_index], X_engineered.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            model.fit(X_train, y_train)
            predictions = model.predict(X_val)

            # Calculate correlation
            correlation = np.corrcoef(y_val, predictions)[0, 1]
            scores.append(correlation)

        return np.mean(scores)

    # Hyperparameter tuning
    logging.info("Performing hyperparameter tuning...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    # Train final model
    logging.info("Training final model...")
    best_params = study.best_params
    lgbm_final = LGBMRegressor(**{k[5:]: v for k, v in best_params.items() if k.startswith('lgbm_')})
    xgb_final = XGBRegressor(**{k[4:]: v for k, v in best_params.items() if k.startswith('xgb_')})
    cb_final = CatBoostRegressor(**{k[3:]: v for k, v in best_params.items() if k.startswith('cb_')}, silent=True)

    models = [
        ('lgbm', lgbm_final),
        ('xgb', xgb_final),
        ('catboost', cb_final)
    ]

    final_estimator = LGBMRegressor(n_estimators=100)
    final_model = StackingRegressor(estimators=models, final_estimator=final_estimator, cv=5)
    final_model.fit(X_engineered, y)

    # Feature neutralization
    logging.info("Applying feature neutralization...")

    def neutralize(df, columns, neutralizers, proportion=1.0):
        neutralizers = df[neutralizers]
        scores = df[columns]
        neutralized = scores - proportion * neutralizers.dot(
            np.linalg.pinv(neutralizers.values).dot(scores.values).T).T
        return neutralized

    neutralizers = selected_features[:20]  # Use top 20 features for neutralization
    predictions = final_model.predict(X_engineered)
    neutralized_predictions = neutralize(pd.DataFrame(predictions, columns=['prediction'], index=X_engineered.index),
                                         ['prediction'], X_engineered[neutralizers])

    # Define prediction function
    def predict(live_features: pd.DataFrame) -> pd.DataFrame:
        live_features_scaled = pd.DataFrame(scaler.transform(live_features), columns=live_features.columns,
                                            index=live_features.index)
        live_features_selected = live_features_scaled[selected_features]

        # Apply feature engineering to live data
        live_engineered = live_features_selected.copy()
        for i in range(min(10, len(selected_features))):
            for j in range(i + 1, min(11, len(selected_features))):
                live_engineered[f"interaction_{i}_{j}"] = live_features_selected.iloc[:,
                                                          i] * live_features_selected.iloc[:, j]

        live_predictions = final_model.predict(live_engineered)
        neutralized_live_predictions = neutralize(
            pd.DataFrame(live_predictions, columns=['prediction'], index=live_engineered.index),
            ['prediction'], live_engineered[neutralizers])

        return neutralized_live_predictions

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