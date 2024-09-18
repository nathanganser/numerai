import logging
import json
import pandas as pd
import numpy as np
from numerapi import NumerAPI
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
import optuna
from scipy.stats import spearmanr
import cloudpickle
import requests

logging.basicConfig(level=logging.INFO)
model_name = "claude_jul"

# Hyperparameters
NUM_ERAS_TEST = 5  # Test value: 5, Production value: 20
NUM_FEATURES = 5  # Test value: 50, Production value: 500
NUM_CLUSTERS = 1  # Test value: 10, Production value: 50
NUM_INTERACTION_FEATURES = 1  # Test value: 10, Production value: 100
NUM_POLYNOMIAL_FEATURES = 1  # Test value: 10, Production value: 100
NUM_TREES = 3  # Test value: 100, Production value: 10000
LEARNING_RATE = 0.1  # Test value: 0.1, Production value: 0.01
NUM_LEAVES = 4  # Test value: 31, Production value: 255
FEATURE_FRACTION = 0.8  # Test value: 0.8, Production value: 0.5
NUM_TRIALS = 2  # Test value: 10, Production value: 100
BATCH_SIZE = 10  # Adjust based on available memory


def compute_me():
    napi = NumerAPI()
    DATA_VERSION = "v4.3"
    featureset = "small"  # 'small' for testing, 'medium' or 'legacy' for production
    logging.info("Downloading dataset...")
    # Uncomment these lines for actual use
    # napi.download_dataset(f"{DATA_VERSION}/validation_int8.parquet")
    # napi.download_dataset(f"{DATA_VERSION}/live_int8.parquet")
    # napi.download_dataset(f"{DATA_VERSION}/features.json")
    feature_metadata = json.load(open(f"{DATA_VERSION}/features.json"))
    feature_set = feature_metadata["feature_sets"][featureset]
    targets = feature_metadata["targets"]
    feature_count = len(feature_set)

    # Load and preprocess data
    logging.info("Loading and preprocessing data...")
    data = pd.read_parquet(f"{DATA_VERSION}/validation_int8.parquet", columns=["era"] + targets + feature_set)

    # Reduce data amount for testing
    data = data[data["era"].isin(data["era"].unique()[::72])]

    # Handle missing values
    data = data.dropna()

    # Feature selection
    def feature_importance(df, features, target):
        correlations = [spearmanr(df[f], df[target])[0] for f in features]
        return [f for _, f in sorted(zip(correlations, features), key=lambda x: abs(x[0]), reverse=True)]

    logging.info("Performing feature selection...")
    important_features = feature_importance(data, feature_set, "target")[:NUM_FEATURES]

    # Feature engineering
    logging.info("Performing feature engineering...")

    def create_interaction_features(df, features, num_interactions):
        interactions = []
        for i in range(num_interactions):
            f1, f2 = np.random.choice(features, 2, replace=False)
            df[f"{f1}_{f2}_interact"] = df[f1] * df[f2]
            interactions.append(f"{f1}_{f2}_interact")
        return interactions

    def create_polynomial_features(df, features, num_polynomial):
        polynomials = []
        for i in range(num_polynomial):
            f = np.random.choice(features)
            df[f"{f}_poly2"] = df[f] ** 2
            polynomials.append(f"{f}_poly2")
        return polynomials

    interaction_features = create_interaction_features(data, important_features, NUM_INTERACTION_FEATURES)
    polynomial_features = create_polynomial_features(data, important_features, NUM_POLYNOMIAL_FEATURES)

    # Feature clustering
    logging.info("Performing feature clustering...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
    cluster_features = [f"cluster_{i}" for i in range(NUM_CLUSTERS)]
    data[cluster_features] = kmeans.fit_transform(data[important_features])

    all_features = important_features + interaction_features + polynomial_features + cluster_features

    # Prepare data for modeling
    X = data[all_features]
    y = data["target"]
    eras = data["era"]

    # Define time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Define objective function for Optuna
    def objective(trial):
        params = {
            "device": "gpu",
            "gpu_platform_id": 0,
            "gpu_device_id": 0,
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 20, 3000),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.1),
            "feature_fraction": trial.suggest_uniform("feature_fraction", 0.5, 1.0),
            "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
            "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }

        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[val_data])
            preds = model.predict(X_val)
            score = spearmanr(y_val, preds)[0]
            scores.append(score)

        return np.mean(scores)

    # Hyperparameter tuning
    logging.info("Performing hyperparameter tuning...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=NUM_TRIALS)

    best_params = study.best_params
    best_params["num_boost_round"] = NUM_TREES

    # Train final LightGBM model
    logging.info("Training final LightGBM model...")
    train_data = lgb.Dataset(X, label=y)
    lgb_model = lgb.train(best_params, train_data)

    # Train Neural Network model
    logging.info("Training Neural Network model...")
    scaler = QuantileTransformer(output_distribution='normal')
    X_scaled = scaler.fit_transform(X)
    nn_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    nn_model.fit(X_scaled, y)

    # Neutralization function
    def neutralize(predictions, features, proportion=0.5):
        exposures = np.dot(features, np.linalg.pinv(features).T)
        correction = proportion * np.dot(exposures, predictions)
        return predictions - correction

    # Define prediction function
    def predict(live_features: pd.DataFrame) -> pd.DataFrame:
        # Add engineered features
        live_interaction_features = create_interaction_features(live_features, important_features,
                                                                NUM_INTERACTION_FEATURES)
        live_polynomial_features = create_polynomial_features(live_features, important_features,
                                                              NUM_POLYNOMIAL_FEATURES)
        live_cluster_features = pd.DataFrame(kmeans.transform(live_features[important_features]),
                                             columns=cluster_features)

        live_all_features = live_features[important_features].join(live_features[live_interaction_features]).join(
            live_features[live_polynomial_features]).join(live_cluster_features)

        # Make predictions
        lgb_preds = lgb_model.predict(live_all_features)
        nn_preds = nn_model.predict(scaler.transform(live_all_features))

        # Ensemble predictions
        ensemble_preds = 0.7 * lgb_preds + 0.3 * nn_preds

        # Neutralize predictions
        neutralized_preds = neutralize(ensemble_preds, live_all_features.values)

        submission = pd.Series(neutralized_preds, index=live_features.index)
        return submission.to_frame("prediction")

    # Test prediction function
    logging.info("Testing prediction function...")
    live_data = pd.read_parquet(f"{DATA_VERSION}/live_int8.parquet", columns=feature_set)
    predictions = predict(live_data)
    logging.info(f"Predictions shape: {predictions.shape}")

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