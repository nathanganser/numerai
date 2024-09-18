import logging
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
import cloudpickle
import requests
from numerapi import NumerAPI

logging.basicConfig(level=logging.INFO)
model_name = "gpu"

# Hyperparameters
NUM_TRIALS = 10  # Minimal test value
# NUM_TRIALS = 100  # Production value

NUM_BOOST_ROUND = 100  # Minimal test value
# NUM_BOOST_ROUND = 10000  # Production value

NUM_EPOCHS = 10  # Minimal test value
# NUM_EPOCHS = 200  # Production value

N_SPLITS = 3  # Minimal test value
# N_SPLITS = 5  # Production value

BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
FEATURE_FRACTION = 0.8
NUM_LEAVES = 31
MAX_DEPTH = -1
MIN_DATA_IN_LEAF = 20
BAGGING_FRACTION = 0.8
BAGGING_FREQ = 1


class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


def train_lgbm(X, y, params):
    model = lgb.LGBMRegressor(**params, n_jobs=-1, device='gpu')
    model.fit(X, y)
    return model


def train_nn(X, y, input_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNet(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    dataset = TensorDataset(torch.FloatTensor(X).to(device), torch.FloatTensor(y).to(device))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()

    return model


def objective(trial, X, y, cv):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'num_leaves': trial.suggest_int('num_leaves', 20, 3000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
    }

    scores = []
    for train_idx, val_idx in cv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMRegressor(**params, n_estimators=NUM_BOOST_ROUND, n_jobs=-1, device='gpu')
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        preds = model.predict(X_val)
        score = mean_squared_error(y_val, preds, squared=False)
        scores.append(score)

    return np.mean(scores)


def neutralize(df, columns, neutralizers, proportion=1.0, normalize=True, era_col="era"):
    unique_eras = df[era_col].unique()
    computed = []
    for era in unique_eras:
        era_data = df[df[era_col] == era]
        features = era_data[columns]
        neutralizers = era_data[neutralizers].values

        scores = era_data[columns].values
        exposed = neutralizers.dot(neutralizers.T)
        scores -= proportion * exposed.dot(scores)

        if normalize:
            scores /= np.sqrt((scores ** 2).sum())

        computed.append(pd.DataFrame(scores, columns=columns, index=era_data.index))

    return pd.concat(computed, axis=0)


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
    feature_count = len(feature_set)

    logging.info("Loading and preprocessing data...")
    data = pd.read_parquet(f"{DATA_VERSION}/validation_int8.parquet", columns=["era"] + targets + feature_set)

    # Reduce data amount for testing
    data = data[data["era"].isin(data["era"].unique()[::72])]

    # Handle missing values
    data = data.dropna()

    X = data[feature_set].values
    y_multi = data[targets].values

    # Feature selection using correlation with targets
    corr_matrix = np.abs(np.corrcoef(X.T, y_multi.T)[-len(targets):, :-len(targets)])
    top_features = np.argsort(corr_matrix.max(axis=0))[-int(feature_count * FEATURE_FRACTION):]
    X = X[:, top_features]
    selected_features = [feature_set[i] for i in top_features]

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    # Train models for each target
    lgbm_models = []
    nn_models = []

    for target_idx, target in enumerate(targets):
        y = y_multi[:, target_idx]

        logging.info(f"Optimizing LightGBM for target: {target}")
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, X, y, tscv), n_trials=NUM_TRIALS)

        best_params = study.best_params
        best_params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
        })

        logging.info(f"Training LightGBM for target: {target}")
        lgbm_model = train_lgbm(X, y, best_params)
        lgbm_models.append(lgbm_model)

        logging.info(f"Training Neural Network for target: {target}")
        nn_model = train_nn(X, y, X.shape[1])
        nn_models.append(nn_model)

    # Define prediction function
    def x(live_features: pd.DataFrame) -> pd.DataFrame:
        # Preprocess live features
        live_features = live_features[selected_features]
        live_features = scaler.transform(live_features)

        # Make predictions
        lgbm_preds = np.mean([model.predict(live_features) for model in lgbm_models], axis=0)

        nn_preds = []
        for model in nn_models:
            model.eval()
            with torch.no_grad():
                nn_pred = model(torch.FloatTensor(live_features).cuda()).cpu().numpy().squeeze()
            nn_preds.append(nn_pred)
        nn_preds = np.mean(nn_preds, axis=0)

        # Ensemble predictions
        final_preds = (lgbm_preds + nn_preds) / 2

        # Apply neutralization
        neutralizers = feature_set[:20]  # Use first 20 features as neutralizers
        neutral_preds = neutralize(pd.DataFrame(final_preds, columns=["prediction"], index=live_features.index),
                                   ["prediction"], neutralizers)

        return neutral_preds

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