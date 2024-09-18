import pandas as pd
import numpy as np
import json
import gc
import logging
import cloudpickle
from numerapi import NumerAPI
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from typing import List

# Constants
DATA_VERSION = 'v5.0'  # Replace with the actual data version
FEATURESET = 'all'     # Using all features
TARGET = 'target'
SEED = 42
num_secondary_targets = 3
ITERATIONS = 100000
DEPTH = 6
n_splits = 10


# Initialize NumerAPI
napi = NumerAPI()

# Download feature metadata
napi.download_dataset(f"{DATA_VERSION}/features.json")
with open(f"{DATA_VERSION}/features.json", 'r') as f:
    feature_metadata = json.load(f)

# Get the feature set
feature_set = feature_metadata["feature_sets"][FEATURESET]
print(f"Number of features in '{FEATURESET}': {len(feature_set)}")

# Download and load training data
napi.download_dataset(f"{DATA_VERSION}/validation.parquet")
train = pd.read_parquet(f"{DATA_VERSION}/validation.parquet")
print(f"Training data shape: {train.shape}")

import pandas as pd
import numpy as np



# Assuming 'train' is your DataFrame
# Select all columns that start with 'target_'
all_targets = [col for col in train.columns if col.startswith('target_')]

# Calculate correlations between all target columns
target_cols = [col for col in train.columns if col.startswith('target_')] + [TARGET]
corr_matrix = train[target_cols].corr()

# Display correlation of each target with the main target
main_target_corr = corr_matrix[TARGET].abs().sort_values()

# Print correlation matrix
print(f"Correlation matrix of targets done")

# Sort correlations and select the least correlated secondary targets
main_target_corr_sorted = main_target_corr.sort_values()

# Remove the main target from the sorted correlations if present
if TARGET in main_target_corr_sorted.index:
    main_target_corr_sorted = main_target_corr_sorted.drop(TARGET)

# Select a fixed number of least correlated targets (e.g., 3)
selected_secondary_targets = main_target_corr_sorted.index[:num_secondary_targets].tolist()

print(f"Selected secondary targets based on lowest correlation: {selected_secondary_targets}")

if 'era' in train.columns:
    # Handling NaN values: Avoid calculating the median for groups that are entirely NaN
    for col in selected_secondary_targets:
        train[col] = train.groupby('era')[col].transform(
            lambda x: x.fillna(x.median()) if not x.isna().all() else x.fillna(0)
        )

# Combine main and secondary targets for multi-target regression
all_targets = [TARGET] + selected_secondary_targets

if 'era' in train.columns:
    # Handling NaN values: Avoid calculating the median for groups that are entirely NaN
    for col in all_targets:
        train[col] = train.groupby('era')[col].transform(
            lambda x: x.fillna(x.median()) if not x.isna().all() else x.fillna(0)
        )

# Prepare feature matrix and target vector
X = train[feature_set]
y = train[all_targets]



# Era-wise GroupKFold Cross-Validation
if 'era' in train.columns:
    eras = train['era']
    groups = eras.astype('category').cat.codes.values
else:
    # If 'era' is not present, use a simple KFold
    groups = np.zeros(len(X))

# Initialize CatBoost parameters
catboost_params = {
    'iterations': ITERATIONS,
    'learning_rate': 0.019,
    'depth': DEPTH,
    'loss_function': 'MultiRMSE',
    'eval_metric': 'MultiRMSE',
    'random_seed': SEED,
    'od_type': 'Iter',
    'od_wait': 500,
    'task_type': 'GPU',
    'verbose': round(ITERATIONS/3),
    'bagging_temperature': 0.5,
    'leaf_estimation_iterations': 10,
    'leaf_estimation_method': 'Newton',
    'l2_leaf_reg': 6,
    'min_data_in_leaf': 100
}

# Cross-validation setup
gkf = GroupKFold(n_splits=n_splits)
models: List[CatBoostRegressor] = []
scores = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y[TARGET], groups)):
    print(f"Starting fold {fold + 1}/{n_splits}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)

    model = CatBoostRegressor(**catboost_params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    # Evaluate
    val_preds = model.predict(X_val)
    score = mean_squared_error(y_val, val_preds, squared=False)
    scores.append(score)
    print(f"Fold {fold + 1} RMSE: {score:.5f}")

    models.append(model)


print(f"Average RMSE across folds: {np.mean(scores):.5f}")



# Prediction function
def predict(live_features: pd.DataFrame) -> pd.DataFrame:
    # Prepare features
    X_live = live_features[feature_set]

    # Aggregate predictions from all models
    preds = np.zeros((X_live.shape[0], len(all_targets)))
    for model in models:
        preds += model.predict(X_live)
    preds /= len(models)

    # Use the main target predictions
    live_predictions = preds[:, 0]  # Assuming TARGET is the first in all_targets

    # Prepare submission DataFrame
    submission = pd.Series(live_predictions, index=live_features.index)
    print("Submission shape:, ", submission.shape)
    return submission.to_frame("prediction")

# Serialize and upload prediction function
print("Serializing and uploading prediction function...")
p = cloudpickle.dumps(predict)
with open("model.pkl", "wb") as f:
    f.write(p)

print("Model serialization complete. Ready for submission.")
# Load live data
napi.download_dataset(f"{DATA_VERSION}/live.parquet")
live_features = pd.read_parquet(f"{DATA_VERSION}/live.parquet")
print(f"Live data shape: {live_features.shape}")

with open("model.pkl", "rb") as f:
    loaded_predict = cloudpickle.loads(f.read())

# Use the loaded function to make predictions
predictions = loaded_predict(live_features)
print(f"Predictions shape: {predictions.shape}")
print(f"Predictions head:\n{predictions.head()}")
