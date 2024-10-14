import cloudpickle
import logging

from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
import pandas as pd
from numerapi import NumerAPI
import json

# Initialize NumerAPI
napi = NumerAPI()

# Set data version
DATA_VERSION = "v5.0"
PERCENTAGE_DATA_USED = 1
ITERATIONS = 300 # 200000
LEARNING_RATE = 0.01
DEPTH = 3 # 8
FEATURESET = 'small' # medium, all, small

# Download and load feature metadata
napi.download_dataset(f"{DATA_VERSION}/features.json")
feature_metadata = json.load(open(f"{DATA_VERSION}/features.json"))
feature_set = feature_metadata["feature_sets"][FEATURESET]

# Download and load training data
napi.download_dataset(f"{DATA_VERSION}/train.parquet")
train = pd.read_parquet(
    f"{DATA_VERSION}/train.parquet",
    columns=["era", "target"] + feature_set
)

# Reduce train to only be 1% of the data (to speed up training)
train = train.sample(frac=PERCENTAGE_DATA_USED/100, random_state=42)


# Assuming 'train' is your original DataFrame
train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)

# Reset indices for both DataFrames
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# Print the shapes of the resulting DataFrames
print(f"Training set shape: {train_df.shape}")
print(f"Validation set shape: {val_df.shape}")

# prompt: Generate X_train, y_train, X_test, y_test

# Define features and target
features = feature_set
target = "target"

# Create training data
X_train = train_df[features]
y_train = train_df[target]

# Create validation data
X_test = val_df[features]
y_test = val_df[target]


# Create CatBoost pools
train_pool = Pool(X_train,
                  y_train,
)

test_pool = Pool(
    X_test,
    y_test,
)

model = CatBoostRegressor(
    iterations=ITERATIONS,
    learning_rate=LEARNING_RATE,
    depth=DEPTH,
    loss_function='RMSE',
    early_stopping_rounds=round(ITERATIONS/10),
    bagging_temperature = 0.3,
    random_strength=0.3,
    leaf_estimation_iterations=10,
    leaf_estimation_method='Newton',
    #task_type='GPU',
    thread_count=-1)

model.fit(train_pool, eval_set=test_pool, verbose=round(ITERATIONS/10), use_best_model=True);



# Download dataset
napi.download_dataset(f"{DATA_VERSION}/live.parquet")

def predict(live_features: pd.DataFrame) -> pd.DataFrame:
    live_predictions = model.predict(live_features[feature_set])
    submission = pd.Series(live_predictions, index=live_features.index)
    return submission.to_frame("prediction")

# Serialize and upload prediction function
logging.info("Serializing and uploading prediction function...")
p = cloudpickle.dumps(predict)
with open("model.pkl", "wb") as f:
    f.write(p)

# Test prediction function using the pickle file
logging.info("Testing prediction function from pickle file...")
live_data = pd.read_parquet(f"{DATA_VERSION}/live.parquet", columns=["era"] + feature_set)

# Load the pickled prediction function
with open("model.pkl", "rb") as f:
    loaded_predict = cloudpickle.loads(f.read())

# Use the loaded function to make predictions
predictions = loaded_predict(live_data)
print(f"Predictions shape: {predictions.shape}")
print(f"Predictions head:\n{predictions.head()}")
