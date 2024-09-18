import json
from numerapi import NumerAPI
import pandas as pd
import cloudpickle
napi = NumerAPI()
DATA_VERSION = "v5.0"
FEATURESET = 'all'

napi.download_dataset(f"{DATA_VERSION}/features.json")

feature_metadata = json.load(open(f"{DATA_VERSION}/features.json"))

feature_set = feature_metadata["feature_sets"][FEATURESET]


# Test prediction function using the pickle file
napi.download_dataset(f"{DATA_VERSION}/live.parquet")

print("Testing prediction function from pickle file...")
live_data = pd.read_parquet(f"{DATA_VERSION}/live.parquet", columns=["era"] + feature_set)

# Load the pickled prediction function
with open("model.pkl", "rb") as f:
    loaded_predict = cloudpickle.loads(f.read())

# Use the loaded function to make predictions
predictions = loaded_predict(live_data)
print(f"Predictions shape: {predictions.shape}")
print(f"Predictions head:\n{predictions.head()}")