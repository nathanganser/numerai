from numerapi import NumerAPI
import pandas as pd
import json
napi = NumerAPI()

# use one of the latest data versions
DATA_VERSION = "v4.3"

# Download data
#napi.download_dataset(f"{DATA_VERSION}/train_int8.parquet")
#napi.download_dataset(f"{DATA_VERSION}/features.json")

# Load data
feature_metadata = json.load(open(f"{DATA_VERSION}/features.json"))
features = feature_metadata["feature_sets"]["small"] # use "all" for better performance. Requires more RAM.
train = pd.read_parquet(f"{DATA_VERSION}/train_int8.parquet", columns=["era"]+features+["target"])


# Downsample for speed
train = train[train["era"].isin(train["era"].unique()[::4])]  # skip this step for better performance

# Train model
import lightgbm as lgb

model = lgb.LGBMRegressor(
    n_estimators=2000,  # If you want to use a larger model we've found 20_000 trees to be better
    learning_rate=0.01, # and a learning rate of 0.001
    max_depth=5, # and max_depth=6
    num_leaves=2**5-1, # and num_leaves of 2**6-1
    colsample_bytree=0.1
)
print("Fitting...")
model.fit(
    train[features],
    train["target"]
)

# Define predict function
def predict(
    live_features: pd.DataFrame,
    live_benchmark_models: pd.DataFrame
 ) -> pd.DataFrame:
    live_predictions = model.predict(live_features[features])
    submission = pd.Series(live_predictions, index=live_features.index)
    return submission.to_frame("prediction")

# Pickle predict function
import cloudpickle
p = cloudpickle.dumps(predict)
with open("predict_barebones.pkl", "wb") as f:
    f.write(p)

