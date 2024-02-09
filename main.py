from numerapi import NumerAPI
import pandas as pd
import json

from numerai_helper_functions import neutralize

napi = NumerAPI()

# use one of the latest data versions
DATA_VERSION = "v4.3"

# Download data
# napi.download_dataset(f"{DATA_VERSION}/train_int8.parquet")
# napi.download_dataset(f"{DATA_VERSION}/features.json")

# Load data
feature_metadata = json.load(open(f"{DATA_VERSION}/features.json"))
features = feature_metadata["feature_sets"]["all"]  # use "all" for better performance. Requires more RAM.
train = pd.read_parquet(f"{DATA_VERSION}/train_int8.parquet", columns=["era"] + features + ["target"])

print('data is loaded')
# Perform a correlation analysis between all features and the target variable to identify which features are most correlated with the target.

# Calculate correlation matrix
print("correlation matrix...")
correlation_matrix = train.corr()

# Extract correlations with the target variable, excluding the target itself
target_correlations = correlation_matrix['target'].drop('target')

# Sort the correlations to find the most correlated features with the target
sorted_target_correlations = target_correlations.abs().sort_values(ascending=False)

# Identify features to neutralize based on a 0.01 correlation threshold
features_to_neutralize = sorted_target_correlations[sorted_target_correlations > 0.01].index.tolist()

print("Neutralising those features:")
print(features_to_neutralize)

# Perform neutralization
neutralized_train = neutralize(
    df=train,
    columns=features_to_neutralize,  # Features you want to neutralize
    neutralizers=None,
    # In this context, it seems you're neutralizing features based on their correlation, not using specific neutralizers
    proportion=0.3,  # Fully neutralize
    era_col="era"  # Column that identifies the era
)

# Ensure the index is aligned
neutralized_train.index = train.index

# Step 2: Merge neutralized features back into the original dataset
# This step involves replacing the original columns in `train` with their neutralized counterparts
for feature in features_to_neutralize:
    train[feature] = neutralized_train[feature]

# Train model
import lightgbm as lgb

model = lgb.LGBMRegressor(
    n_estimators=2000,  # If you want to use a larger model we've found 20_000 trees to be better
    learning_rate=0.01,  # and a learning rate of 0.001
    max_depth=5,  # and max_depth=6
    num_leaves=2 ** 5 - 1,  # and num_leaves of 2**6-1
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
with open("predict.pkl", "wb") as f:
    f.write(p)
