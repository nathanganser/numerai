import matplotlib
from matplotlib import pyplot as plt
from numerapi import NumerAPI
import pandas as pd
import json

napi = NumerAPI()
from numerai_helper_functions import neutralize
from numerai_tools.scoring import numerai_corr
import numpy as np

# use one of the latest data versions
DATA_VERSION = "v4.3"
matplotlib.use('TkAgg')

keep_fast = False

# Download data
napi.download_dataset(f"{DATA_VERSION}/train_int8.parquet")
napi.download_dataset(f"{DATA_VERSION}/features.json")

# Load data
feature_metadata = json.load(open(f"{DATA_VERSION}/features.json"))
if keep_fast:
    features = feature_metadata["feature_sets"]["small"]  # use "all" for better performance. Requires more RAM.
else:
    features = feature_metadata["feature_sets"]["all"]

train = pd.read_parquet(f"{DATA_VERSION}/train_int8.parquet", columns=["era", "target"] + features)

# Downsample for speed
if keep_fast:
    train = train[train["era"].isin(train["era"].unique()[::4])]  # skip this step for better performance
    train = train[train['era'] > "0400"]

print(train)

# Perform a correlation analysis between all features and the target variable to identify which features are most correlated with the target.

# Calculate correlation matrix
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
    proportion=1.0,  # Fully neutralize
    era_col="era"  # Column that identifies the era
)

# Ensure the index is aligned
neutralized_train.index = train.index

# Step 2: Merge neutralized features back into the original dataset
# This step involves replacing the original columns in `train` with their neutralized counterparts
for feature in features_to_neutralize:
    train[feature] = neutralized_train[feature]

print(train.head())


