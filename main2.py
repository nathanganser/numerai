import numpy as np
import requests
import json
import pandas as pd
import lightgbm as lgb
import cloudpickle
from numerapi import NumerAPI

import logging

logging.basicConfig(level=logging.INFO)
model_name = "claude_v2"

def neutralize(
    df: pd.DataFrame,
    neutralizers: np.ndarray,
    proportion: float = 1.0,
) -> pd.DataFrame:
    """Neutralize each column of a given DataFrame by each feature in a given
    neutralizers DataFrame. Neutralization uses least-squares regression to
    find the orthogonal projection of each column onto the neutralizers, then
    subtracts the result from the original predictions.

    Arguments:
        df: pd.DataFrame - the data with columns to neutralize
        neutralizers: pd.DataFrame - the neutralizer data with features as columns
        proportion: float - the degree to which neutralization occurs

    Returns:
        pd.DataFrame - the neutralized data
    """
    assert not neutralizers.isna().any().any(), "Neutralizers contain NaNs"
    assert len(df.index) == len(neutralizers.index), "Indices don't match"
    assert (df.index == neutralizers.index).all(), "Indices don't match"
    df[df.columns[df.std() == 0]] = np.nan
    df_arr = df.values
    neutralizer_arr = neutralizers.values
    neutralizer_arr = np.hstack(
        # add a column of 1s to the neutralizer array in case neutralizer_arr is a single column
        (neutralizer_arr, np.array([1] * len(neutralizer_arr)).reshape(-1, 1))
    )
    inverse_neutralizers = np.linalg.pinv(neutralizer_arr, rcond=1e-6)
    adjustments = proportion * neutralizer_arr.dot(inverse_neutralizers.dot(df_arr))
    neutral = df_arr - adjustments
    return pd.DataFrame(neutral, index=df.index, columns=df.columns)


def compute_me():
    # Initialize NumerAPI
    napi = NumerAPI()

    # Set appropriate data version
    DATA_VERSION = "v4.3"

    # Download datasets
    logging.info("Downloading datasets...")
    napi.download_dataset(f"{DATA_VERSION}/train_int8.parquet")
    napi.download_dataset(f"{DATA_VERSION}/features.json")
    napi.download_dataset(f"{DATA_VERSION}/validation_int8.parquet")

    # Load features metadata
    logging.info("Loading features metadata...")
    feature_metadata = json.load(open(f"{DATA_VERSION}/features.json"))
    feature_cols = feature_metadata["feature_sets"]["small"]
    feature_set = feature_cols
    target_cols = feature_metadata["targets"]
    train = pd.read_parquet(
        f"{DATA_VERSION}/train_int8.parquet",
        columns=["era"] + feature_cols + target_cols
    )
    train = train[train["era"].isin(train["era"].unique()[::36])]

    # Load validation data
    logging.info("Loading validation data...")
    validation = pd.read_parquet(f"{DATA_VERSION}/validation_int8.parquet",
                                 columns=["era", "data_type", "target"] + feature_set)
    validation = validation[validation["data_type"] == "validation"]
    del validation["data_type"]

    logging.info("Preprocessing validation data...")
    validation = validation[validation["era"].isin(validation["era"].unique()[::4])]
    eras_to_embargo = [str(int(train["era"].unique()[-1]) + i).zfill(4) for i in range(4)]
    validation = validation[~validation["era"].isin(eras_to_embargo)]

    # Select auxiliary targets and drop redundant columns
    target_candidates = [col for col in feature_metadata["targets"] if col != 'target']
    train = train.drop(columns=['target'])

    # Print available target columns for debugging
    target_candidates = target_candidates[0:2]

    models = {}

    logging.info("Training models for auxiliary targets...")
    # Train models
    for target in target_candidates:
        if target not in train.columns:
            logging.error(f"Target column {target} not found in training data. Skipping.")
            continue

        logging.info(f"Training {target} model...")
        model = lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.005,
            max_depth=6,
            num_leaves=2 ** 6 - 1,
            colsample_bytree=0.2,
            subsample=0.8,
            reg_alpha=0.1,
            reg_lambda=0.3,
            random_state=42
        )
        train_target = train.dropna(subset=[target])
        model.fit(train_target[feature_set], train_target[target])
        models[target] = model

    # Generate validation predictions
    logging.info("Generating validation predictions...")
    for target in target_candidates:
        if target in models:
            validation[f"prediction_{target}"] = models[target].predict(validation[feature_set])
    validation_preds = [f"prediction_{target}" for target in target_candidates if
                        f"prediction_{target}" in validation.columns]

    # Feature neutralization
    logging.info("Neutralizing features...")
    for pred in validation_preds:
        neutralized = validation.groupby("era").apply(lambda x: neutralize(x[[pred]], x[feature_set], proportion=0.5))
        validation[f"neutralized_{pred}"] = neutralized.values

    neutralized_preds = [f"neutralized_{pred}" for pred in validation_preds]

    # Ensemble approach
    logging.info("Applying ensemble approach...")
    validation["ensemble_predictions"] = validation.groupby("era")[neutralized_preds].rank(pct=True).mean(axis=1)

    # Function to generate live predictions
    def predict(live_features):
        def neutralize(
                df: pd.DataFrame,
                neutralizers: np.ndarray,
                proportion: float = 1.0,
        ) -> pd.DataFrame:
            """Neutralize each column of a given DataFrame by each feature in a given
            neutralizers DataFrame. Neutralization uses least-squares regression to
            find the orthogonal projection of each column onto the neutralizers, then
            subtracts the result from the original predictions.

            Arguments:
                df: pd.DataFrame - the data with columns to neutralize
                neutralizers: pd.DataFrame - the neutralizer data with features as columns
                proportion: float - the degree to which neutralization occurs

            Returns:
                pd.DataFrame - the neutralized data
            """
            assert not neutralizers.isna().any().any(), "Neutralizers contain NaNs"
            assert len(df.index) == len(neutralizers.index), "Indices don't match"
            assert (df.index == neutralizers.index).all(), "Indices don't match"
            df[df.columns[df.std() == 0]] = np.nan
            df_arr = df.values
            neutralizer_arr = neutralizers.values
            neutralizer_arr = np.hstack(
                # add a column of 1s to the neutralizer array in case neutralizer_arr is a single column
                (neutralizer_arr, np.array([1] * len(neutralizer_arr)).reshape(-1, 1))
            )
            inverse_neutralizers = np.linalg.pinv(neutralizer_arr, rcond=1e-6)
            adjustments = proportion * neutralizer_arr.dot(inverse_neutralizers.dot(df_arr))
            neutral = df_arr - adjustments
            return pd.DataFrame(neutral, index=df.index, columns=df.columns)

        live_predictions = pd.DataFrame(index=live_features.index)
        for target in target_candidates:
            if target in models:
                live_predictions[target] = models[target].predict(live_features[feature_set])

        neutralized_live = pd.DataFrame(index=live_features.index)
        for target in target_candidates:
            if target in live_predictions:
                neutralized_live[target] = (
                    neutralize(pd.DataFrame(live_predictions[target]), live_features[feature_set],
                               proportion=0.5)).values.ravel()

        combined_live = neutralized_live.rank(pct=True).mean(axis=1)
        submission = combined_live.rank(pct=True, method="first")
        return submission.to_frame("prediction")

    # Serialize the function using cloudpickle
    logging.info("Serializing prediction function...")
    pickle_path = f"{model_name}.pkl"
    with open(pickle_path, "wb") as f:
        cloudpickle.dump(predict, f)

    # Test predictions
    logging.info("Testing prediction function...")
    napi.download_dataset(f"{DATA_VERSION}/live_int8.parquet")
    live_features = pd.read_parquet(f"{DATA_VERSION}/live_int8.parquet", columns=feature_set)
    example_predictions = predict(live_features)
    logging.info(f"Example Predictions: {example_predictions.head()}")

    # Upload the serialized model
    url = "https://api.bytescale.com/v2/accounts/12a1yew/uploads/form_data"
    headers = {"Authorization": "Bearer public_12a1yewAHfRPdqAXnHXQDib1RwoJ"}
    files = {"file": open(pickle_path, "rb")}
    response = requests.post(url, headers=headers, files=files)
    if response.status_code == 200:
        logging.info("File uploaded successfully.")
    else:
        logging.error(f"File upload failed with status code: {response.status_code}")

if __name__ == "__main__":
    compute_me()