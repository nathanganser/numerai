import logging
import json
import pandas as pd
import numpy as np
from numerapi import NumerAPI
import requests
import cloudpickle
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import shap

logging.basicConfig(level=logging.INFO)
model_name = "claude_ensemble"


def compute_me():
    napi = NumerAPI()
    DATA_VERSION = "v5"
    featureset = "all"
    logging.info("Downloading dataset...")
    napi.download_dataset(f"{DATA_VERSION}/validation_int8.parquet")
    napi.download_dataset(f"{DATA_VERSION}/live_int8.parquet")
    napi.download_dataset(f"{DATA_VERSION}/features.json")
    feature_metadata = json.load(open(f"{DATA_VERSION}/features.json"))
    feature_set = feature_metadata["feature_sets"][featureset]
    targets = feature_metadata["targets"]
    feature_count = len(feature_set)

    # Load and preprocess data
    logging.info("Loading and preprocessing data...")
    data = pd.read_parquet(f"{DATA_VERSION}/validation_int8.parquet", columns=["era"] + targets + feature_set)

    # Handle missing values
    data = data.dropna()

    # Separate features and target
    X = data[feature_set]
    y = data[targets[0]]  # Assuming we're predicting the first target

    # Feature engineering and selection
    logging.info("Performing feature engineering and selection...")

    # Quantile transformation
    qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal')
    X_transformed = pd.DataFrame(qt.fit_transform(X), columns=X.columns, index=X.index)

    # Feature importance analysis
    mi_scores = mutual_info_regression(X_transformed, y)
    important_features = X_transformed.columns[np.argsort(mi_scores)[-100:]]

    # Create interaction features
    for i in range(len(important_features)):
        for j in range(i + 1, len(important_features)):
            X_transformed[f"{important_features[i]}_{important_features[j]}"] = X_transformed[important_features[i]] * \
                                                                                X_transformed[important_features[j]]

    # Dimensionality reduction
    pca = PCA(n_components=50)
    pca_features = pca.fit_transform(X_transformed)
    pca_df = pd.DataFrame(pca_features, columns=[f"pca_{i}" for i in range(50)], index=X_transformed.index)
    X_transformed = pd.concat([X_transformed, pca_df], axis=1)

    # Feature clustering
    kmeans = KMeans(n_clusters=100)
    cluster_labels = kmeans.fit_predict(X_transformed)
    selected_features = []
    for cluster in range(100):
        cluster_features = X_transformed.columns[cluster_labels == cluster]
        selected_features.append(cluster_features[np.argmax(mi_scores[cluster_labels == cluster])])

    X_final = X_transformed[selected_features]

    # Model development
    logging.info("Developing ensemble model...")

    def objective(params):
        lgb_model = LGBMRegressor(**params['lgb'])
        xgb_model = XGBRegressor(**params['xgb'])
        nn_model = MLPRegressor(**params['nn'])

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for train_index, val_index in kf.split(X_final):
            X_train, X_val = X_final.iloc[train_index], X_final.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            lgb_model.fit(X_train, y_train)
            xgb_model.fit(X_train, y_train)
            nn_model.fit(X_train, y_train)

            lgb_preds = lgb_model.predict(X_val)
            xgb_preds = xgb_model.predict(X_val)
            nn_preds = nn_model.predict(X_val)

            stacked_features = np.column_stack((lgb_preds, xgb_preds, nn_preds))
            meta_model = LinearRegression()
            meta_model.fit(stacked_features, y_val)

            final_preds = meta_model.predict(stacked_features)
            score = np.corrcoef(final_preds, y_val)[0, 1]
            scores.append(score)

        return {'loss': -np.mean(scores), 'status': STATUS_OK}

    space = {
        'lgb': {
            'n_estimators': hp.quniform('lgb_n_estimators', 100, 1000, 50),
            'learning_rate': hp.loguniform('lgb_learning_rate', np.log(0.01), np.log(0.2)),
            'num_leaves': hp.quniform('lgb_num_leaves', 20, 200, 10),
            'feature_fraction': hp.uniform('lgb_feature_fraction', 0.5, 1.0),
        },
        'xgb': {
            'n_estimators': hp.quniform('xgb_n_estimators', 100, 1000, 50),
            'learning_rate': hp.loguniform('xgb_learning_rate', np.log(0.01), np.log(0.2)),
            'max_depth': hp.quniform('xgb_max_depth', 3, 10, 1),
            'subsample': hp.uniform('xgb_subsample', 0.5, 1.0),
        },
        'nn': {
            'hidden_layer_sizes': hp.choice('nn_hidden_layer_sizes', [
                (50,), (100,), (50, 50), (100, 50), (100, 100)
            ]),
            'alpha': hp.loguniform('nn_alpha', np.log(1e-5), np.log(1e-2)),
        }
    }

    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=50, trials=trials)

    # Train final model with best hyperparameters
    logging.info("Training final model...")
    lgb_model = LGBMRegressor(**best['lgb'])
    xgb_model = XGBRegressor(**best['xgb'])
    nn_model = MLPRegressor(**best['nn'])

    lgb_model.fit(X_final, y)
    xgb_model.fit(X_final, y)
    nn_model.fit(X_final, y)

    # SHAP analysis for feature importance
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X_final)

    # Define prediction function
    def predict(live_features: pd.DataFrame) -> pd.DataFrame:
        live_features_transformed = qt.transform(live_features)
        live_features_final = live_features_transformed[selected_features]

        lgb_preds = lgb_model.predict(live_features_final)
        xgb_preds = xgb_model.predict(live_features_final)
        nn_preds = nn_model.predict(live_features_final)

        stacked_features = np.column_stack((lgb_preds, xgb_preds, nn_preds))
        meta_model = LinearRegression()
        meta_model.fit(stacked_features, y)

        final_preds = meta_model.predict(stacked_features)

        # Neutralization
        feature_exposures = shap_values.mean(axis=0)
        neutralized_preds = final_preds - live_features_final.dot(feature_exposures)

        submission = pd.Series(neutralized_preds, index=live_features.index)
        return submission.to_frame("prediction")

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