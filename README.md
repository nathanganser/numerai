# numerai
Exploring Numerai

## Ideas
- Ensemble a model that is super specific (low learning rate, many neurons) and have a model that is super general (high learning rate, few neurons) and square the predictions (the more different the general model, the more it takes over).
- Use 1%, 10% & 100% of the data
- Check how to do evaluations in the code (diagnostics)
- init_model= to conitnue training

` modal run --detach main2.py`

## Next steps
- Find the most crazy features and neutralise them. 
- Submit


brev.dev -> NVIDIA RTX A6000 (48GiB)



`sudo apt install screen -y`
`sudo apt install nvidia-driver-550 nvidia-cuda-toolkit clinfo`

`screen -S compute`

python3 -m venv env
`source env/bin/activate`

`pip install numerapi pandas numpy==1.26.4 cloudpickle==2.2.1 requests catboost scikit-learn==1.2.2 pyarrow`


`python compute.py`


screen -D -R compute

```sudo apt install screen && screen -S compute bash -c "python3 -m venv env && source env/bin/activate && pip install numerapi pandas numpy==1.26.4 cloudpickle==2.2.1 requests pyarrow lightgbm catboost scikit-learn optuna && python compute.py"```


# Prompt
You are an extremely skilled and competent AI engineer, data scientist and finance quant. 
You have years of experience participating in the Numerai tournament and creating great models that outperform 95% of the other submitted models. 
You are about to create the most performant model of your career and will stake a lot of NUM on it. 
Keep in mind that Numerai data is obfuscated and that features and targets are normalised. Therefore, do not do any feature engineering or normalising. Keep the features as is and don't create or remove features.
Ensure all the parameters are variables set at the beginning of the code with minimal test values (minimal values so the code runs fast) and commented production values (which you think will lead to the best model)

Your starting code looks like this:
```
logging.basicConfig(level=logging.INFO)
model_name = "model"

def compute_me():
    napi = NumerAPI()
    DATA_VERSION = "v4.3"
    featureset = "small" # 
    logging.info("Downloading dataset...")
    #napi.download_dataset(f"{DATA_VERSION}/validation_int8.parquet")
    #napi.download_dataset(f"{DATA_VERSION}/live_int8.parquet")
    #napi.download_dataset(f"{DATA_VERSION}/features.json")
    feature_metadata = json.load(open(f"{DATA_VERSION}/features.json"))
    feature_set = feature_metadata["feature_sets"][featureset]
    targets = feature_metadata["targets"]
    feature_count = len(feature_set)

    # Load and preprocess data
    logging.info("Loading and preprocessing data...")
    data = pd.read_parquet(f"{DATA_VERSION}/validation_int8.parquet", columns=["era"] + targets + feature_set)
    
    # to reduce the data amount for testing
    data = data[data["era"].isin(data["era"].unique()[::72])]

    # Handle missing values
    data = data.dropna()
```

And the end of your code will look something like this:
```
    # Define prediction function
    def predict(live_features: pd.DataFrame) -> pd.DataFrame:
        live_predictions = model.predict(live_features[selected_features])
        submission = pd.Series(live_predictions, index=live_features.index)
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
```

data (validation_int8.parquet) looks something like this:
id era target feature_abating_unadaptable_weakfish feature_ablest_mauritanian_elding ...
n003bba8a98662e4 0001 0.25 0 4
... ... ... ... ...
nffc2d5e4b79a7ae 0573 0.75	4	2

and live_data (live_int8.parquet) looks like that
id feature_acclimatisable_unfeigned_maghreb  ...  feature_wistful_tussive_cycloserine                                  
n001a3f99a7ad339 3  ...  0
n0027a46913f6384 4  ... 3

There are about 3000 features and 700 eras in the dataset and your code will run on an NVIDIA RTX A6000 (48GiB) GPU. 
Do not use tensorflow as a dependency.
Think for a very long time first. Brainstorm with yourself first for a while. Jot down ideas on how to go about this. 
You need to complete the compute_me function in a way that will generate the best possible model. Everything has to happen in that python file. There is no research or data experiments you can do outside of that function. Ensure your code will run flawlessly.
Describe your approach and write the complete compute.py file 

