# numerai
Exploring Numerai


` modal run --detach main2.py`

## Next steps
- Find the most crazy features and neutralise them. 
- Submit


brev.dev -> NVIDIA RTX A6000 (48GiB)



`sudo apt install screen`

`screen -S compute`

python3 -m venv env
`source env/bin/activate`

`pip install numerapi pandas numpy==1.26.4 cloudpickle==2.2.1 requests pyarrow lightgbm catboost scikit-learn optuna`


`python compute.py`


screen -D -R compute

```sudo apt install screen && screen -S compute bash -c "python3 -m venv env && source env/bin/activate && pip install numerapi pandas numpy==1.26.4 cloudpickle==2.2.1 requests pyarrow lightgbm catboost scikit-learn optuna && python compute.py"```


# Prompt
You are an extremely skilled and competent AI engineer, data scientist and finance quant. 
You have years of experience participating in the Numerai tournament and creating great models that outperform 95% of the other submitted models. 
You are about to create the most performant model of your career and will stake a lot of NUM on it. 
Keep in mind that Numerai data is obfuscated and that features and targets are normalised.

Your starting code looks like this:
```
logging.basicConfig(level=logging.INFO)
model_name = "llama31"

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
```

And the end of your code will look something like this:
```
    # Define prediction function
    def predict(live_features: pd.DataFrame) -> pd.DataFrame:
        live_predictions = model.predict(live_features[selected_features])
        submission = pd.Series(live_predictions, index=live_features.index)
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
```

There are about 3000 features and 700 eras in the dataset and your code will run on an NVIDIA RTX A6000 (48GiB) GPU. 

Think for a very long time first. Brainstorm with yourself first for a while. Jot down ideas on how to go about this. 
You need to complete the compute_me function in a way that will generate the best possible model. Everything has to happen in that python file. There is no research or data experiments you can do outside of that function. Ensure your code will run flawlessly.
Describe your approach and write the complete compute.py file 
