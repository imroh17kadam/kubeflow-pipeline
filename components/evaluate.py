from kfp.dsl import component, Input, Output, Dataset, Model, Metrics
from sklearn.metrics import mean_squared_error
import pandas as pd
import joblib

@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def evaluate_model(
    test_data: Input[Dataset],
    model: Input[Model],
    metrics: Output[Metrics]
):
    df = pd.read_csv(test_data.path)

    X = df.drop("target", axis=1)
    y = df["target"]

    model_obj = joblib.load(model.path)
    preds = model_obj.predict(X)

    mse = mean_squared_error(y, preds)
    metrics.log_metric("mse", mse)