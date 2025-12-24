from kfp.dsl import component, Input, Output, Dataset, Model
from sklearn.linear_model import LinearRegression
import pandas as pd
import joblib

@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def train_model(
    train_data: Input[Dataset],
    model: Output[Model]
):
    df = pd.read_csv(train_data.path)

    X = df.drop("target", axis=1)
    y = df["target"]

    model_obj = LinearRegression()
    model_obj.fit(X, y)

    joblib.dump(model_obj, model.path)