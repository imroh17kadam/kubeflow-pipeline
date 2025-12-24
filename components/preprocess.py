from kfp.dsl import component, Output, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn"]
)
def preprocess_data(
    input_path: str,
    train_data: Output[Dataset],
    test_data: Output[Dataset]
):
    df = pd.read_csv(input_path)

    X = df.drop("medv", axis=1)
    y = df["medv"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    train_df = pd.DataFrame(X_train)
    train_df["target"] = y_train.values

    test_df = pd.DataFrame(X_test)
    test_df["target"] = y_test.values

    train_df.to_csv(train_data.path, index=False)
    test_df.to_csv(test_data.path, index=False)