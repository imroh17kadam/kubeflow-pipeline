from kfp.dsl import pipeline
from components.preprocess import preprocess_data
from components.train import train_model
from components.evaluate import evaluate_model

@pipeline(
    name="basic-ml-pipeline",
    description="End-to-end ML pipeline with preprocessing, training and evaluation"
)
def ml_pipeline(input_data_path: str):
    preprocess_task = preprocess_data(
        input_path=input_data_path
    )

    train_task = train_model(
        train_data=preprocess_task.outputs["train_data"]
    )

    evaluate_model(
        test_data=preprocess_task.outputs["test_data"],
        model=train_task.outputs["model"]
    )