from kfp.compiler import Compiler
from pipeline.pipeline import ml_pipeline

Compiler().compile(
    pipeline_func=ml_pipeline,
    package_path="ml_pipeline.yaml"
)