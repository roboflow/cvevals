import argparse
import os

from evaluations.dataloaders import (RoboflowDataLoader,
                                     RoboflowPredictionsDataLoader)
from evaluations.roboflow import RoboflowEvaluator

parser = argparse.ArgumentParser()

parser.add_argument(
    "--eval_data_path",
    type=str,
    required=True,
    help="Absolute path to YOLOv5 PyTorch TXT or Classification dataset",
)
parser.add_argument(
    "--roboflow_workspace_url", type=str, required=True, help="Roboflow workspace ID"
)
parser.add_argument(
    "--roboflow_project_url", type=str, required=True, help="Roboflow project ID"
)
parser.add_argument(
    "--roboflow_model_version", type=int, required=True, help="Roboflow model version"
)

args = parser.parse_args()

EVAL_DATA_PATH = args.eval_data_path
ROBOFLOW_WORKSPACE_URL = args.roboflow_workspace_url
ROBOFLOW_PROJECT_URL = args.roboflow_project_url
ROBOFLOW_MODEL_VERSION = args.roboflow_model_version

class_names, data, model = RoboflowDataLoader(
    workspace_url=ROBOFLOW_WORKSPACE_URL,
    project_url=ROBOFLOW_PROJECT_URL,
    project_version=ROBOFLOW_MODEL_VERSION,
    image_files=EVAL_DATA_PATH,
).download_dataset()

predictions = RoboflowPredictionsDataLoader(
    model=model,
    model_type="object-detection",
    image_files=EVAL_DATA_PATH,
    class_names=class_names,
).process_files()

evaluator = RoboflowEvaluator(
    ground_truth=data, predictions=predictions, class_names=class_names, mode="batch"
)

cf = evaluator.eval_model_predictions()

print(cf)

data = evaluator.calculate_statistics()

print("Precision:", data.precision)
print("Recall:", data.recall)
print("f1 Score:", data.f1)
