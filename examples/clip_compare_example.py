import argparse
import os

from models.clip import run_clip_inference

from evaluations.classification import ClassificationEvaluator
from evaluations.compare import CompareEvaluations
from evaluations.dataloaders import RoboflowDataLoader
from evaluations.dataloaders.classification import \
    ClassificationFolderDataLoader

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

IMAGE_PATH = EVAL_DATA_PATH + "/images/valid"

class_names, ground_truth, model = RoboflowDataLoader(
    workspace_url=ROBOFLOW_WORKSPACE_URL,
    project_url=ROBOFLOW_PROJECT_URL,
    project_version=ROBOFLOW_MODEL_VERSION,
    image_files=EVAL_DATA_PATH,
    model_type="classification",
).download_dataset()

evals = [
    {"classes": ["orange", "background"], "confidence": 1},
    {"classes": ["pear", "background"], "confidence": 0.9},
]

evaluations = []

for cn in evals:
    all_predictions = {}

    for file in ClassificationFolderDataLoader(IMAGE_PATH).get_files():
        all_predictions.update(run_clip_inference(file, class_names))

        evaluations.append(
            ClassificationEvaluator(
                predictions=all_predictions,
                ground_truth=ground_truth,
                confidence_threshold=cn["confidence"],
                class_names=class_names + cn["classes"],
                mode="batch",
            )
        )

best = CompareEvaluations(evaluations)

cf = best.compare()

print(cf)
