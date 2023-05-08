import argparse

import torch
from models.blip import run_blip_albef_inference

from evaluations.classification import ClassificationEvaluator
from evaluations.dataloaders import RoboflowDataLoader
from evaluations.dataloaders.classification import \
    ClassificationFolderDataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

# translate above to argparse
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
parser.add_argument(
    "--feature_extractor",
    type=str,
    required=True,
    help="Feature extractor to use ('blip' or 'albef')",
)

args = parser.parse_args()

EVAL_DATA_PATH = args.eval_data_path
ROBOFLOW_WORKSPACE_URL = args.roboflow_workspace_url
ROBOFLOW_PROJECT_URL = args.roboflow_project_url
ROBOFLOW_MODEL_VERSION = args.roboflow_model_version
FEATURE_EXTRACTOR = args.feature_extractor

if FEATURE_EXTRACTOR not in ("blip", "albef"):
    raise ValueError("feature_extractor must be either 'blip' or 'albef'")

# use validation set
IMAGE_PATH = EVAL_DATA_PATH + "/images/valid"

class_names, ground_truth, model = RoboflowDataLoader(
    workspace_url=ROBOFLOW_WORKSPACE_URL,
    project_url=ROBOFLOW_PROJECT_URL,
    project_version=ROBOFLOW_MODEL_VERSION,
    image_files=EVAL_DATA_PATH,
    model_type="classification",
).download_dataset()

all_predictions = {}

for file in ClassificationFolderDataLoader(IMAGE_PATH).get_files()[1]:
    all_predictions.update(
        run_blip_albef_inference(file, class_names, FEATURE_EXTRACTOR, device)
    )

evaluator = ClassificationEvaluator(
    ground_truth=ground_truth,
    predictions=all_predictions,
    class_names=["banana", "apple"],
    mode="batch",
)

cf = evaluator.eval_model_predictions()

data = evaluator.calculate_statistics()

print("Precision:", data.precision)
print("Recall:", data.recall)
print("f1 Score:", data.f1)
