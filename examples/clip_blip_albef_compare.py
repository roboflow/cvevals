import argparse

import torch
from models.blip import run_blip_albef_inference
from models.clip import run_clip_inference

from evaluations.classification import ClassificationEvaluator
from evaluations.compare import CompareEvaluations
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

args = parser.parse_args()

EVAL_DATA_PATH = args.eval_data_path
ROBOFLOW_WORKSPACE_URL = args.roboflow_workspace_url
ROBOFLOW_PROJECT_URL = args.roboflow_project_url
ROBOFLOW_MODEL_VERSION = args.roboflow_model_version

IMAGE_PATH = EVAL_DATA_PATH + "/valid"

class_names, ground_truth, model = RoboflowDataLoader(
    workspace_url=ROBOFLOW_WORKSPACE_URL,
    project_url=ROBOFLOW_PROJECT_URL,
    project_version=ROBOFLOW_MODEL_VERSION,
    image_files=EVAL_DATA_PATH,
    model_type="classification",
).download_dataset()

# dedupe class names

class_names = list(set(class_names))

all_clip_predictions = {}

for file in ClassificationFolderDataLoader(IMAGE_PATH).get_files()[1]:
    # print(file)
    all_clip_predictions.update(run_clip_inference(file, class_names))

all_blip_predictions = {}

for file in ClassificationFolderDataLoader(IMAGE_PATH).get_files()[1]:
    all_blip_predictions.update(
        run_blip_albef_inference(file, class_names, "blip", device)
    )

all_albef_predictions = {}

for file in ClassificationFolderDataLoader(IMAGE_PATH).get_files()[1]:
    all_albef_predictions.update(
        run_blip_albef_inference(file, class_names, "albef", device)
    )

evals = [
    {"name": "clip", "predictions": all_clip_predictions},
    {"name": "blip", "predictions": all_blip_predictions},
    {"name": "albef", "predictions": all_albef_predictions},
]

best = CompareEvaluations(
    [
        ClassificationEvaluator(
            predictions=cn["predictions"],
            ground_truth=ground_truth,
            class_names=class_names,
            mode="batch",
            name=cn["name"],
        )
    ]
    for cn in evals
)

cf = best.compare()

print(cf)
