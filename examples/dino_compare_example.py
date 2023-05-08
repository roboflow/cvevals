import argparse
import os

import cv2
from groundingdino.util.inference import Model

from evaluations import CompareEvaluations, Evaluator
from evaluations.confusion_matrices import plot_confusion_matrix
from evaluations.dataloaders import RoboflowDataLoader

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
    "--config_path", type=str, required=True, help="Path to GroundingDINO config"
)
parser.add_argument(
    "--weights_path", type=str, required=True, help="Path to GroundingDINO weights"
)
parser.add_argument("--box_threshold", type=float, required=False, help="Box threshold")
parser.add_argument(
    "--text_threshold", type=float, required=False, help="Text threshold"
)

args = parser.parse_args()

if not args.box_threshold:
    BOX_THRESHOLD = 0.35

if not args.text_threshold:
    TEXT_THRESHOLD = 0.25

EVAL_DATA_PATH = args.eval_data_path
ROBOFLOW_WORKSPACE_URL = args.roboflow_workspace_url
ROBOFLOW_PROJECT_URL = args.roboflow_project_url
ROBOFLOW_MODEL_VERSION = args.roboflow_model_version
CONFIG_PATH = args.config_path
WEIGHTS_PATH = args.weights_path

# use validation set
IMAGE_PATH = EVAL_DATA_PATH + "/valid/images"

class_names, ground_truth, model = RoboflowDataLoader(
    workspace_url=ROBOFLOW_WORKSPACE_URL,
    project_url=ROBOFLOW_PROJECT_URL,
    project_version=ROBOFLOW_MODEL_VERSION,
    image_files=EVAL_DATA_PATH,
).download_dataset()

model = Model(
    model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH, device="cpu"
)


def get_dino_predictions(class_names):
    all_predictions = {}

    for file in os.listdir(IMAGE_PATH)[:2]:
        file = os.path.join(IMAGE_PATH, file)
        if file.endswith(".jpg"):
            image = cv2.imread(file)

            detections = model.predict_with_classes(
                image=image,
                classes=class_names,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
            )

            all_predictions[file] = {"filename": file, "predictions": detections}

    return all_predictions


evals = [
    {"classes": ["banana", "background"], "confidence": 0.5},
    {"classes": ["coffee cup", "background"], "confidence": 0.7},
    {"classes": ["coffee cup", "background"], "confidence": 0.9},
    {"classes": ["blue coffee cup", "background"], "confidence": 0.9},
]

best = CompareEvaluations(
    [
        Evaluator(
            predictions=get_dino_predictions(class_names=cn["classes"]),
            ground_truth=ground_truth,
            confidence_threshold=cn["confidence"],
            class_names=cn["classes"],
            mode="batch",
        )
        for cn in evals
    ]
)

comparison = best.compare()

plot_confusion_matrix(comparison[4], class_names=comparison[3], mode="interactive")

print("F1 Score:", comparison[0])
print("Precision:", comparison[1])
print("Recall:", comparison[2])
print("Class Names:", comparison[3])
print("Confidence Threshold:", comparison[5])
