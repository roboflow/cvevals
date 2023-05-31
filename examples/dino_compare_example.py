import argparse
import os
import numpy as np

import cv2
from groundingdino.util.inference import Model

import supervision as sv

from evaluations import CompareEvaluations, Evaluator
from evaluations.dataloaders import RoboflowDataLoader

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
    model_type="object-detection"
).download_dataset()

model = Model(
    model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH, device="cpu"
)


def get_dino_predictions(class_names: dict) -> dict:
    all_predictions = {}

    class_names = [pred["inference"] for pred in class_names]

    for file in os.listdir(IMAGE_PATH):
        file = os.path.join(IMAGE_PATH, file)
        if file.endswith(".jpg"):
            image = cv2.imread(file)

            detections = model.predict_with_classes(
                image=image,
                classes=class_names,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
            )

            annotations = sv.detection.core.Detections(
                xyxy=np.array(detections.xyxy).astype(np.float32),
                class_id=np.array(detections.class_id),
                confidence=np.array(detections.confidence)
            )

            all_predictions[file] = {"filename": file, "predictions": annotations}

    return all_predictions

evals = [
    {"classes": [{"ground_truth": "", "inference": ""}], "confidence": 0.5},
]

# map eval inferences to ground truth
for e in evals:
    new_ground_truth = ground_truth.copy()

    for key in new_ground_truth.keys():
        result = list(new_ground_truth[key]["ground_truth"])

        for idx, item in enumerate(result):
            if item[4] == class_names.index(e["classes"][0]["ground_truth"]):
                item = list(item)
                item[4] = 0
                result[idx] = tuple(item)

        result = [item for item in result if item[4] == 0]

        new_ground_truth[key]["ground_truth"] = result

    evals[evals.index(e)]["ground_truth"] = new_ground_truth.copy()
    evals[evals.index(e)]["inference_class_names"] = [e["classes"][0]["inference"], "background"]

best = CompareEvaluations(
    [
        Evaluator(
            predictions=get_dino_predictions(class_names=cn["classes"]),
            ground_truth=cn["ground_truth"],
            confidence_threshold=cn["confidence"],
            class_names=cn["inference_class_names"],
            name=cn["classes"][0]["inference"],
            mode="batch",
            model_type="object-detection"
        )
        for cn in evals
    ]
)

comparison = best.compare()

print("F1 Score:", comparison[0])
print("Precision:", comparison[1])
print("Recall:", comparison[2])
print("Class Names:", comparison[3])
print("Confidence Threshold:", comparison[5])
