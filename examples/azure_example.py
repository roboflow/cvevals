import argparse
import glob
import os

import cv2
import numpy as np
import requests
import supervision as sv

from evaluations.dataloaders import RoboflowDataLoader
from evaluations.roboflow import RoboflowEvaluator

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
parser.add_argument("--azure_endpoint", type=str, required=True, help="Azure endpoint")
parser.add_argument("--azure_api_key", type=str, required=True, help="Azure API key")

args = parser.parse_args()

EVAL_DATA_PATH = args.eval_data_path
ROBOFLOW_WORKSPACE_URL = args.roboflow_workspace_url
ROBOFLOW_PROJECT_URL = args.roboflow_project_url
ROBOFLOW_MODEL_VERSION = args.roboflow_model_version
AZURE_ENDPOINT = args.azure_endpoint
AZURE_API_KEY = args.azure_api_key

VALIDATION_SET_PATH = EVAL_DATA_PATH + "/valid/images/*.jpg"


# map class names to class ids
class_mappings = {
    "cup": 0,
}


def run_inference_azure(image_filename):
    with open(image_filename, "rb") as image_file:
        data = image_file.read()

    image = cv2.imread(image_filename)

    headers = {
        "Content-Type": "application/octet-stream",
        "Ocp-Apim-Subscription-Key": AZURE_API_KEY,
    }

    response = requests.request("POST", AZURE_ENDPOINT, headers=headers, data=data)

    predictions = response.json()

    if "error" in predictions.keys():
        print(predictions["error"]["message"])
        return sv.detection.core.Detections(
            xyxy=np.array([[0, 0, 0, 0]]),
            class_id=np.array([None]),
            confidence=np.array([100]),
        )

    predictions = predictions["objects"]

    # turn predictions into xyxys
    xyxys = []

    predictions = [
        prediction
        for prediction in predictions
        if prediction["object"] in class_mappings.keys()
    ]

    if len(predictions) == 0:
        return sv.detection.core.Detections(
            xyxy=np.array([[0, 0, 0, 0]]),
            class_id=np.array([None]),
            confidence=np.array([100]),
        )

    for obj in predictions:
        # scale up to full image size
        x0 = obj["rectangle"]["x"] * image.shape[1]
        y0 = obj["rectangle"]["y"] * image.shape[0]
        x1 = (obj["rectangle"]["x"] + obj["rectangle"]["w"]) * image.shape[1]
        y1 = (obj["rectangle"]["y"] + obj["rectangle"]["h"]) * image.shape[0]

        xyxys.append([x0, y0, x1, y1])

    annotations = sv.detection.core.Detections(
        xyxy=np.array(xyxys).astype(np.float32),
        class_id=np.array([class_mappings[obj["object"]] for obj in predictions]),
        confidence=np.array([obj["confidence"] for obj in predictions]),
    )

    return annotations


class_names, data, model = RoboflowDataLoader(
    workspace_url=ROBOFLOW_WORKSPACE_URL,
    project_url=ROBOFLOW_PROJECT_URL,
    project_version=ROBOFLOW_MODEL_VERSION,
    image_files=EVAL_DATA_PATH,
).download_dataset()

all_predictions = {}

for file in glob.glob(VALIDATION_SET_PATH):
    response = run_inference_azure(file)

    all_predictions[file] = {"filename": file, "predictions": response}

print(all_predictions)
print(data)

evaluator = RoboflowEvaluator(
    ground_truth=data,
    predictions=all_predictions,
    class_names=class_names,
    mode="batch",
)

cf = evaluator.eval_model_predictions()

data = evaluator.calculate_statistics()

print("Precision:", data.precision)
print("Recall:", data.recall)
print("f1 Score:", data.f1)
