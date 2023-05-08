import argparse
import base64
import glob
import os
from typing import Sequence

import cv2
import numpy as np
import supervision as sv
from google.cloud import vision

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

args = parser.parse_args()

EVAL_DATA_PATH = args.eval_data_path
ROBOFLOW_WORKSPACE_URL = args.roboflow_workspace_url
ROBOFLOW_PROJECT_URL = args.roboflow_project_url
ROBOFLOW_MODEL_VERSION = args.roboflow_model_version

VALIDATION_SET_PATH = EVAL_DATA_PATH + "/valid/images/*.jpg"

# map class names to class ids
class_mappings = {
    "Mug": 0,
}


def analyze_image_from_uri(
    image_uri: str,
    feature_types: Sequence,
) -> vision.AnnotateImageResponse:
    client = vision.ImageAnnotatorClient()

    image = vision.Image()

    loaded_image = cv2.imread(image_uri)

    base64_image = base64.b64encode(open(image_uri, "rb").read())

    image.source.image_uri = "base64://" + base64_image.decode()

    features = [vision.Feature(type_=feature_type) for feature_type in feature_types]

    request = {
        "image": {"content": base64_image.decode()},
        "features": features,
    }

    response = client.annotate_image(request)

    localized_object_annotations = response.localized_object_annotations

    localized_object_annotations = [
        annotation
        for annotation in localized_object_annotations
        if annotation.name in class_mappings.keys()
    ]

    if len(localized_object_annotations) == 0:
        return sv.detection.core.Detections(
            xyxy=np.array([[0, 0, 0, 0]]),
            class_id=np.array([None]),
            confidence=np.array([100]),
        )

    xyxys = []

    image_size = loaded_image.shape

    for obj in localized_object_annotations:
        # scale up to full image size
        x0 = obj.bounding_poly.normalized_vertices[0].x * image_size[1]
        y0 = obj.bounding_poly.normalized_vertices[0].y * image_size[0]
        x1 = obj.bounding_poly.normalized_vertices[2].x * image_size[1]
        y1 = obj.bounding_poly.normalized_vertices[2].y * image_size[0]

        xyxys.append([x0, y0, x1, y1])

    annotations = sv.detection.core.Detections(
        xyxy=np.array(xyxys).astype(np.float32),
        class_id=np.array(
            [class_mappings[obj.name] for obj in localized_object_annotations]
        ),
        confidence=np.array([obj.score for obj in localized_object_annotations]),
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
    # add base dir
    current_dir = os.getcwd()

    response = analyze_image_from_uri(file, [vision.Feature.Type.OBJECT_LOCALIZATION])

    all_predictions[file] = {"filename": file, "predictions": response}

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
