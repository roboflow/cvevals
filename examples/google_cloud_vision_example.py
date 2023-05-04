# This script is a WIP

from evaluations.dataloaders import (RoboflowDataLoader)
from evaluations.roboflow import RoboflowEvaluator
import os

from typing import Sequence

from google.cloud import vision


def analyze_image_from_uri(
    image_uri: str,
    feature_types: Sequence,
) -> vision.AnnotateImageResponse:
    client = vision.ImageAnnotatorClient()

    image = vision.Image()
    image.source.image_uri = image_uri
    features = [vision.Feature(type_=feature_type) for feature_type in feature_types]
    request = vision.AnnotateImageRequest(image=image, features=features)

    response = client.annotate_image(request=request)

    return response


def print_objects(response: vision.AnnotateImageResponse):
    print("=" * 80)
    for obj in response.localized_object_annotations:
        nvertices = obj.bounding_poly.normalized_vertices
        print(
            f"{obj.score:4.0%}",
            f"{obj.name:15}",
            f"{obj.mid:10}",
            ",".join(f"({v.x:.1f},{v.y:.1f})" for v in nvertices),
            sep=" | ",
        )
        

class_names, data, model = RoboflowDataLoader(
    workspace_url="james-gallagher-87fuq",
    project_url="mug-detector-eocwp",
    project_version=12,
    image_files="/Users/james/src/clip/model_eval/dataset-new",
).download_dataset()

all_predictions = {}

for file in os.listdir("/Users/james/src/clip/model_eval/dataset-new/valid"):
    # add base dir
    current_dir = os.getcwd()
    file = os.path.join(current_dir, file)

    response = analyze_image_from_uri(file, [vision.Feature.Type.OBJECT_LOCALIZATION])
    # print_labels(response)

    print(response)

    all_predictions[file] = {"filename": file, "predictions": data}

evaluator = RoboflowEvaluator(
    ground_truth=data, predictions=all_predictions, class_names=class_names, mode="batch"
)

cf = evaluator.eval_model_predictions()

data = evaluator.calculate_statistics()

print("Precision:", data.precision)
print("Recall:", data.recall)
print("f1 Score:", data.f1)
