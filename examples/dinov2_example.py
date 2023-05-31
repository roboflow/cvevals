import argparse
import os

import clip
from evaluations import CompareEvaluations
from evaluations.classification import ClassificationEvaluator
from evaluations.confusion_matrices import plot_confusion_matrix
from evaluations.dataloaders import RoboflowDataLoader

import models.clip as clip
import models.dinov2 as dinov2

parser = argparse.ArgumentParser()

parser.add_argument(
    "--eval_data_path",
    type=str,
    required=True,
    help="Absolute path to Classification dataset (if you don't have one, this is the path in which your Roboflow dataset will be saved)",
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

IMAGE_PATH = EVAL_DATA_PATH + "/test"

class_names, ground_truth, _ = RoboflowDataLoader(
    workspace_url=ROBOFLOW_WORKSPACE_URL,
    project_url=ROBOFLOW_PROJECT_URL,
    project_version=ROBOFLOW_MODEL_VERSION,
    image_files=EVAL_DATA_PATH,
    model_type="classification",
).download_dataset()

labels = {}

dinov2_predictions = {}
clip_predictions = {}

for folder in os.listdir(IMAGE_PATH):
    for file in os.listdir(os.path.join(IMAGE_PATH, folder)):
        if file.endswith(".jpg"):
            full_name = os.path.join(IMAGE_PATH, folder, file)
            labels[full_name] = folder

files = labels.keys()

model = dinov2.train_dinov2_svm_model(IMAGE_PATH)

print(
    "DINOv2 Model Trained. Starting inference (this may take a while depending on how many images you are using)."
)

all_predictions = {}

for file in list(ground_truth.keys())[:3]:
    print("Running inference on", file)
    dinov2_result = dinov2.run_dinov2_inference(model, file, class_names)
    clip_result = clip.run_clip_inference(file, class_names)

    dinov2_predictions[file] = dinov2_result
    clip_predictions[file] = clip_result

print("Running comparison.")

best = CompareEvaluations(
    [
        ClassificationEvaluator(
            ground_truth=ground_truth,
            predictions=dinov2_predictions,
            class_names=class_names,
            mode="batch",
            model_type="multiclass",
            name="DINOv2",
        ),
        ClassificationEvaluator(
            ground_truth=ground_truth,
            predictions=clip_predictions,
            class_names=class_names,
            mode="batch",
            model_type="multiclass",
            name="CLIP",
        ),
    ]
)

highest_eval, comparison = best.compare()

plot_confusion_matrix(highest_eval[4], class_names)

print("F1 Score:", highest_eval[0])
print("Precision:", highest_eval[1])
print("Recall:", highest_eval[2])
print("Class Names:", highest_eval[3])
print("Confidence Threshold:", highest_eval[5])
