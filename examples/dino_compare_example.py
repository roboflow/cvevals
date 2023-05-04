import os

import cv2
from groundingdino.util.inference import Model

from evaluations import Evaluator, CompareEvaluations
from evaluations.dataloaders import RoboflowDataLoader

from evaluations.confusion_matrices import plot_confusion_matrix

DIRECTORY = "/Users/james/src/clip/model_eval/dataset-new"

class_names, ground_truth, model = RoboflowDataLoader(
    workspace_url="james-gallagher-87fuq",
    project_url="mug-detector-eocwp",
    project_version=12,
    image_files=DIRECTORY,
).download_dataset()

IMAGE_PATH = DIRECTORY + "/valid/images"

CONFIG_PATH: str = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH: str = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

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

            all_predictions[file] = {"filename": file, "predictions": detections}#normalize_dino(image, detections)}

    return all_predictions

evals = [
    {"classes": ["banana", "background"], "confidence": 0.5},
    {"classes": ["coffee cup", "background"], "confidence": 1.2},
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