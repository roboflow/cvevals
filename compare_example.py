import copy

from evaluations import CompareEvaluations
from evaluations.clip import CLIPEvaluator
from evaluations.dataloaders import (RoboflowDataLoader,
                                     RoboflowPredictionsDataLoader)
from evaluations.dataloaders.cliploader import CLIPDataLoader
from evaluations.roboflow import RoboflowEvaluator

EVAL_DATA_PATH = "/Users/james/src/clip/model_eval/dataset-new/"

class_names, ground_truth, model = RoboflowDataLoader(
    workspace_url="mit-3xwsm",
    project_url="appling",
    project_version=1,
    image_files=EVAL_DATA_PATH,
    model_type="object-detection",
).download_dataset()

evals = [
    {"classes": ["orange", "background"], "confidence": 1},
    {"classes": ["orange", "background"], "confidence": 0.9},
]

best = CompareEvaluations(
    [
        RoboflowEvaluator(
            predictions=RoboflowPredictionsDataLoader(
                model=model,
                model_type="object-detection",
                image_files="/Users/james/src/clip/model_eval/dataset-new/",
                class_names=cn["classes"],
            ).process_files(),
            ground_truth=ground_truth,
            confidence_threshold=cn["confidence"],
            class_names=cn["classes"],
            mode="batch",
        )
        for cn in evals
    ]
)

print(best.compare())
