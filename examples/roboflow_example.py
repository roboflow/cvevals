from evaluations.dataloaders import (RoboflowDataLoader,
                                     RoboflowPredictionsDataLoader)
from evaluations.roboflow import RoboflowEvaluator

ROBOFLOW_WORKSPACE_URL = "james-gallagher-87fuq"
ROBOFLOW_PROJECT_URL = "retail-shelf-detection"
EVAL_DATA_PATH = "/Users/james/src/clip/model_eval/dataset-retail"
ROBOFLOW_MODEL_VERSION = 1

class_names, data, model = RoboflowDataLoader(
    workspace_url=ROBOFLOW_WORKSPACE_URL,
    project_url=ROBOFLOW_PROJECT_URL,
    project_version=ROBOFLOW_MODEL_VERSION,
    image_files=EVAL_DATA_PATH
).download_dataset()

predictions = RoboflowPredictionsDataLoader(
    model=model,
    model_type="object-detection",
    image_files=EVAL_DATA_PATH,
    class_names=class_names,
).process_files()

evaluator = RoboflowEvaluator(
    ground_truth=data, predictions=predictions, class_names=class_names, mode="batch"
)

cf = evaluator.eval_model_predictions()

print(cf)

data = evaluator.calculate_statistics()

print("Precision:", data.precision)
print("Recall:", data.recall)
print("f1 Score:", data.f1)
