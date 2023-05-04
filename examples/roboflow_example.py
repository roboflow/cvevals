from evaluations.dataloaders import (RoboflowDataLoader, RoboflowPredictionsDataLoader)
from evaluations.roboflow import RoboflowEvaluator

class_names, data, model = RoboflowDataLoader(
    workspace_url="james-gallagher-87fuq",
    project_url="mug-detector-eocwp",
    project_version=12,
    image_files="/Users/james/src/clip/model_eval/dataset-new",
).download_dataset()

predictions = RoboflowPredictionsDataLoader(
    model=model,
    model_type="object-detection",
    image_files="/Users/james/src/clip/model_eval/dataset-new/",
    class_names=class_names,
).process_files()

evaluator = RoboflowEvaluator(
    ground_truth=data, predictions=predictions, class_names=class_names, mode="batch"
)

cf = evaluator.eval_model_predictions()

data = evaluator.calculate_statistics()

print("Precision:", data.precision)
print("Recall:", data.recall)
print("f1 Score:", data.f1)
