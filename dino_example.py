from evaluations import Evaluator
from evaluations.dataloaders import (RoboflowDataLoader,
                                     RoboflowPredictionsDataLoader)
from evaluations.dataloaders.dino import DINODataLoader

class_names, data, model = RoboflowDataLoader(
    workspace_url="james-gallagher-87fuq",
    project_url="mug-detector-eocwp",
    project_version=12,
    image_files="/Users/james/src/clip/model_eval/dataset-new",
).download_dataset()

predictions = DINODataLoader(
    data=data,
    class_names=["flames", "background"],
    mode="batch",
    image_files="/Users/james/src/clip/model_eval/dataset-new",
).process_files()

merged_data = {}

for key, value in predictions.items():
    gt = data.get(key, [])

    merged_data[key] = {
        "ground_truth": gt["ground_truth"],
        "predictions": predictions[key]["predictions"],
        "filename": key,
    }

evaluator = Evaluator(
    data=merged_data, class_names=class_names, confidence_threshold=0.2, mode="batch"
)

cf = evaluator.eval_model_predictions()

precision, recall, f1 = evaluator.calculate_statistics()

print("Precision:", precision)
print("Recall:", recall)
print("f1 Score:", f1)
