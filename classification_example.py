from evaluations.clip import CLIPEvaluator
from evaluations.dataloaders import CLIPDataLoader, RoboflowDataLoader

class_names, predictions, model = RoboflowDataLoader(
    workspace_url="mit-3xwsm",
    project_url="appling",
    project_version=1,
    image_files="/Users/james/src/clip/model_eval/dataset-new-apples",
    model_type="classification",
).download_dataset()

data = CLIPDataLoader(
    data=predictions,
    class_names=["banana", "apple"],
    eval_data_path="/Users/james/src/clip/model_eval/dataset-new-apples",
).process_files()

evaluator = CLIPEvaluator(data=data, class_names=["banana", "apple"], mode="batch")

cf = evaluator.eval_model_predictions()

precision, recall, f1 = evaluator.calculate_statistics()

print("Precision:", precision)
print("Recall:", recall)
print("f1 Score:", f1)
