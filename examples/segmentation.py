import argparse
import os
import supervision as sv

from evaluations.dataloaders import (RoboflowDataLoader,
                                     RoboflowPredictionsDataLoader)
from evaluations.segmentation import SegmentationEvaluator

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

class_names, _, model = RoboflowDataLoader(
    workspace_url=ROBOFLOW_WORKSPACE_URL,
    project_url=ROBOFLOW_PROJECT_URL,
    project_version=ROBOFLOW_MODEL_VERSION,
    image_files=EVAL_DATA_PATH,
    model_type="segmentation",
).download_dataset()

ds = sv.DetectionDataset.from_yolo(
    images_directory_path=os.path.join(EVAL_DATA_PATH, "test/images"),
    annotations_directory_path=os.path.join(EVAL_DATA_PATH, "test/labels"),
    data_yaml_path=os.path.join(EVAL_DATA_PATH, "data.yaml"),
)

predictions = RoboflowPredictionsDataLoader(
    model=model,
    model_type="segmentation",
    image_files=EVAL_DATA_PATH,
    class_names=class_names,
).process_files()

# add EVAL_DATA_PATH to predictions

evaluator = SegmentationEvaluator(
    ground_truth=ds, predictions=predictions, class_names=class_names, eval_data_path = EVAL_DATA_PATH
)

cf = evaluator.eval_model_predictions()

from evaluations.confusion_matrices import plot_confusion_matrix

plot_confusion_matrix(
    cf,
    class_names=class_names,
    aggregate=True,
)


data = evaluator.calculate_statistics()

print("Precision:", data.precision)
print("Recall:", data.recall)
print("f1 Score:", data.f1)
