import data
import torch
from models import imagebind_model
from models.imagebind_model import ModalityType

import argparse
import os

import clip
import torch
from PIL import Image

from evaluations.classification import ClassificationEvaluator
from evaluations.dataloaders import RoboflowDataLoader
from evaluations.dataloaders.classification import ClassificationDetections

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

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

# use validation set
IMAGE_PATH = EVAL_DATA_PATH + "/valid"

class_names, ground_truth, model = RoboflowDataLoader(
    workspace_url=ROBOFLOW_WORKSPACE_URL,
    project_url=ROBOFLOW_PROJECT_URL,
    project_version=ROBOFLOW_MODEL_VERSION,
    image_files=EVAL_DATA_PATH,
    model_type="multiclass",
).download_dataset()

all_predictions = {}

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)


def run_imagebind_inference(class_names: str, img_filename: str) -> str:
    """
    Run inference on an image using ImageBind.

    Args:
        class_names: The class names to use for inference.
        img_filename: The filename of the image on which to run inference.

    Returns:
        The predicted class name.
    """
    image_paths=[".assets/dog_image.jpg"]

    # Load data
    inputs = {
        ModalityType.TEXT: data.load_and_transform_text(class_names, device),
        ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
    }

    with torch.no_grad():
        embeddings = model(inputs)

    text_result = torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1)

    top_k = torch.topk(text_result, k=1, dim=-1)
    confidence = top_k.values[0][0].item()

    return class_names[top_k.indices[0][0]], confidence


for file in ground_truth.keys():
    # add base dir
    current_dir = os.getcwd()
    file = os.path.join(current_dir, file)

    # run inference
    class_name, confidence = run_imagebind_inference(class_names, file)

    result = ClassificationDetections(
        image_id=file,
        predicted_class_names=[class_name],
        predicted_class_ids=[class_names.index(class_name)],
        confidence=[1.0, confidence],
    )

    all_predictions[file] = {"filename": file, "predictions": result}

evaluator = ClassificationEvaluator(
    ground_truth=ground_truth,
    predictions=all_predictions,
    class_names=class_names,
    mode="batch",
    model_type="multiclass",
)

cf = evaluator.eval_model_predictions()

data = evaluator.calculate_statistics()

print("Precision:", data.precision)
print("Recall:", data.recall)
print("f1 Score:", data.f1)
