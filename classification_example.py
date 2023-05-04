import os

import clip
import torch
from PIL import Image

from evaluations.classification import ClassificationEvaluator
from evaluations.dataloaders import RoboflowDataLoader
from evaluations.dataloaders.classification import (
    ClassificationDetections, ClassificationFolderDataLoader)

class_names, ground_truth, model = RoboflowDataLoader(
    workspace_url="mit-3xwsm",
    project_url="appling",
    project_version=1,
    image_files="/Users/james/src/model_eval/dataset-new-apples",
    model_type="classification",
).download_dataset()

all_predictions = {}

IMAGE_PATH = "dataset-new-apples/valid"

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

for file in ClassificationFolderDataLoader(IMAGE_PATH).get_files()[1]:
    # add base dir
    current_dir = os.getcwd()
    file = os.path.join(current_dir, file)

    image = preprocess(Image.open(file)).unsqueeze(0).to(device)
    text = clip.tokenize(class_names).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    values, indices = similarity[0].topk(1)

    data = ClassificationDetections(
        image_id=file,
        predicted_class_names=[class_names[indices[0]]],
        predicted_class_ids=[indices[0]],
        confidence=values[0],
    )

    all_predictions[file] = {"filename": file, "predictions": data}

evaluator = ClassificationEvaluator(
    ground_truth=ground_truth,
    predictions=all_predictions,
    class_names=["banana", "apple"],
    mode="batch",
)

cf = evaluator.eval_model_predictions()

data = evaluator.calculate_statistics()

print("Precision:", data.precision)
print("Recall:", data.recall)
print("f1 Score:", data.f1)
