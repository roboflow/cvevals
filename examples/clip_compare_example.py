import os

import clip
import torch
from PIL import Image

from evaluations.classification import ClassificationEvaluator
from evaluations.compare import CompareEvaluations
from evaluations.dataloaders import RoboflowDataLoader
from evaluations.dataloaders.classification import (
    ClassificationDetections, ClassificationFolderDataLoader)

# use absolute path
EVAL_DATA_PATH = ""
ROBOFLOW_WORKSPACE_URL = ""
ROBOFLOW_PROJECT_URL = ""
ROBOFLOW_MODEL_VERSION = 1

IMAGE_PATH = EVAL_DATA_PATH + "/images/valid"

class_names, ground_truth, model = RoboflowDataLoader(
    workspace_url=ROBOFLOW_WORKSPACE_URL,
    project_url=ROBOFLOW_PROJECT_URL,
    project_version=ROBOFLOW_MODEL_VERSION,
    image_files=EVAL_DATA_PATH,
    model_type="classification",
).download_dataset()


def get_clip_predictions(class_names):
    all_predictions = {}

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

    return all_predictions


evals = [
    {"classes": ["orange", "background"], "confidence": 1},
    {"classes": ["pear", "background"], "confidence": 0.9},
]

best = CompareEvaluations(
    [
        ClassificationEvaluator(
            predictions=get_clip_predictions(cn["classes"]),
            ground_truth=ground_truth,
            confidence_threshold=cn["confidence"],
            class_names=class_names + cn["classes"],
            mode="batch",
        )
        for cn in evals
    ]
)

cf = best.compare()

print(cf)
