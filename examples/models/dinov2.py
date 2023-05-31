import json
import os

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn import svm
from tqdm import tqdm

from evaluations.dataloaders.classification import ClassificationDetections

cwd = os.getcwd()

ROOT_DIR = os.path.join(cwd, "MIT-Indoor-Scene-Recognition-5/train")

dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dinov2_vits14.to(device)

transform_image = T.Compose(
    [T.ToTensor(), T.Resize(244), T.CenterCrop(224), T.Normalize([0.5], [0.5])],
)


def load_image(img: str) -> torch.Tensor:
    """
    Load an image and return a tensor that can be used as an input to DINOv2.
    """
    img = Image.open(img)

    transformed_img = transform_image(img)[:3].unsqueeze(0)

    return transformed_img


def embed_image(img: str) -> torch.Tensor:
    """
    Embed an image using DINOv2.
    """
    with torch.no_grad():
        return dinov2_vits14(load_image(img).to(device))


def compute_embeddings(files: list) -> dict:
    """
    Create an index that contains all of the images in the specified list of files.
    """
    all_embeddings = {}

    with torch.no_grad():
        for i, file in enumerate(tqdm(files)):
            embeddings = dinov2_vits14(load_image(file).to(device))

            all_embeddings[file] = (
                np.array(embeddings[0].cpu().numpy()).reshape(1, -1).tolist()
            )

    with open("all_embeddings.json", "w") as f:
        f.write(json.dumps(all_embeddings))

    return all_embeddings

def train_dinov2_svm_model(training_dir):
    """
    Train an SVM model using the embeddings from DINOv2.
    """
    labels = {}

    for folder in os.listdir(training_dir):
        for file in os.listdir(os.path.join(training_dir, folder)):
            if file.endswith(".jpg"):
                full_name = os.path.join(training_dir, folder, file)
                labels[full_name] = folder

    files = labels.keys()

    embeddings = compute_embeddings(files)

    clf = svm.SVC(gamma="scale")

    y = [labels[file] for file in files]

    embedding_list = list(embeddings.values())

    clf.fit(np.array(embedding_list).reshape(-1, 384), y)

    return clf

def run_dinov2_inference(model, image: str, class_names: list) -> dict:
    """
    Run inference on a single image using the DINOv2 model.
    """

    result = model.predict(embed_image(image))

    data = ClassificationDetections(
        image_id=image,
        predicted_class_names=result,
        predicted_class_ids=[class_names.index(result[0])],
        confidence=1.0,
    )

    return {"filename": image, "predictions": data}