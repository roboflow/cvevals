import os

import torch
from lavis.models import load_model_and_preprocess
from PIL import Image

from evaluations.dataloaders.classification import ClassificationDetections

SUPPORTED_MODELS = ("blip", "albef")

def run_blip_albef_inference(file: str, class_names: list, model_type: str, device: str) -> dict:
    """
    Run inference on a single image using the BLIP or ALBEF model by Salesforce.

    Args:

        file (str): Path to the image file.
        class_names (list): List of class names.
        model_type (str): Model type to use. Either "blip" or "albef".
        device (str): Device to run the model on.

    Returns:

        dict: Dictionary containing the filename and the predictions.
    """
    current_dir = os.getcwd()
    file = os.path.join(current_dir, file)

    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"Model type {model_type} is not supported by the run_blip_albef_inference function. Supported models are {SUPPORTED_MODELS}")

    raw_image = Image.open("./IMG_3322.jpeg").convert("RGB")

    classes = ["A picture of " + c for c in class_names]

    model, vis_processors, _ = load_model_and_preprocess(
        name=model_type + "_classification",
        model_type="base",
        is_eval=True,
        device=device,
    )

    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    sample = {"image": image, "text_input": classes}

    image_features = model.extract_features(sample, mode="image").image_embeds_proj[
        :, 0
    ]
    text_features = model.extract_features(sample, mode="text").text_embeds_proj[:, 0]

    sims = (image_features @ text_features.t())[0] / model.temp
    probs = torch.nn.Softmax(dim=0)(sims).tolist()

    for cls_nm, prob in zip(classes, probs):
        print(f"{cls_nm}: \t {prob:.3%}")

    top_prob = None
    top_prob_confidence = -1

    for cls_nm, prob in zip(classes, probs):
        if prob > top_prob_confidence:
            top_prob = cls_nm
            top_prob_confidence = prob

    top_prob = top_prob.lstrip("A picture of ")

    data = ClassificationDetections(
        image_id=file,
        predicted_class_names=[class_names[top_prob[0]]],
        predicted_class_ids=[top_prob[0]],
        confidence=top_prob_confidence[0],
    )

    return {"filename": file, "predictions": data}
