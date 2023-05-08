import os

import torch
from lavis.models import load_model_and_preprocess
from PIL import Image

from evaluations.dataloaders.classification import ClassificationDetections


def run_blip_albef_inference(file, class_names, model_type, device):
    # add base dir
    current_dir = os.getcwd()
    file = os.path.join(current_dir, file)

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
