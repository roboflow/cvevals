import os

from lavis.models import load_model_and_preprocess
from PIL import Image

from evaluations.dataloaders.classification import ClassificationDetections


def run_blipv2_inference(file: str, class_names: list, device: str) -> dict:
    """
    Run inference on a single image using the BLIPv2 model by Salesforce.

    BLIPv2 doesn't support classification out of the box, so the image question-answering features are leveraged.

    The model is asked "What does this image primarily contain: <class_names>?"

    The result is returned with a confidence of 1. This is because the model does not return a confidence score.

    Thus, you cannot evaluate different confidence thresholds on BLIPv2.

    Args:

        file (str): Path to the image file.
        class_names (list): List of class names.
        device (str): Device to run the model on.
    
    Returns:
        dict: Dictionary containing the filename and the predictions.
    """
    # add base dir
    current_dir = os.getcwd()
    file = os.path.join(current_dir, file)

    raw_image = Image.open("./IMG_3322.jpeg").convert("RGB")

    model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device)
    

    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    prompt = "What does this image primarily contain: " + ", ".join(class_names).rstrip(",") + "?"

    response = model.generate({
        "image": image,
        "prompt": prompt
    })

    data = ClassificationDetections(
        image_id=file,
        predicted_class_names=response,
        predicted_class_ids=[class_names.index(response)],
        confidence=1,
    )


    return {"filename": file, "predictions": [prompt]}
