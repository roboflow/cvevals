import cv2
import torch
from groundingdino.util.inference import load_image, load_model
from groundingdino.util.utils import get_phrases_from_posmap

from .evaluator import Evaluator

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_grounding_output(
    model,
    image,
    caption: str,
    box_threshold: int,
    text_threshold: int,
    with_logits: bool = True,
):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."

    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(
            logit > text_threshold, tokenized, tokenlizer
        )
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


class DINOEvaluator(Evaluator):
    """
    Evaluate Grounding DINO prompts.
    """

    def __init__(
        self,
        data: dict = {},
        class_names: list = [],
        mode: str = "batch",
        confidence_threshold: int = 0.2,
        box_threshold: int = 0.35,
        text_threshold: int = 0.25,
        config_dir: str = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        weights_dir: str = "GroundingDINO/weights/groundingdino_swint_ogc.pth",
    ):
        self.data = data
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.mode = mode

        self.text_prompt = " . ".join(self.class_names)

        self.main_model = load_model(config_dir, weights_dir, device=device)

    def get_predictions_for_image(self, img_filename: str) -> list:
        """
        Retrieve predictions for a Roboflow, Grounding DINO, and CLIP model for a single image.

        Args:
            img_filename (str): Path to image file

        Returns:
            predictions (list): List of predictions
        """

        _, image = load_image(img_filename)

        image = cv2.imread(img_filename)

        _, img = load_image(img_filename)

        all_boxes, pred_phrases = get_grounding_output(
            self.main_model,
            img,
            self.text_prompt,
            self.box_threshold,
            self.text_threshold,
            with_logits=False,
        )

        H = image.shape[0]
        W = image.shape[1]

        results = []

        for box in all_boxes:
            box = box * torch.Tensor([W, H, W, H])
            # from xywh to xyxy
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]

            x0, y0, x1, y1 = box
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

            results.append((x0, y0, x1, y1))

        predictions = []

        for item, p in zip(results, pred_phrases):
            # get text before ()
            p = p.split("(")[0].strip()

            if len(item) == 0:
                continue

            try:
                index = self.class_names.index(p)
            except:
                index = 0

            predictions.append((item[0], item[1], item[2], item[3], index, 50))

        return predictions
