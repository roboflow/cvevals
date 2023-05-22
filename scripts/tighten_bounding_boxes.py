import argparse
import os
import json

from evaluations.dataloaders import RoboflowDataLoader

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
parser.add_argument(
    "--increase_bounding_box_size",
    type=float,
    required=False,
    default=0.1,
    help="Increase bounding box size by this percentage",
)

parser.add_argument(
    "--sam_checkpoint",
    type=str,
    required=True,
    help="Path to SAM checkpoint",
)

args = parser.parse_args()

EVAL_DATA_PATH = args.eval_data_path
ROBOFLOW_WORKSPACE_URL = args.roboflow_workspace_url
ROBOFLOW_PROJECT_URL = args.roboflow_project_url
ROBOFLOW_MODEL_VERSION = args.roboflow_model_version

if args.increase_bounding_box_size > 1.0:
    raise ValueError("Increase bounding box size must be less than 1.0")

class_names, data, model = RoboflowDataLoader(
    workspace_url=ROBOFLOW_WORKSPACE_URL,
    project_url=ROBOFLOW_PROJECT_URL,
    project_version=ROBOFLOW_MODEL_VERSION,
    image_files=EVAL_DATA_PATH,
    model_type="object-detection",
).download_dataset()

images = data.keys()

import cv2
import supervision as sv
from segment_anything import SamPredictor, sam_model_registry

from evaluations.iou import box_iou

sam = sam_model_registry["default"](
    checkpoint=SAM_CHECKPOINT
)
import numpy as np

recommended_boxes = {}

predictor = SamPredictor(sam)

for i in images:
    print(i)
    bboxes = data[i]["ground_truth"]

    img = cv2.imread(i)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    predictor.set_image(img_rgb)

    input_box = bboxes[0]

    annotator = sv.BoxAnnotator()

    input_box = list(map(int, input_box[:4]))

    # cast box as 2d np.ndarray
    box = np.array([input_box])

    box = list(map(int, input_box[:4]))

    # make box bigger
    box[0] = box[0] - int(box[2] * args.increase_bounding_box_size)
    box[1] = box[1] - int(box[3] * args.increase_bounding_box_size)
    box[2] = box[2] + int(box[2] * args.increase_bounding_box_size)
    box[3] = box[3] + int(box[3] * args.increase_bounding_box_size)

    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.array(box[:4]),
        multimask_output=False,
    )

    box_annotator = sv.BoxAnnotator(color=sv.Color.red())
    mask_annotator = sv.MaskAnnotator(color=sv.Color.red())

    detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=masks), mask=masks)

    detections = detections[detections.area == np.max(detections.area)]

    recommended_box = detections.xyxy[0].tolist()
    recommended_boxes[i] = {}
    recommended_boxes[i]["rec"] = recommended_box
    recommended_boxes[i]["gt"] = input_box
    recommended_boxes[i]["iou"] = box_iou(input_box, recommended_box)

    annotator = sv.BoxAnnotator()

    # merge two detections
    detections = sv.Detections(
        xyxy=np.concatenate([np.array([input_box]), detections.xyxy]),
    )

    annotated_image = annotator.annotate(
        img,
        detections=detections,
    )

    sv.plot_image(annotated_image)


with open("recommended_boxes.json", "w") as f:
    json.dump(recommended_boxes, f)

# calculate avg iou
ious = [v["iou"] for v in recommended_boxes.values()]
print(np.mean(ious))
