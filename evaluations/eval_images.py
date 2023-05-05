import cv2
import numpy as np

# from supervision.metrics.iou import box_iou
from .iou import box_iou


def find_best_prediction(
    gt_box: list, predictions: list, iou_threshold: int = 0.5
) -> tuple:
    """
    Given ground truth data and associated predictions, find the prediction with the highest IOU.

    Args:
        gt_box (list): Ground truth box coordinates.
        predictions (list): A list of model predictions.
        iou_threshold (list): The Intersection Over Union threshold above which the ground truth and top prediction must be to be returned.
    """
    if len(predictions) == 0:
        return None, None

    ious = [box_iou(gt_box, pred_box) for pred_box in predictions]

    selected_pred_idx = np.argmax(ious)
    selected_pred = predictions[selected_pred_idx]

    if ious[selected_pred_idx] > iou_threshold:
        return [selected_pred_idx, selected_pred]
    else:
        return None, None


def draw_bounding_boxes(
    image: np.ndarray, ground_truth: list, predictions: list, class_names: list
) -> np.ndarray:
    """
    Draw bounding boxes on an image to show the ground truth and predictions.

    Args:
        image (np.ndarray): The image to draw bounding boxes on.
        ground_truth (list): A list of ground truth bounding boxes.
        predictions (list): A list of predicted bounding boxes.
        class_names (list): A list of class names.

    Returns:
        np.ndarray: The image with bounding boxes drawn on it.
    """
    output_image = image.copy()

    # Draw ground truth bounding boxes with green color
    for x0, y0, x1, y1, label in ground_truth:
        # x0 = int(x0 * image.shape[1])
        # y0 = int(y0 * image.shape[0])
        # x1 = int(x1 * image.shape[1])
        # y1 = int(y1 * image.shape[0])
        print(x0, y0, x1, y1, "gt")
        cv2.rectangle(output_image, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(
            output_image,
            str(label),
            (x0, y0 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )

    print(ground_truth, predictions.xyxy)

    # Draw predicted bounding boxes with red color
    for pred in range(len(predictions.xyxy)):
        prediction = predictions.xyxy[pred].tolist()
        x0, y0, x1, y1 = prediction
        class_name = class_names[int(predictions.class_id[pred])]
        # scale to image
        cv2.rectangle(output_image, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(
            output_image,
            str(class_name),
            (x0, y0 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            2,
        )

    # show
    cv2.imshow("image", output_image)

    return output_image


def visualize_image_experiment(img_eval_data: dict, class_names: list) -> None:
    """
    Show the ground truth and predictions of an image using OpenCV.

    Args:
        img_eval_data: Evaluation data with ground truth and predictions.
        class_names: The list of class names.

    Returns:
        None
    """

    img_filename = img_eval_data["filename"]
    predictions = img_eval_data["predictions"]
    ground_truth = img_eval_data["ground_truth"]

    image = cv2.imread(img_filename)

    output_image = draw_bounding_boxes(image, ground_truth, predictions, class_names)

    base_file_name = img_filename.split("/")[-1]

    # save to ./output/images

    print(f"Saving image to ./output/images/{base_file_name}")

    cv2.imwrite(f"./output/images/{base_file_name}", output_image)
