import copy
import os
from dataclasses import dataclass

from .confusion_matrices import plot_confusion_matrix
from .eval_images import find_best_prediction, visualize_image_experiment

ACCEPTED_MODES = ["interactive", "batch"]


@dataclass
class EvaluatorResponse:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


class Evaluator:
    """
    Evaluates the output of a model on a dataset.
    """

    def __init__(
        self,
        predictions: dict = {},
        ground_truth: dict = {},
        class_names: list = [],
        confidence_threshold: int = 0.2,
        mode: str = "interactive",
    ) -> None:
        if mode not in ACCEPTED_MODES:
            raise ValueError(f"mode must be one of {ACCEPTED_MODES}")

        self.confidence_threshold = confidence_threshold
        self.mode = mode
        self.class_names = class_names

        if not os.path.exists("./output/images"):
            os.makedirs("./output/images")

        if not os.path.exists("./output/matrices"):
            os.makedirs("./output/matrices")

        merged_data = {}

        for key in predictions.keys():
            gt = ground_truth.get(key, [])

            if gt:
                gt = gt["ground_truth"]
            else:
                gt = []

            merged_data[key] = {
                "ground_truth": gt,
                "predictions": predictions[key]["predictions"],
                "filename": key,
            }

        self.data = merged_data

    def eval_model_predictions(self) -> dict:
        """
        Compute confusion matrices for a Roboflow, Grounding DINO, and CLIP model.

        Returns:
            combined_cf (dict): A dictionary with a confusion matrix composed of the result of all inferences.
        """

        confusion_matrices = {}

        combined_cf = {"fn": 0, "fp": 0}
        combined_cf = {}

        for i, _ in enumerate(self.class_names):
            for j, _ in enumerate(self.class_names):
                combined_cf[(i, j)] = 0

        for key, value in self.data.items():
            print(
                "evaluating image predictions against ground truth",
                key,
                "...",
                self.class_names,
            )

            gt = value["ground_truth"]
            pred = value["predictions"]

            cf = self.compute_confusion_matrix([gt, pred], self.class_names)

            for k, v in cf.items():
                combined_cf[k] += v

            confusion_matrices[key] = cf

            visualize_image_experiment(value, self.class_names)

            plot_confusion_matrix(cf, self.class_names, False, key, self.mode)

        plot_confusion_matrix(
            combined_cf, self.class_names, True, "aggregate", self.mode
        )

        self.combined_cf = combined_cf

        print("done evaluating model predictions")
        print(combined_cf)

        return combined_cf

    def compute_confusion_matrix(
        self,
        image_eval_data,
        class_names,
    ) -> dict:
        # image eval data looks like:
        # {filename: "path/to/image.jpg", ground_truth: [{}, {}, {}, ...], predictions: [{},{},{},...]}
        # each ground truth is a tuple/list of (x_min, y_min, x_max, y_max, label)
        # each prediction is a tuple/list of (x_min, y_min, x_max, y_max, label, confidence)

        predictions = copy.deepcopy(image_eval_data[1])

        all_predictions = []

        for i in range(len(predictions)):
            if (
                predictions.confidence[i] > self.confidence_threshold
                and predictions.class_id[i] is not None
            ):
                merged_prediction = predictions.xyxy[i].tolist() + [
                    predictions.class_id[i]
                ]

                all_predictions.append(merged_prediction)

        ground_truths = copy.deepcopy(image_eval_data[0])

        confusion_data = {}
        for i, _ in enumerate(class_names):
            for j, _ in enumerate(class_names):
                confusion_data[(i, j)] = 0

        for gt_box in ground_truths:
            match_idx, match = find_best_prediction(gt_box, all_predictions)

            if match is None:
                confusion_data[gt_box[4], len(class_names) - 1] += 1
            else:
                all_predictions.pop(match_idx)
                confusion_data[gt_box[4], match[4]] += 1

        for p in all_predictions:
            confusion_data[len(class_names) - 1, p[4]] += 1

        return confusion_data

    def calculate_statistics(self) -> tuple:
        """
        Calculate precision, recall, and f1 score for the evaluation.

        Returns:
            precision: The precision of the model
            recall: The recall of the the model
            f1: The f1 score of the model
        """
        cf = self.combined_cf

        # compute precision, recall, and f1 score
        tp = 0
        fp = 0
        fn = 0

        for x in range(len(self.class_names)):  # ground truth
            for y in range(len(self.class_names)):  # predictions
                if (
                    x == len(self.class_names) - 1
                ):  # last column / prediction with no ground truth
                    fp += cf[(x, y)]
                elif (
                    y == len(self.class_names) - 1
                ):  # bottom row / ground truth with no prediction
                    fn += cf[(x, y)]
                elif x == y:  # true positives across the diagonal
                    tp += cf[(x, y)]
                else:  # misclassification
                    fp += cf[(x, y)]

        if tp + fp == 0:
            precision = 1
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 1
        else:
            recall = tp / (tp + fn)

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        print(tp, fp, fn, precision, recall, f1)

        return EvaluatorResponse(
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1=f1,
        )
