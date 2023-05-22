import numpy as np
from .evaluator import Evaluator
from .confusion_matrices import plot_confusion_matrix
import cv2

class SegmentationEvaluator(Evaluator):
    def __init__(self, ground_truth, predictions, class_names, eval_data_path, dataset = "test") -> None:
        self.ground_truth = ground_truth
        self.predictions = predictions
        self.class_names = class_names
        self.eval_data_path = eval_data_path
        self.dataset = dataset

    def eval_model_predictions(self) -> dict:
        confusion_matrices = {}

        class_names = self.class_names
        predictions = self.predictions

        combined_cf = {"fn": 0, "fp": 0}
        combined_cf = {}

        for i, _ in enumerate(class_names):
            for j, _ in enumerate(class_names):
                combined_cf[(i, j)] = 0

        for image_name, prediction in self.ground_truth.annotations.items():
            mask = prediction.mask

            image_name = self.eval_data_path + "/test/images/" + image_name

            best_mask = 0
            best_iou = 0

            cf = {}

            for i, _ in enumerate(class_names):
                for j, _ in enumerate(class_names):
                    cf[(i, j)] = 0

            if image_name not in predictions:
                continue

            for pred in predictions[image_name]["predictions"]:
                class_id = pred[3]
                pred = pred[1]

                # show two masks on image
                image = cv2.imread(image_name)

                # convert img to 3 channels
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # convert mask to 1 channel
                mask = mask.astype(np.uint8)

                for m in mask:
                    intersection = np.logical_and(m, pred)
                    union = np.logical_or(m, pred)
                    iou = np.sum(intersection) / np.sum(union)

                    if iou > best_iou:
                        best_iou = iou
                        best_mask = class_id

            # if best mask has iou > threshold, add to confusion matrix
            if best_iou > 0.5:
                cf[(best_mask, best_mask)] += 1
            else:
                background_class_id = class_names.index("background")
                cf[(background_class_id, background_class_id)] += 1

            
            for i, _ in enumerate(class_names):
                for j, _ in enumerate(class_names):
                    combined_cf[(i, j)] += cf[(i, j)]

            confusion_matrices[image_name] = cf

            plot_confusion_matrix(cf, self.class_names, False, key, self.mode)

        plot_confusion_matrix(
            combined_cf, self.class_names, True, "aggregate", self.mode
        )

        self.combined_cf = combined_cf

        return combined_cf