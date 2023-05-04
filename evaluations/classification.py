from .evaluator import Evaluator

class ClassificationEvaluator(Evaluator):
    """
    Evaluate classification models.
    """

    def compute_confusion_matrix(self, result: dict, class_names: list) -> dict:
        """
        Compute a confusion matrix for a classification model.

        Args:
            result (dict): A dictionary containing the ground truth and predictions.
            class_names (list): A list of class names.

        """
        
        confusion_data = {}

        # matrix is
        # [[tp, fp]]

        for i, _ in enumerate(class_names):
            for j, _ in enumerate(class_names):
                confusion_data[(i, j)] = 0

        is_match = result[0][0] == result[1].predicted_class_names[0]

        if is_match:
            r0index = class_names.index(result[0][0])
            confusion_data[(r0index, r0index)] += 1
        else:
            r0index = class_names.index(result[0][0])
            r1index = class_names.index(result[1].predicted_class_names[0])
            confusion_data[(r0index, r1index)] += 1

        return confusion_data
