from tabulate import tabulate

SUPPORTED_COMPARATORS = ["f1", "precision", "recall"]


class CompareEvaluations:
    """
    Compare multiple evaluations and return the best one.
    """

    def __init__(self, evaluations, comparator: str = "f1"):
        self.evaluations = evaluations
        self.comparator = comparator

    def compare(self):
        """
        Compare the evaluations and return the best one according to the specified comparator.
        """
        highest = -1
        highest_eval = None

        evaluations = []

        for evaluation in self.evaluations:
            cf = evaluation.eval_model_predictions()

            cf = evaluation.combined_cf

            data = evaluation.calculate_statistics()

            results = (
                data.f1,
                data.precision,
                data.recall,
                evaluation.class_names,
                cf,
                evaluation.confidence_threshold,
                evaluation.name,
            )

            if self.comparator not in SUPPORTED_COMPARATORS:
                raise Exception(
                    f"Comparator {self.comparator} not supported. Please use one of {SUPPORTED_COMPARATORS}."
                )

            if self.comparator == "f1":
                if data.f1 > highest:
                    highest = data.f1
                    highest_eval = results
            elif self.comparator == "precision":
                if data.precision > highest:
                    highest = data.precision
                    highest_eval = results
            elif self.comparator == "recall":
                if data.recall > highest:
                    highest = data.recall
                    highest_eval = results

            evaluations.append(results)

        table = tabulate(
            [
                [
                    "F1",
                    "Precision",
                    "Recall",
                    "Class Names",
                    "Confusion Matrix",
                    "Confidence",
                    "Name",
                ]
            ]
            + evaluations,
            headers="firstrow",
            tablefmt="fancy_grid",
        )

        print(table)

        return highest_eval, evaluations
