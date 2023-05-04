class CompareEvaluations:
    """
    Compare multiple evaluations and return the best one.
    """

    def __init__(self, evaluations):
        self.evaluations = evaluations

    def compare(self):
        # find highest f1
        highest_f1 = -1
        highest_eval = None

        for evaluation in self.evaluations:
            cf = evaluation.eval_model_predictions()

            cf = evaluation.combined_cf

            data = evaluation.calculate_statistics()

            if data.f1 > highest_f1:
                highest_f1 = data.f1
                highest_eval = (
                    data.f1,
                    data.precision,
                    data.recall,
                    evaluation.class_names,
                    cf,
                    evaluation.confidence_threshold,
                )

        return highest_eval
