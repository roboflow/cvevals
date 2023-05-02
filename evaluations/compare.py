class CompareEvaluations:
    def __init__(self, evaluations):
        self.evaluations = evaluations

    def compare(self):
        # find highest f1
        highest_f1 = 0
        highest_eval = None

        for evaluation in self.evaluations:
            cf = evaluation.eval_model_predictions()
            precision, recall, f1 = evaluation.calculate_statistics()

            if f1 > highest_f1:
                highest_f1 = f1
                highest_eval = (
                    f1,
                    precision,
                    recall,
                    evaluation.class_names[0],
                    cf,
                    evaluation.confidence_threshold,
                )

        return highest_eval
