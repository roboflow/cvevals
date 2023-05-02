Evaluate Object Detection Models
==========================================

Using `evaluations`, you can evaluate:

1. How close your Roboflow model output is to your ground truth annotations.
2. Which Grounding DINO prompt returns results closest to a set of annotated data.

Roboflow Evaluation
-------------------

To evaluate the performance of a Roboflow model against ground truth, use:

.. code-block:: python

    from evaluations.dataloaders import (
        RoboflowDataLoader,
        RoboflowPredictionsDataLoader,
    )
    from evaluations.roboflow import RoboflowEvaluator

    class_names, data, model = RoboflowDataLoader(
        workspace_url="james-gallagher-87fuq",
        project_url="mug-detector-eocwp",
        project_version=12,
        image_files="/Users/james/src/clip/model_eval/dataset-new",
    ).download_dataset()

    predictions = RoboflowPredictionsDataLoader(
        model=model,
        model_type="object-detection",
        image_files="/Users/james/src/clip/model_eval/dataset-new/",
        class_names=class_names,
    ).process_files()

    merged_data = {}

    for key, value in predictions.items():
        gt = data.get(key, [])

        merged_data[key] = {
            "ground_truth": gt["ground_truth"],
            "predictions": predictions[key]["predictions"],
            "filename": key,
        }

    evaluator = RoboflowEvaluator(data=merged_data, class_names=class_names, mode="batch")

    cf = evaluator.eval_model_predictions()

    precision, recall, f1 = evaluator.calculate_statistics()

    print("Precision:", precision)
    print("Recall:", recall)
    print("f1 Score:", f1)

.. autoclass:: evaluations.roboflow.RoboflowEvaluator
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

Grounding DINO Evaluation
-------------------------

Grounding DINO relies on having the Grounding DINO repository installed in a folder in your user path.

This is necessary because the Grounding DINO model is not packaged as a pip package and weights have to be installed separately.

To install Grounding DINO, follow the `instructions in the Grounding DINO repository <https://github.com...>`

To evaluate a Grounding DINO model against a dataset on Roboflow, use:

.. code-block:: python

    from evaluations import Evaluator
    from evaluations.dataloaders import RoboflowDataLoader, RoboflowPredictionsDataLoader
    from evaluations.dataloaders.dino import DINODataLoader

    class_names, data, model = RoboflowDataLoader(
        workspace_url="james-gallagher-87fuq",
        project_url="mug-detector-eocwp",
        project_version=12,
        image_files="/Users/james/src/clip/model_eval/dataset-new",
    ).download_dataset()

    predictions = DINODataLoader(
        data=data,
        class_names=["flames", "background"],
        mode="batch",
        image_files="/Users/james/src/clip/model_eval/dataset-new",
    ).process_files()

    merged_data = {}

    for key, value in predictions.items():
        gt = data.get(key, [])

        merged_data[key] = {
            "ground_truth": gt["ground_truth"],
            "predictions": predictions[key]["predictions"],
            "filename": key,
        }

    evaluator = Evaluator(data=merged_data, class_names=class_names, confidence_threshold=0.2, mode="batch")

    cf = evaluator.eval_model_predictions()

    precision, recall, f1 = evaluator.calculate_statistics()

    print("Precision:", precision)
    print("Recall:", recall)
    print("f1 Score:", f1)
