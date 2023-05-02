Compare Prompts and Confidence Levels
=====================================

You can use `evaluations` to compare different prompts for zero-shot computer vision models and confidence levels for all models to help you choose the best prompt and confidence level for your use case.

To compare prompts and confidence levels, use the `CompareEvaluations` class. This class accepts two or more evaluation configurations and returns the best prompt and confidence level.

.. code-block:: python

    from evaluations.clip import CLIPEvaluator
    from evaluations.dataloaders import CLIPDataLoader, RoboflowDataLoader
    from evaluations import CompareEvaluations
    import copy

    EVAL_DATA_PATH = "/Users/james/src/clip/model_eval/dataset-new-apples"

    class_names, predictions, model = RoboflowDataLoader(
        workspace_url="mit-3xwsm",
        project_url="appling",
        project_version=1,
        image_files=EVAL_DATA_PATH,
        model_type="classification",
    ).download_dataset()

    evals = [
        ["orange", "background"],
        ["red apple", "background"],
    ]

    best = CompareEvaluations(
        [
            CLIPEvaluator(
                data=CLIPDataLoader(
                    data=copy.deepcopy(predictions),
                    class_names=cn,
                    eval_data_path=EVAL_DATA_PATH,
                ).process_files(),
                class_names=cn,
                mode="batch",
            )
            for cn in evals
        ]
    )

    print(best.compare())

.. autoclass:: evaluations.CompareEvaluations
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members: