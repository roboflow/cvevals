# Roboflow Object Detection Evaluation

To evaluate a model hosted on Roboflow, run the `roboflow_example.py` file with the following arguments:

```
python3 examples/roboflow_example.py --eval_data_path <path_to_eval_data> \
--roboflow_workspace_url <workspace_url> \
--roboflow_project_url <project_url> \
--roboflow_model_version <model_version>
```

## Compare Roboflow Model Versions

You can also compare different versions of Roboflow models.

To do so, open the `examples/compare_roboflow_models.py` file and add the workspace URL, project URL, and versions for each model you want to compare. You should also set a confidence threshold for each model version that will be used to filter out low confidence predictions.

Then, run the `compare_roboflow_models.py` script:

```
python3 examples/compare_roboflow_models.py
```

This script will generate a PDF called `roboflow_model_comparison.pdf` that shows the precision, recall, f1 score, and confusion matrix for each model.