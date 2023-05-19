# Azure Evaluation

We have included an Azure Vision example to help you evaluate the performance of the large vision they offer against your data.

Before you can run the Azure example, you'll need to have an Azure Computer Vision endpoint set up on your Microsoft Azure account.

When you have your account ready, open up the `examples/azure_example.py` file and update the `class_mappings` dictionary with the classes you want to evaluate.

This dictionary should follow this format:

```
"azure_class_name": "roboflow_class_name"
```

For example, if you want to evaluate the performance of a model on the `Person` class from Azure and a `person` class in Roboflow, you would update the dictionary to look like this:

```
class_mappings = {
    "Person": "person"
}
```

This mapping is case sensitive.

Then, run the `azure_example.py` file with the following arguments:

```
python3 examples/azure_example.py --eval_data_path <path_to_eval_data> --roboflow_workspace_url <workspace_url> --roboflow_project_url <project_url> --roboflow_model_version <model_version> --azure_endpoint <azure_endpoint> --azure_api_key <azure_api_key>
```