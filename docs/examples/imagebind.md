# ImageBind Classification

The `examples/imagebind_example.py` file lets you evaluate ImageBind set up for zero-shot classification on a dataset.

To run the example, first install and configure ImageBind:

```
chmod +x imagebind_setup.sh
```

Then, navigate to the `ImageBind` directory and run the example:

```
cd ImageBind/
python3 imagebind_example.py --eval_data_path=<path_to_eval_data> \ 
--roboflow_workspace_url=<workspace_url> \ 
--roboflow_project_url<project_url> \ 
--roboflow_model_version=<model_version>
```