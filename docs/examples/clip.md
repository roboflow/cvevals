# CLIP

## Installation

When you first run any of the examples that use CLIP, the weights for OpenAI's CLIP model will be downloaded if you do not already have the weights on the system.

## Usage

You can run CLIP over images in your validation set to see how well the model performs at zero-shot annotation.

Run the `clip_example.py` file with the following arguments:

```
python3 examples/classification_example.py --eval_data_path=<path_to_eval_data> \ 
--roboflow_workspace_url=<workspace_url> \ 
--roboflow_project_url<project_url> \ 
--roboflow_model_version=<model_version>
```