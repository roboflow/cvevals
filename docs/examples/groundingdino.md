# Grounding DINO Zero-Shot Evaluation

## Installation

To use GroundingDINO, run the `dinosetup.sh` script in the root `evaluations` directory. This will install the model relative to the `evaluations` directory and download the model weights.

## Usage

You can use Grounding DINO to test whether Grounding DINO can effectively label a specified class in your Roboflow dataset.

To evaluate your data using Grounding DINO, run the `dino_example.py` file with the following arguments:

```
python3 examples/dino_example.py --eval_data_path=<path_to_eval_data> \ 
--roboflow_workspace_url=<workspace_url> \ 
--roboflow_project_url<project_url> \ 
--roboflow_model_version=<model_version> \
--config_path=<path_to_dino_config_file> \
--weights_path=<path_to_dino_weights_file>
```