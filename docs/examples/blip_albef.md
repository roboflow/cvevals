# BLIP and ALBEF

## Installation

To use BLIP or ALBEF, you will need to install the [Salesforce LAVIS](https://github.com/salesforce/LAVIS/) package:

```
pip3 install salesforce-lavis
```

Note: This package will not install on M1 Macs.

## Usage

You can use BLIP and ALBEF as a zero-shot labeling model for classification tasks like you would BLIP.

```
python3 examples/blip_albef_example.py --eval_data_path=<path_to_eval_data> \ 
--roboflow_workspace_url=<workspace_url> \ 
--roboflow_project_url<project_url> \ 
--roboflow_model_version=<model_version>
```

To evaluate your data using BLIP, pass the `--feature-extractor=blip` argument. To evaluate your data using ALBEF, pass the `--feature-extractor=albef` argument.