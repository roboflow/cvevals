# Helper Scripts

The `cvevals` repository contains a number of helper scripts for evaluating annotation quality. These are stored in the `scripts` folder in the project GitHub repository, separate from the main evaluation codebase.

Here are the scripts currently available:

- [scripts/cutout.py](https://github.com/roboflow/cvevals/blob/main/scripts/cutout.py): Use CLIP to identify annotations that may be incorrectly labelled.
- [scripts/tighten_bounding_boxes.py](https://github.com/roboflow/cvevals/blob/main/scripts/tighten_bounding_boxes.py): Use the Segment Anything Model to recommend more precise bounding boxes around a feature in an image.