![CV Evaluations banner](https://github.com/roboflow/cvevals/raw/main/images/model-eval-banner.png)

# CVevals 🔎

`cvevals` is a framework for evaluating the results of computer vision models. 

Roboflow `evaluations` is a Python package for evaluating computer vision models.

Using evaluations, you can:

1. Compare ground truth to a Roboflow model to benchmark and visualize model performance on images in bulk;
2. Test different Grounding DINO prompts to see which one most effectively annotates a specified class in an image;
3. Test different CLIP prompts to see which one most effectively classifies an image, and;
4. Evaluate resuts of different confidence levels for active learning.

Performance is measured using an aggregate of the following metrics:

- **Precision**
- **Recall**
- **F1 Score**

The following data formats are supported:

- YOLOv5 PyTorch TXT (object detection, segmentation)
- Multiclass Classification TXT (multi-class classification)
- Classification Folder (single-class classification)

## Results

Evaluators in the `CVevals` package return the precision, accuracy, and f1 score of model performance.

In addition, an `output/` folder will be created with two subfolders:

1. `images`: Images with ground truth and predictions (only for object detection models)
2. `matrices`: Confusion matrices for each class. `aggregate.png` shows an aggregate confusion matrix for the model.

Here is an example confusion matrix from the `output/matrices` folder:

![Confusion matrix](https://github.com/roboflow/cvevals/raw/main/images/example.png)

Here is an example of an image with ground truth and predictions from the `output/images` folder:

![Image of two cola bottles with ground truth and predictions](https://github.com/roboflow/cvevals/raw/main/images/annotated_example.jpg)

### Prompt Comparison

When you compare two prompts and/or confidence levels using the `CompareEvaluations.compare()` function, the results will be printed to the console, like so:

![Console output of prompt comparison](https://github.com/roboflow/cvevals/raw/main/images/prompt_comparison_table.png)

## Getting Started

To get started, clone the repository and install the required dependencies:

```bash

git clone https://github.com/roboflow/cvevals.git
cd cvevals
pip install -r requirements.txt
pip install -e .
```

Now you're ready to use CVevals!