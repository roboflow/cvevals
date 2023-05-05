![CV Evaluations banner](model-eval-banner.png)

# CV Evaluations 🔎

`cvevals` is a framework for evaluating the results of computer vision models. Think [OpenAI Evals](https://github.com/openai/evals), but for computer vision models.

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

Evaluations works with YOLOv5 PyTorch TXT data for object detection and Multiclass Classification TXT for classification.

This repository contains evaluation code for the following models:

- Roboflow models
- CLIP
- Grounding DINO

## Getting Started

To get started, clone the repository and install the required dependencies:

```bash

git clone https://github.com/roboflow/evaluations.git
cd evaluations
pip install -r requirements.txt
```

Next, you'll need to create an evaluator. Check out the `*_examples.py` files for examples on evaluators. For instance, if you want to evaluate a Roboflow model, open `roboflow_example.py` and paste in your model ID, version number, and workspace ID in the constants at the top of the file. Then, run the script to see the results of the evaluation.

## Evaluation Results

The script will print the precision, accuracy, and f1 score of the model performance to the console.

An `output/` folder will be created with two subfolders:

1. `images`: Images with ground truth and predictions.
2. `matrices`: Confusion matrices for each class. `aggregate.png` shows an aggregate confusion matrix for the model.

Here is an example confusion matrix from the `output/matrices` folder:

![Confusion matrix](aggregate.png)

## Notes on Using CLIP and Grounding DINO

### CLIP

When you first run the `clip_example.py` file, the weights for OpenAI's CLIP model will be downloaded if you do not already have the weights on the system.

### Grounding DINO

To use GroundingDINO, run the `dinosetup.sh` script in the root `evaluations` directory. This will install the model relative to the `evaluations` directory and download the model weights.

## License

This project is licensed under an [MIT License](LICENSE).

## Contributing

Interested in contributing to evaluations? Check out our [contributing guidelines](CONTRIBUTING.md).
