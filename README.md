![CV Evaluations banner](images/model-eval-banner.png)

# CV Evaluations 🔎

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

- YOLOv5 PyTorch TXT (object detection)
- Multiclass Classification TXT (classification)
- Classification Folder (classification)

## Getting Started

To get started, clone the repository and install the required dependencies:

```bash

git clone https://github.com/roboflow/cvevals.git
cd cvevals
pip install -r requirements.txt
pip install -e .
```

Now you're ready to use this package!

Out of the box, we have created examples that let you evaluate the performance of the following models against your Roboflow datasets:

- CLIP (Classification)
- BLIP (Classification)
- ALBEF (Classification)
- Grounding DINO (Object Detection)
- DINOv2 and SVM (Classification)
- ImageBind (Classification)

## License

This project is licensed under an [MIT License](LICENSE).

## Contributing

Interested in contributing to evaluations? Check out our [contributing guidelines](CONTRIBUTING.md).
