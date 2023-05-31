# Compare CLIP and DINOv2

Both CLIP and DINOv2 can be used for image classifcation.

CLIP can be used for zero-shot classification, whereas you need to train a classification model using DINOv2 embeddings.

The `compare_clip_dinov2.py` script accepts a Roboflow dataset and compares the performance of CLIP vs. an SVM linear classifier trained using the embeddings from DINOv2.

To use the script, run the following code:

```
python3 examples/dinov2_example.py \
    --eval_data_path=./images/ \
    --roboflow_workspace_url=<> \
    --roboflow_project_url=<> \
    --roboflow_model_version=<>
```

This script returns a table showing whether zero-shot CLIP or an SVM model trained using DINOv2 embeddings is better at classifying the data in your dataset.

This script may take some time to run depending on how many images are in the test set of the dataset on which you run inference.