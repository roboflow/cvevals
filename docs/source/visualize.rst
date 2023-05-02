Visualizations
==============

Evaluations creates two visualizations:

1. A comparison of ground truth to model predictions for object detection models, and;
2. Confusion matrices showing the performance of a model.

To create these visualizations, you can run the functions outlined below on any Evaluator object.

Plot Ground Truth vs. Predictions
---------------------------------

The following function plots ground truth vs. predictions. This function *only works for object detection tasks*.

.. autofunction:: evaluations.eval_images.visualize_image_experiment

Plot Confusion Matrix
---------------------

.. autofunction:: evaluations.confusion_matrices.plot_confusion_matrix