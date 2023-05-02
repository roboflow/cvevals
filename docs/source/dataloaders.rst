Data Loaders
============

Before you can run an evaluation on a dataset, you need to prepare data in the format expected by the evaluations library.

Evaluations supports two methods of loading data:

1. Load from a JSON file with ground truth, the image file name and, optionally, predictions data;
2. Load from a dataset on the Roboflow platform.

Load from Roboflow
------------------

When you load data from Roboflow, a `dataset` directory will be created in which your data is stored.

This directory will not be purged by default. If you want to run an evaluation on multiple datasets, you need to purge the `datasets` directory after each directory.

.. code-block:: python

    from evaluations.dataloaders import RoboflowDataLoader

    data, model = RoboflowDataLoader(
        workspace_url="james-gallagher-87fuq",
        project_url="mug-detector-eocwp",
        project_version=12,
    )

.. autoclass:: evaluations.dataloaders.RoboflowDataLoader
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

Load from JSON
--------------

JSON data must be in one of two formats:

Object Detection
~~~~~~~~~~~~~~~~

.. code-block:: json

    [
        {
            "file.jpeg": {
                "predictions": [
                    [
                        x,
                        y,
                        w,
                        h
                    ]
                ],
                "ground_truth": [
                    [
                        x,
                        y,
                        w,
                        h
                    ]
                ]
                "file_name": "img.jpg"
            }
        }
        ...
    ]

Classification
~~~~~~~~~~~~~~

.. code-block:: json

    [
        {
            "predictions": ["prediction label"]
            "ground_truth": ["label"]
            "file_name": "img.jpg"
        }
        ...
    ]

To ingest a file with one of the aforementioned structures, use the following code:

.. code-block:: python

    from evaluations.dataloaders import JSONDataLoader

    data = RoboflowDataLoader("data.json")

.. autoclass:: evaluations.dataloaders.JSONDataLoader
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members: