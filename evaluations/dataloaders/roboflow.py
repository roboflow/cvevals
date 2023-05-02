import os

import roboflow
import yaml

from .dataloader import DataLoader


class RoboflowDataLoader(DataLoader):
    def __init__(
        self,
        workspace_url: str,
        project_url: str,
        project_version: int,
        image_files: str,
        model_type: str = "object-detection",
    ):
        """
        Load a dataset from Roboflow. Saves the result to ./dataset/

        Args:
            workspace_url (str): The Roboflow workspace URL
            project_url (str): The Roboflow project URL
            project_version (int): The Roboflow project version
            model_type (str): The model type. Either "object-detection" or "classification"

        Returns:
            None
        """
        self.workspace_url = workspace_url
        self.project_url = project_url
        self.project_version = project_version
        self.model_type = model_type
        self.data = {}
        self.image_files = image_files

        self.model = None
        self.dataset_version = None

    def get_ground_truth_for_image(self, img_filename):
        labels = []
        label_file = img_filename.replace("/images/", "/labels/").replace(
            ".jpg", ".txt"
        )
        with open(label_file, "r") as file:
            for line in file:
                # Split the line into a list of values and convert them to floats
                values = list(map(float, line.strip().split()))

                # Extract the label, and scale the coordinates and dimensions
                label = int(values[0])
                cx = values[1]
                cy = values[2]
                width = values[3]
                height = values[4]

                x0 = cx - width / 2
                y0 = cy - height / 2
                x1 = cx + width / 2
                y1 = cy + height / 2

                # Add the extracted data to the output list
                labels.append((x0, y0, x1, y1, label))

        return labels

    def download_dataset(self) -> None:
        """
        Download a dataset from Roboflow. Saves the result to ./dataset/

        Returns:
            None
        """

        roboflow.login()
        rf = roboflow.Roboflow()

        self.data = {}

        project = rf.workspace(self.workspace_url).project(self.project_url)

        self.dataset_version = project.version(self.project_version)
        self.model = self.dataset_version.model

        if self.model_type == "classification":
            data_format = "folder"
        elif self.model_type == "object-detection":
            data_format = "yolov5"
        else:
            raise ValueError("Model type not supported")

        # TODO: Calculate correct path
        root_path = self.image_files

        # download if needed
        if not os.path.exists(root_path):
            self.dataset_version.download(data_format, root_path)

        if data_format == "yolov5":
            yaml_data = os.path.join(root_path, "data.yaml")
            if os.path.exists(yaml_data):
                # load class names map
                with open(yaml_data, "r") as file:
                    dataset_yaml = yaml.safe_load(file)
                    self.class_names = [
                        i.replace("-", " ") for i in dataset_yaml["names"]
                    ]
        else:
            # class names are folder names in test/
            self.class_names = [
                name
                for name in os.listdir(os.path.join(root_path, "valid"))
                if os.path.isdir(os.path.join(root_path, "valid", name))
            ]

        for root, dirs, files in os.walk(self.image_files.rstrip("/") + "/valid/"):
            for file in files:
                if file.endswith(".jpg"):
                    if self.model_type == "object-detection":
                        ground_truth = self.get_ground_truth_for_image(
                            os.path.join(root, file)
                        )
                    else:
                        ground_truth = []

                    self.data[os.path.join(root, file)] = {
                        "filename": os.path.join(root, file),
                        "predictions": [],
                        "ground_truth": ground_truth,
                    }

        self.class_names.append("background")

        return self.class_names, self.data, self.model


class RoboflowPredictionsDataLoader(DataLoader):
    def __init__(self, model, model_type, image_files, class_names):
        self.model = model
        self.model_type = model_type
        self.image_files = image_files
        self.data = {}
        self.class_names = class_names

    def get_predictions_for_image(self, img_filename: str) -> list:
        """
        Retrieve predictions for a Roboflow, Grounding DINO, and CLIP model for a single image.

        Args:
            img_filename (str): Path to image file

        Returns:
            predictions (list): List of predictions
        """

        prediction_group = self.model.predict(img_filename)

        image_dimensions = prediction_group.image_dims

        width, height = float(image_dimensions["width"]), float(
            image_dimensions["height"]
        )
        prediction_group = [p.json() for p in prediction_group.predictions]

        predictions = []

        for p in prediction_group:
            # scale predictions to 0 to 1
            cx = p["x"] / width
            cy = p["y"] / height
            w = p["width"] / width
            h = p["height"] / height

            x0 = cx - w / 2
            y0 = cy - h / 2
            x1 = cx + w / 2
            y1 = cy + h / 2

            predictions.append(
                (
                    x0,
                    y0,
                    x1,
                    y1,
                    self.class_names.index(p["class"]),
                    p["confidence"],
                )
            )

        return predictions
