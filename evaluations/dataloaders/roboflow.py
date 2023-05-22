import csv
import os

import cv2
import numpy as np
import roboflow
import yaml
from supervision.detection.core import Detections

from .dataloader import DataLoader
from .yolov5 import get_ground_truth_for_image


class RoboflowDataLoader(DataLoader):
    def __init__(
        self,
        workspace_url: str,
        project_url: str,
        project_version: int,
        image_files: str,
        model_type: str = "object-detection",
        dataset: str = "test",
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
        self.dataset = dataset

        self.model = None
        self.dataset_version = None

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
        self.dataset_content = self.dataset_version
        self.model = project.version(self.project_version).model

        if self.model_type == "classification":
            data_format = "folder"
        elif self.model_type == "multiclass":
            data_format = "multiclass"
        elif self.model_type == "object-detection":
            data_format = "yolov5"
        elif self.model_type == "segmentation":
            data_format = "yolov5"
        else:
            raise ValueError("Model type not supported")

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
        elif data_format == "multiclass":
            with open(os.path.join(root_path, "valid/", "_classes.csv")) as f:
                reader = csv.reader(f)
                results = list(reader)

                class_names = results[0]

                # first item will be "filename", so we need to remove it
                self.class_names = [c.strip() for c in class_names][1:]

                self.class_names.append("background")

                for row in results[1:]:
                    self.data[os.path.join(root_path, "valid/", row[0])] = {
                        "filename": os.path.join(root_path, "valid/", row[0]),
                        "predictions": [],
                        "ground_truth": [
                            self.class_names[c - 1].strip()
                            for c in range(1, len(row))
                            if row[c].strip() == "1"
                        ],
                    }

                return self.class_names, self.data, self.model
        else:
            # class names are folder names in test/
            self.class_names = [
                name
                for name in os.listdir(os.path.join(root_path, "valid"))
                if os.path.isdir(os.path.join(root_path, "valid", name))
            ]

        for root, dirs, files in os.walk(
            self.image_files.rstrip("/") + f"/{self.dataset}/"
        ):
            for file in files:
                if file.endswith(".jpg"):
                    if self.model_type == "object-detection" or self.model_type == "segmentation":
                        ground_truth, masks = get_ground_truth_for_image(
                            os.path.join(root, file)
                        )
                        if masks != []:
                            ground_truth = masks
                    else:
                        # folder name
                        ground_truth = [os.path.basename(root)]

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
        class_names = []
        confidence = []
        masks = []

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

            # multiply by image dimensions to get pixel values
            x0 = int(x0 * width)
            y0 = int(y0 * height)
            x1 = int(x1 * width)
            y1 = int(y1 * height)

            if p.get("points"):
                # points are {x, y} pairs, turn into mask
                points = p["points"]

                # print(points)

                mask = np.zeros((int(height), int(width)), dtype=np.uint8)
                points = np.array(
                    [(p["x"], p["y"]) for p in points], dtype=np.int32
                ).reshape((-1, 1, 2))

                # entire points should be filled

                cv2.fillPoly(mask, [points], 1)

                # should be T or F
                mask = mask.astype(bool)

                # print # of True bools
                print(np.count_nonzero(mask))
                # cv2.imshow("mask", mask.astype(np.uint8) * 255)
                # if cv2.waitKey(0) == ord("q"):
                #     break

                masks.append(mask)

            predictions.append(
                (
                    x0,
                    y0,
                    x1,
                    y1,
                )
            )
            class_names.append(self.class_names.index(p["class"]))
            confidence.append(p["confidence"])

        if len(predictions) == 0:
            return Detections.empty()

        predictions = Detections(
            xyxy=np.array(predictions),
            class_id=np.array(class_names),
            mask=np.array(masks),
            confidence=np.array(confidence),
        )

        return predictions
