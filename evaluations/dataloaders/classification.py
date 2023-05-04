from dataclasses import dataclass
import os

@dataclass
class ClassificationDetections:
    image_id: str
    predicted_class_names: list
    predicted_class_ids: list
    confidence: float

class ClassificationFolderDataLoader:
    def __init__(self, directory):
        self.directory = directory

    def get_files(self):
        class_names = []
        file_names = []

        for root, dirs, files in os.walk(self.directory):
            for file in files:
                if file.endswith(".jpg"):
                    file_name = os.path.join(root, file)
                    file_names.append(file_name)
                    class_names.append(root.split("/")[-1])

        return class_names, file_names