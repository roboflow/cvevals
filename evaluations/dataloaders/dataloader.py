import glob
import os

N = 10


class DataLoader:
    def __init__(self):
        self.eval_data = {}
        self.data = {}
        self.image_files = None

    def process_files(self) -> None:
        """
        Process all input files in the dataset.

        Returns:
            None
        """

        for root, dirs, files in os.walk(self.image_files.rstrip("/") + "/valid/"):
            for file in files:
                if file.endswith(".jpg"):
                    file_name = os.path.join(root, file)
                    if file_name not in self.data:
                        self.data[file_name] = {}

                    self.data[file_name][
                        "predictions"
                    ] = self.get_predictions_for_image(file_name)

        return self.data
