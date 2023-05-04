import json

from .dataloader import DataLoader


class JSONDataLoader(DataLoader):
    """
    Load JSON data for use in evaluation.
    """

    def __init__(self, json_file):
        with open(json_file, "r") as data:
            return json.loads(data)
