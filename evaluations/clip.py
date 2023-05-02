import glob
import os

import clip
import torch
from PIL import Image

from .evaluator import Evaluator

device = "cuda" if torch.cuda.is_available() else "cpu"


class CLIPEvaluator(Evaluator):
    """
    Evaluate CLIP prompts for classification tasks.
    """

    def compute_confusion_matrix(self, result: dict, class_names: list) -> dict:

        confusion_data = {}

        # matrix is
        # [[tp, fp]]

        for i, _ in enumerate(class_names):
            for j, _ in enumerate(class_names):
                confusion_data[(i, j)] = 0

        is_match = result[0][0] == result[1]

        if is_match:
            r0index = class_names.index(result[0][0])
            confusion_data[(r0index, r0index)] += 1
        else:
            r0index = class_names.index(result[0][0])
            r1index = class_names.index(result[1])
            confusion_data[(r0index, r1index)] += 1

        return confusion_data
