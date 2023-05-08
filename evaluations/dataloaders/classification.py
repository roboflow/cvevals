from dataclasses import dataclass


@dataclass
class ClassificationDetections:
    image_id: str
    predicted_class_names: list
    predicted_class_ids: list
    confidence: float
