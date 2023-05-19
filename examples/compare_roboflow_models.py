import os

import pypandoc

from evaluations import CompareEvaluations
from evaluations.confusion_matrices import plot_confusion_matrix
from evaluations.dataloaders import (RoboflowDataLoader,
                                     RoboflowPredictionsDataLoader)
from evaluations.roboflow import RoboflowEvaluator

MODEL_TYPE = "object-detection"

models = []

evals = []
datasets = []

for m in models:
    loader = RoboflowDataLoader(
        workspace_url=m["workspace_url"],
        project_url=m["project_url"],
        project_version=m["model_version"],
        image_files=m["project_url"],
        model_type=MODEL_TYPE,
    )

    class_names, data, model = loader.download_dataset()

    predictions = RoboflowPredictionsDataLoader(
        model=model,
        model_type="object-detection",
        image_files=m["project_url"],
        class_names=class_names,
    ).process_files()

    long_name = f"{m['workspace_url']}/{m['project_url']}/{m['model_version']}"

    evals.append(
        RoboflowEvaluator(
            ground_truth=data,
            predictions=predictions,
            class_names=class_names,
            mode="batch",
            name=long_name,
            model_type=MODEL_TYPE,
        )
    )

    print(loader.dataset_content)

    datasets.append(loader.dataset_content)

best = CompareEvaluations(evals)

full_path = os.path.join(os.getcwd(), "output", "matrices")

best, evaluations = best.compare()

plot_confusion_matrix(
    best[4], class_names=best[3], file_name=f"Best Confusion Matrix", mode="return"
)

best_cf = (
    f"![Confusion Matrix]({full_path}/Best Confusion Matrix.png)" + "{ width=400px }"
)

cfs = [
    f"![Confusion Matrix]({full_path}/{c[6].replace('/', '_')}.png)" + "{ width=400px }"
    for c in evaluations
]

cfs = "\n\n".join(cfs)

dataset_meta = ""

for d in datasets:
    preprocessing = d.preprocessing
    augmentations = d.augmentation
    # convert to str
    augmentations = str(augmentations)
    preprocessing = str(preprocessing)
    splits = d.splits

    dataset_meta += f"### {d.name} (Version {d.version})\n\n"

    dataset_meta += f"#### Preprocessing\n\n"

    dataset_meta += f"> ```python\n{preprocessing}\n```\n\n"

    dataset_meta += f"\n\n#### Augmentations\n\n"

    dataset_meta += f"> ```python\n{augmentations}\n```\n\n"

    dataset_meta += f"\n\n#### Splits\n\n"

    dataset_meta += "Valid: " + str(splits["valid"]) + "\n\n"
    dataset_meta += "Train: " + str(splits["train"]) + "\n\n"
    dataset_meta += "Test: " + str(splits["test"]) + "\n\n"

model_table_data = ""

for c in evaluations:
    plot_confusion_matrix(
        c[4],
        file_name=f"{c[6].replace('/', '_')}",
        class_names=c[3],
        mode="return",
    )
    model_table_data += f"| {c[6]} | {round(c[0], 2)} | {round(c[1], 2)} | {round(c[2], 2)} | {c[5]} |\n"

report = f"""
# Roboflow Model Comparison

Total evaluations run: {len(evaluations)}

## Models

| Model | F1 Score | Precision | Recall | Confidence Threshold |
| --- | --- | --- | --- | --- |
{model_table_data}

## Datasets

{dataset_meta}

## Best Model (measured by f1 score)

{best_cf}

## Confusion Matrices by Model

{cfs}
"""

with open("roboflow_model_comparison.md", "w") as f:
    f.write(report)

pypandoc.convert_file(
    "roboflow_model_comparison.md",
    "pdf",
    outputfile="roboflow_model_comparison.pdf",
    extra_args=["--extract-media", ".", "-V", "geometry:margin=1in"],
)
