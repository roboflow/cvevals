import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(
    image_eval_results: dict,
    class_names: list,
    aggregate: bool = False,
    file_name: str = "",
    mode: str = "interactive",
) -> None:
    """
    Plot an aggregate confusion matrix showing the results of the evaluation.
    """
    confusion = []

    # get base name
    file_name = file_name.split("/")[-1]

    for x in range(len(class_names)):
        row = []
        for y in range(len(class_names)):
            row.append(image_eval_results[(x, y)])
        confusion.append(row)

    fig = plt.figure(figsize=(10, 10))

    plt.title("Confusion Matrix for " + file_name)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    if aggregate:
        plt.title("Confusion Matrix (Aggregated)")

    heatmap = sns.heatmap(
        confusion,
        annot=True,
        xticklabels=class_names,
        yticklabels=class_names,
        fmt='g'
    )

    # axis names
    heatmap.set_xlabel("Predicted")
    heatmap.set_ylabel("Actual")

    if mode == "interactive":
        plt.title(file_name)
        plt.show()

    # save to ./output/matrices
    plt.savefig("./output/matrices/" + file_name + ".png")

    plt.close(fig)
