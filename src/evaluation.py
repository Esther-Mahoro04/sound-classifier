"""Evaluation: precision, recall, F1 and the confusion matrix."""
import json

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)


def text_report(y_true, y_pred, class_names):
    return classification_report(y_true, y_pred, target_names=class_names)


def dict_report(y_true, y_pred, class_names):
    return classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )


def save_report(report, path):
    with open(path, "w") as file:
        json.dump(report, file, indent=2)


def plot_confusion_matrix(y_true, y_pred, class_names, path):
    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(matrix, display_labels=class_names)
    display.plot(xticks_rotation=45, cmap="Blues")
    plt.tight_layout()
    plt.savefig(path)
    plt.show()