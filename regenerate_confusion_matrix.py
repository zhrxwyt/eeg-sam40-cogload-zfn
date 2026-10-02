"""Generate ulang gambar confusion matrix dari trial_predictions.csv
yang sudah tersimpan, tanpa perlu training ulang model.

Cara pakai:
    python regenerate_confusion_matrix.py --output-dir output_results_patience50 --model dnn --task binary
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # WAJIB sebelum import pyplot: hindari error Tcl/Tk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def class_names_for(task: str) -> list[str]:
    if task == "binary":
        return ["Relax", "Stress-task"]
    return ["Relax", "Stroop", "Mirror", "Arithmetic"]


def save_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
    model_name: str,
    dpi: int = 300,
) -> None:
    row_sum = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_sum,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sum != 0,
    )
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        normalized * 100,
        annot=True,
        fmt=".2f",
        cmap="PuRd",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Percentage (%)"},
        ax=axis,
    )
    axis.set_title(f"{model_name.upper()} normalized confusion matrix")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"Tersimpan: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Folder output eksperimen, misal output_results_patience50")
    parser.add_argument("--model", required=True, help="Nama model, misal dnn / cnn / rnn / hg-zfn")
    parser.add_argument("--task", choices=["binary", "4class"], default="binary")
    parser.add_argument("--figure-dpi", type=int, default=300)
    arguments = parser.parse_args()

    model_directory = Path(arguments.output_dir) / arguments.model
    predictions_path = model_directory / "trial_predictions.csv"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Tidak ditemukan: {predictions_path}")

    predictions = pd.read_csv(predictions_path)
    class_names = class_names_for(arguments.task)
    labels_order = list(range(len(class_names)))

    matrix = confusion_matrix(
        predictions["y_true"],
        predictions["y_pred"],
        labels=labels_order,
    )

    output_path = model_directory / "confusion_matrix.png"
    save_confusion_matrix(
        matrix,
        class_names,
        output_path,
        arguments.model,
        arguments.figure_dpi,
    )


if __name__ == "__main__":
    main()
