"""Gabungkan semua summary.json per-model jadi satu tabel perbandingan.

Berguna karena model_comparison.csv bawaan script ditimpa ulang setiap kali
satu model dijalankan, sehingga hanya berisi model terakhir.

Cara pakai:
    python gabung_hasil.py --output-dir output_results_patience50
    python gabung_hasil.py --output-dir output_results_patience50 --sort-by roc_auc
"""

import argparse
import json
from pathlib import Path

import pandas as pd

KOLOM = [
    "model",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "stress_f1",
    "roc_auc",
    "subject_accuracy_ci_low",
    "subject_accuracy_ci_high",
    "number_of_trial_decisions",
    "debug_run",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Folder hasil, misal output_results_patience50")
    parser.add_argument("--sort-by", default="accuracy", help="Kolom untuk mengurutkan (default: accuracy)")
    parser.add_argument("--nama-file", default="ringkasan_semua_model.csv")
    arguments = parser.parse_args()

    folder = Path(arguments.output_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder tidak ditemukan: {folder}")

    baris = []
    for sub in sorted(folder.iterdir()):
        berkas = sub / "summary.json"
        if not berkas.is_file():
            continue
        with berkas.open(encoding="utf-8") as f:
            data = json.load(f)
        data["model"] = sub.name
        baris.append(data)

    if not baris:
        raise SystemExit(f"Tidak ada summary.json ditemukan di dalam {folder}")

    tabel = pd.DataFrame(baris)
    tabel = tabel[[k for k in KOLOM if k in tabel.columns]]
    if arguments.sort_by in tabel.columns:
        tabel = tabel.sort_values(arguments.sort_by, ascending=False)
    tabel.insert(0, "rank", range(1, len(tabel) + 1))

    tujuan = folder / arguments.nama_file
    tabel.to_csv(tujuan, index=False)

    print(tabel.to_string(index=False))
    print(f"\nTersimpan: {tujuan}")


if __name__ == "__main__":
    main()
