#!/usr/bin/env python3
"""Leakage-aware HG-ZFN benchmark for binary EEG stress-task classification.

This standalone script is the GitHub-ready version of the SAM-40 notebook. It
implements the Hybrid Gated Zero-Dimension Feature Network (HG-ZFN) together
with matched CNN, DNN, and RNN baselines.

Evaluation unit hierarchy
-------------------------
* Participant: train/test partition and statistical cluster.
* Trial: final classification target.
* Window: training sample whose probabilities are averaged within its trial.

The default experiment performs 40-fold subject-independent leave-one-subject-
out (LOSO) evaluation. Every learned preprocessing transform is fitted without
access to the held-out participant.

Examples
--------
Train and evaluate HG-ZFN using the pinned prefiltered SAM-40 repository::

    python hg_zfn_sam40.py --model hg-zfn

Run all four models::

    python hg_zfn_sam40.py --model all

Use an existing filtered_data directory::

    python hg_zfn_sam40.py --data-dir /path/to/filtered_data --model hg-zfn

The full LOSO experiment is computationally expensive because every outer fold
contains an inner validation fit followed by a full-development refit.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # cegah error Tcl/Tk saat generate PNG tanpa GUI

import argparse
import json
import os
import random
import re
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from scipy import signal
import seaborn as sns
import tensorflow as tf
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.layers import (
    Activation,
    AveragePooling1D,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LSTM,
    LayerNormalization,
    Multiply,
)
from tensorflow.keras.models import Sequential


DEFAULT_REPO_URL = "https://github.com/wavesresearch/eeg_stress_detection.git"
DEFAULT_DATA_COMMIT = "b97846d42ba9453c1b8e0004afd3408671140759"

SAMPLING_FREQUENCY = 128
EPOCH_LENGTH = SAMPLING_FREQUENCY
WINDOW_LENGTH = 512
WINDOW_HOP = 256
FREQUENCY_BANDS = [(1, 4), (4, 8), (8, 12), (12, 30), (30, 50)]

CHANNEL_ORDER = [
    "Cz", "Fz", "Fp1", "F7", "F3", "FC1", "C3", "FC5",
    "FT9", "T7", "CP5", "CP1", "P3", "P7", "PO9", "O1",
    "Pz", "Oz", "O2", "PO10", "P8", "P4", "CP2", "CP6",
    "T8", "FT10", "FC6", "C4", "FC2", "F4", "F8", "Fp2",
]
CHANNEL_INDEX = {name: index for index, name in enumerate(CHANNEL_ORDER)}
SYMMETRIC_PAIRS = [
    ("Fp1", "Fp2"), ("F7", "F8"), ("F3", "F4"),
    ("FC1", "FC2"), ("FC5", "FC6"), ("C3", "C4"),
    ("T7", "T8"), ("CP5", "CP6"), ("CP1", "CP2"),
    ("P3", "P4"), ("P7", "P8"), ("O1", "O2"),
    ("PO9", "PO10"), ("FT9", "FT10"),
]
SYMMETRIC_PAIR_INDEX = [
    (CHANNEL_INDEX[left], CHANNEL_INDEX[right])
    for left, right in SYMMETRIC_PAIRS
]
LABEL_PRIORITY = [
    ("relaxing", 0), ("relax", 0), ("stroop", 1),
    ("mirror_image", 2), ("mirror", 2), ("arithmetic", 3),
]


@dataclass(frozen=True)
class ExperimentConfig:
    task: str = "binary"
    seed: int = 42
    maximum_epochs: int = 120
    inner_validation_subjects: int = 5
    batch_size: int = 64
    learning_rate: float = 3e-4
    selected_features: int = 150
    figure_dpi: int = 300
    debug_folds: int | None = None
    save_fold_models: bool = False

    @property
    def class_names(self) -> list[str]:
        if self.task == "binary":
            return ["Relax", "Stress-task"]
        return ["Relax", "Stroop", "Mirror", "Arithmetic"]

    @property
    def number_of_classes(self) -> int:
        return len(self.class_names)


def set_global_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and TensorFlow seeds."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def enable_deterministic_tensorflow() -> None:
    """Enable deterministic TensorFlow operations when supported."""
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as error:  # pragma: no cover - depends on TF build
        print(f"Warning: TensorFlow op determinism is unavailable: {error}")


def run_command(command: list[str], cwd: Path | None = None) -> str:
    """Run a command and return standard output, raising on failure."""
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def prepare_dataset(
    data_directory: Path | None,
    clone_directory: Path,
    repository_url: str,
    requested_commit: str,
) -> tuple[Path, str]:
    """Resolve filtered SAM-40 data and record its repository revision.

    If ``data_directory`` is supplied, the directory is used directly and no
    download is performed. Otherwise the public prefiltered-data repository is
    cloned into ``clone_directory`` without deleting an existing checkout.
    """
    if data_directory is not None:
        resolved = data_directory.expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"Data directory not found: {resolved}")
        return resolved, "external-directory"

    clone_directory = clone_directory.expanduser().resolve()
    if not clone_directory.exists():
        clone_directory.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning SAM-40 prefiltered data into {clone_directory} ...")
        run_command(["git", "clone", repository_url, str(clone_directory)])
    elif not (clone_directory / ".git").is_dir():
        raise FileExistsError(
            f"{clone_directory} exists but is not a Git repository. "
            "Choose another --clone-dir or provide --data-dir."
        )

    if requested_commit:
        try:
            run_command(["git", "checkout", requested_commit], clone_directory)
        except subprocess.CalledProcessError:
            print("Pinned commit is not available locally; fetching it ...")
            run_command(
                ["git", "fetch", "origin", requested_commit], clone_directory
            )
            run_command(["git", "checkout", requested_commit], clone_directory)

    revision = run_command(["git", "rev-parse", "HEAD"], clone_directory)
    resolved = clone_directory / "Data" / "filtered_data"
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"Expected filtered SAM-40 directory was not found: {resolved}"
        )
    return resolved, revision


def label_from_filename(filename: str) -> int | None:
    lowered = filename.lower()
    for keyword, label in LABEL_PRIORITY:
        if keyword in lowered:
            return label
    return None


def subject_from_filename(filename: str) -> int | None:
    lowered = filename.lower()
    patterns = [r"sub[_\-]?(\d+)", r"subject[_\-]?(\d+)", r"s(\d+)"]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return int(match.group(1))
    return None


def load_sam40_trials(
    data_directory: Path,
    task: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load validated 32 x 3200 prefiltered SAM-40 trials."""
    raw_trials: list[np.ndarray] = []
    four_class_labels: list[int] = []
    subjects: list[int] = []
    filenames: list[str] = []

    mat_files = sorted(data_directory.glob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {data_directory}")

    for path in mat_files:
        label = label_from_filename(path.name)
        subject = subject_from_filename(path.name)
        if label is None or subject is None:
            continue

        matlab_data = scipy.io.loadmat(path)
        if "Clean_data" not in matlab_data:
            continue
        clean_data = np.asarray(matlab_data["Clean_data"], dtype=np.float32)
        if clean_data.shape != (32, 3200):
            continue

        raw_trials.append(np.nan_to_num(clean_data.T))  # time x channel
        four_class_labels.append(label)
        subjects.append(subject)
        filenames.append(path.name)

    raw = np.asarray(raw_trials, dtype=np.float32)
    label4 = np.asarray(four_class_labels, dtype=np.int64)
    subject_id = np.asarray(subjects, dtype=np.int64)
    target = (label4 != 0).astype(np.int64) if task == "binary" else label4

    if len(raw) != 480 or len(np.unique(subject_id)) != 40:
        raise ValueError(
            "Expected 480 valid trials from 40 participants, but found "
            f"{len(raw)} trials from {len(np.unique(subject_id))} participants."
        )

    print(
        f"Loaded {len(raw)} trials from {len(np.unique(subject_id))} participants "
        f"with labels {dict(sorted(Counter(target.tolist()).items()))}."
    )
    return raw, target, subject_id, filenames


def hjorth_parameters(samples: np.ndarray) -> tuple[float, float, float]:
    first_difference = np.diff(samples)
    second_difference = np.diff(first_difference)
    variance0 = np.var(samples) + 1e-12
    variance1 = np.var(first_difference) + 1e-12
    variance2 = np.var(second_difference) + 1e-12
    mobility = np.sqrt(variance1 / variance0)
    complexity = np.sqrt(variance2 / variance1) / (mobility + 1e-12)
    return float(variance0), float(mobility), float(complexity)


def absolute_band_power(
    samples: np.ndarray,
    sampling_frequency: int = SAMPLING_FREQUENCY,
) -> np.ndarray:
    segment_length = min(len(samples), sampling_frequency)
    overlap = segment_length // 2 if segment_length >= 4 else 0
    frequencies, power_spectral_density = signal.welch(
        samples,
        fs=sampling_frequency,
        window="hann",
        nperseg=segment_length,
        noverlap=overlap,
        detrend="constant",
        scaling="density",
    )
    frequency_width = (
        frequencies[1] - frequencies[0] if len(frequencies) > 1 else 1.0
    )
    return np.asarray(
        [
            power_spectral_density[
                (frequencies >= lower) & (frequencies < upper)
            ].sum()
            * frequency_width
            + 1e-12
            for lower, upper in FREQUENCY_BANDS
        ],
        dtype=np.float64,
    )


def extract_epoch_features(epoch: np.ndarray) -> np.ndarray:
    """Extract 678 spectral-statistical descriptors from one 32 x 128 epoch."""
    features: list[float] = []
    band_power = np.zeros((epoch.shape[0], len(FREQUENCY_BANDS)))

    for channel in range(epoch.shape[0]):
        samples = epoch[channel]
        activity, mobility, complexity = hjorth_parameters(samples)
        channel_power = absolute_band_power(samples)
        band_power[channel] = channel_power
        relative_power = channel_power / (channel_power.sum() + 1e-12)
        differential_entropy = 0.5 * np.log(
            2 * np.pi * np.e * channel_power
        )
        _, theta, alpha, beta, _ = channel_power
        ratios = [
            beta / (alpha + 1e-12),
            theta / (beta + 1e-12),
            theta / (alpha + 1e-12),
        ]
        features.extend(
            [
                activity,
                mobility,
                complexity,
                float(np.mean(samples)),
                float(np.std(samples)),
                float(np.sqrt(np.mean(samples**2))),
                *relative_power.tolist(),
                *differential_entropy.tolist(),
                *ratios,
            ]
        )

    for left, right in SYMMETRIC_PAIR_INDEX:
        asymmetry = np.log(band_power[left] + 1e-12) - np.log(
            band_power[right] + 1e-12
        )
        features.extend(asymmetry.tolist())

    feature_array = np.asarray(features, dtype=np.float32)
    if feature_array.shape != (678,):
        raise RuntimeError(
            f"Expected 678 epoch features, received {feature_array.shape}."
        )
    return feature_array


def create_windows(
    raw_trials: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create 4-second raw and feature windows while preserving trial identity."""
    feature_windows: list[np.ndarray] = []
    raw_windows: list[np.ndarray] = []
    window_labels: list[int] = []
    window_subjects: list[int] = []
    window_trials: list[int] = []

    for trial_id, raw_time_channel in enumerate(raw_trials):
        channel_time = raw_time_channel.T
        for start in range(
            0, channel_time.shape[1] - WINDOW_LENGTH + 1, WINDOW_HOP
        ):
            segment = channel_time[:, start : start + WINDOW_LENGTH].copy()
            segment -= segment.mean(axis=1, keepdims=True)

            epoch_features = np.asarray(
                [
                    extract_epoch_features(
                        segment[
                            :,
                            epoch * EPOCH_LENGTH : (epoch + 1) * EPOCH_LENGTH,
                        ]
                    )
                    for epoch in range(WINDOW_LENGTH // EPOCH_LENGTH)
                ]
            )
            feature_windows.append(
                np.concatenate(
                    [epoch_features.mean(axis=0), epoch_features.std(axis=0)]
                )
            )
            raw_windows.append(segment.T)
            window_labels.append(int(labels[trial_id]))
            window_subjects.append(int(subjects[trial_id]))
            window_trials.append(trial_id)

    feature_array = np.nan_to_num(
        np.asarray(feature_windows, dtype=np.float32)
    )
    raw_array = np.nan_to_num(np.asarray(raw_windows, dtype=np.float32))
    label_array = np.asarray(window_labels, dtype=np.int64)
    subject_array = np.asarray(window_subjects, dtype=np.int64)
    trial_array = np.asarray(window_trials, dtype=np.int64)

    print(
        f"Created feature windows {feature_array.shape} and raw windows "
        f"{raw_array.shape}."
    )
    return feature_array, raw_array, label_array, subject_array, trial_array


def verify_no_outer_leakage(
    features: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    trials: np.ndarray,
) -> None:
    """Assert participant- and trial-disjoint outer LOSO partitions."""
    for trial_id in np.unique(trials):
        mask = trials == trial_id
        if len(np.unique(subjects[mask])) != 1:
            raise AssertionError(f"Trial {trial_id} belongs to multiple subjects.")
        if len(np.unique(labels[mask])) != 1:
            raise AssertionError(f"Trial {trial_id} has multiple labels.")

    splitter = LeaveOneGroupOut()
    for fold, (development, test) in enumerate(
        splitter.split(features, labels, groups=subjects), start=1
    ):
        if not set(subjects[development]).isdisjoint(subjects[test]):
            raise AssertionError(f"Subject leakage detected in fold {fold}.")
        if not set(trials[development]).isdisjoint(trials[test]):
            raise AssertionError(f"Trial leakage detected in fold {fold}.")

    print(
        f"Leakage check passed: {len(np.unique(subjects))} subject-disjoint "
        f"folds and {len(np.unique(trials))} intact trials."
    )


def balanced_class_weights(labels: np.ndarray) -> dict[int, float]:
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return dict(zip(classes.astype(int), weights.astype(float)))


def build_dnn(number_of_features: int, number_of_classes: int) -> Model:
    return Sequential(
        [
            Input((number_of_features,)),
            Dense(
                96,
                activation="gelu",
                kernel_regularizer=regularizers.l2(1e-4),
            ),
            LayerNormalization(),
            Dropout(0.30),
            Dense(
                48,
                activation="gelu",
                kernel_regularizer=regularizers.l2(1e-4),
            ),
            LayerNormalization(),
            Dropout(0.20),
            Dense(number_of_classes, activation="softmax"),
        ],
        name="DNN",
    )


def build_cnn(raw_shape: tuple[int, ...], number_of_classes: int) -> Model:
    inputs = Input(raw_shape, name="raw")
    output = Conv1D(32, 9, padding="same", activation="gelu")(inputs)
    output = LayerNormalization()(output)
    output = AveragePooling1D(4)(output)
    output = Conv1D(64, 7, padding="same", activation="gelu")(output)
    output = LayerNormalization()(output)
    output = AveragePooling1D(4)(output)
    output = Conv1D(64, 5, padding="same", activation="gelu")(output)
    output = GlobalAveragePooling1D()(output)
    output = Dropout(0.30)(output)
    predictions = Dense(number_of_classes, activation="softmax")(output)
    return Model(inputs, predictions, name="CNN_raw")


def build_rnn(raw_shape: tuple[int, ...], number_of_classes: int) -> Model:
    inputs = Input(raw_shape, name="raw")
    output = AveragePooling1D(4)(inputs)
    output = LSTM(32, dropout=0.25)(output)
    output = Dense(32, activation="gelu")(output)
    output = Dropout(0.25)(output)
    predictions = Dense(number_of_classes, activation="softmax")(output)
    return Model(inputs, predictions, name="RNN_raw")


def feature_gate(inputs: tf.Tensor, reduction: int = 12) -> tf.Tensor:
    dimensions = int(inputs.shape[-1])
    gate = Dense(max(dimensions // reduction, 8), activation="gelu")(inputs)
    gate = Dense(dimensions, activation="sigmoid")(gate)
    return Multiply()([inputs, gate])


def build_hg_zfn(
    number_of_features: int,
    raw_shape: tuple[int, ...],
    number_of_classes: int,
) -> Model:
    """Build the Hybrid Gated Zero-Dimension Feature Network."""
    raw_inputs = Input(raw_shape, name="raw")
    temporal = Conv1D(32, 9, padding="same", activation="gelu")(raw_inputs)
    temporal = LayerNormalization()(temporal)
    temporal = AveragePooling1D(4)(temporal)
    temporal = Conv1D(64, 7, padding="same", activation="gelu")(temporal)
    temporal = LayerNormalization()(temporal)
    temporal = AveragePooling1D(4)(temporal)
    temporal = Conv1D(64, 5, padding="same", activation="gelu")(temporal)
    temporal = GlobalAveragePooling1D()(temporal)
    temporal = Dropout(0.30)(temporal)

    feature_inputs = Input((number_of_features,), name="features")
    spectral = feature_gate(feature_inputs)
    spectral = Dense(
        48,
        activation="gelu",
        kernel_regularizer=regularizers.l2(1e-4),
    )(spectral)
    spectral = LayerNormalization()(spectral)
    spectral = Dropout(0.20)(spectral)

    fused = layers.Concatenate()([temporal, spectral])
    predictions = Dense(number_of_classes, activation="softmax")(fused)
    return Model(
        [feature_inputs, raw_inputs], predictions, name="HG_ZFN"
    )


def trial_means(
    features: np.ndarray,
    labels: np.ndarray,
    trial_id: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean_features: list[np.ndarray] = []
    trial_labels: list[int] = []
    for trial in np.unique(trial_id[indices]):
        selected = indices[trial_id[indices] == trial]
        mean_features.append(features[selected].mean(axis=0))
        unique_labels = np.unique(labels[selected])
        if len(unique_labels) != 1:
            raise AssertionError(f"Trial {trial} contains multiple labels.")
        trial_labels.append(int(unique_labels[0]))
    return np.asarray(mean_features), np.asarray(trial_labels)


def fit_feature_preprocessor(
    features: np.ndarray,
    labels: np.ndarray,
    trial_id: np.ndarray,
    indices: np.ndarray,
    number_of_features: int,
    seed: int,
) -> tuple[StandardScaler, SelectKBest]:
    scaler = StandardScaler().fit(features[indices])
    trial_features, trial_labels = trial_means(
        features, labels, trial_id, indices
    )
    score_function = partial(mutual_info_classif, random_state=seed)
    selector = SelectKBest(
        score_function, k=min(number_of_features, features.shape[1])
    )
    selector.fit(scaler.transform(trial_features), trial_labels)
    return scaler, selector


def transform_features(
    features: np.ndarray,
    indices: np.ndarray,
    scaler: StandardScaler,
    selector: SelectKBest,
) -> np.ndarray:
    return selector.transform(scaler.transform(features[indices])).astype(
        np.float32
    )


def fit_raw_scaler(
    raw_windows: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = raw_windows[indices].mean(axis=(0, 1), keepdims=True)
    standard_deviation = raw_windows[indices].std(
        axis=(0, 1), keepdims=True
    ) + 1e-8
    return mean, standard_deviation


def transform_raw(
    raw_windows: np.ndarray,
    indices: np.ndarray,
    mean: np.ndarray,
    standard_deviation: np.ndarray,
) -> np.ndarray:
    return (
        (raw_windows[indices] - mean) / standard_deviation
    ).astype(np.float32)


def inner_subject_split(
    development_indices: np.ndarray,
    subject_id: np.ndarray,
    validation_subjects: int,
    fold_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_subjects,
        random_state=fold_seed,
    )
    fitting_relative, validation_relative = next(
        splitter.split(
            development_indices,
            groups=subject_id[development_indices],
        )
    )
    return (
        development_indices[fitting_relative],
        development_indices[validation_relative],
    )


def learning_rate_callback() -> tf.keras.callbacks.Callback:
    def schedule(epoch: int, learning_rate: float) -> float:
        if epoch > 0 and epoch % 30 == 0:
            return max(float(learning_rate) * 0.5, 1e-6)
        return float(learning_rate)

    return tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)


def validation_callbacks() -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=50, restore_best_weights=True
        ),
        learning_rate_callback(),
    ]


def compile_model(model: Model, learning_rate: float) -> Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def aggregate_trial_probabilities(
    window_probabilities: np.ndarray,
    window_labels: np.ndarray,
    window_subjects: np.ndarray,
    window_trials: np.ndarray,
) -> dict[str, np.ndarray]:
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    probabilities: list[np.ndarray] = []
    subjects: list[int] = []
    trials: list[int] = []

    for trial in np.unique(window_trials):
        mask = window_trials == trial
        labels = np.unique(window_labels[mask])
        subject = np.unique(window_subjects[mask])
        if len(labels) != 1 or len(subject) != 1:
            raise AssertionError("Each trial must have one label and one subject.")
        probability = window_probabilities[mask].mean(axis=0)
        true_labels.append(int(labels[0]))
        predicted_labels.append(int(probability.argmax()))
        probabilities.append(probability)
        subjects.append(int(subject[0]))
        trials.append(int(trial))

    return {
        "y_true": np.asarray(true_labels),
        "y_pred": np.asarray(predicted_labels),
        "y_probability": np.asarray(probabilities),
        "subject": np.asarray(subjects),
        "trial": np.asarray(trials),
    }


def merge_fold_results(
    fold_results: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    keys = ["y_true", "y_pred", "y_probability", "subject", "trial"]
    return {
        key: np.concatenate([result[key] for result in fold_results])
        for key in keys
    }


def model_family(model_name: str) -> str:
    if model_name == "dnn":
        return "feature"
    if model_name in {"cnn", "rnn"}:
        return "raw"
    if model_name == "hg-zfn":
        return "hybrid"
    raise ValueError(f"Unknown model: {model_name}")


def build_selected_model(
    model_name: str,
    number_of_classes: int,
    feature_shape: int | None = None,
    raw_shape: tuple[int, ...] | None = None,
) -> Model:
    if model_name == "dnn" and feature_shape is not None:
        return build_dnn(feature_shape, number_of_classes)
    if model_name == "cnn" and raw_shape is not None:
        return build_cnn(raw_shape, number_of_classes)
    if model_name == "rnn" and raw_shape is not None:
        return build_rnn(raw_shape, number_of_classes)
    if (
        model_name == "hg-zfn"
        and feature_shape is not None
        and raw_shape is not None
    ):
        return build_hg_zfn(feature_shape, raw_shape, number_of_classes)
    raise ValueError(f"Missing input shape for {model_name}.")


def prepare_inputs(
    family: str,
    feature_windows: np.ndarray,
    raw_windows: np.ndarray,
    labels: np.ndarray,
    trial_id: np.ndarray,
    fitting_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    selected_features: int,
    seed: int,
) -> tuple[
    np.ndarray | list[np.ndarray],
    np.ndarray | list[np.ndarray],
    int | None,
    tuple[int, ...] | None,
]:
    feature_fitting = feature_evaluation = None
    raw_fitting = raw_evaluation = None

    if family in {"feature", "hybrid"}:
        scaler, selector = fit_feature_preprocessor(
            feature_windows,
            labels,
            trial_id,
            fitting_indices,
            selected_features,
            seed,
        )
        feature_fitting = transform_features(
            feature_windows, fitting_indices, scaler, selector
        )
        feature_evaluation = transform_features(
            feature_windows, evaluation_indices, scaler, selector
        )

    if family in {"raw", "hybrid"}:
        mean, standard_deviation = fit_raw_scaler(
            raw_windows, fitting_indices
        )
        raw_fitting = transform_raw(
            raw_windows, fitting_indices, mean, standard_deviation
        )
        raw_evaluation = transform_raw(
            raw_windows, evaluation_indices, mean, standard_deviation
        )

    if family == "feature":
        assert feature_fitting is not None and feature_evaluation is not None
        return feature_fitting, feature_evaluation, feature_fitting.shape[1], None
    if family == "raw":
        assert raw_fitting is not None and raw_evaluation is not None
        return raw_fitting, raw_evaluation, None, raw_fitting.shape[1:]

    assert feature_fitting is not None and feature_evaluation is not None
    assert raw_fitting is not None and raw_evaluation is not None
    return (
        [feature_fitting, raw_fitting],
        [feature_evaluation, raw_evaluation],
        feature_fitting.shape[1],
        raw_fitting.shape[1:],
    )


def run_subject_independent_loso(
    model_name: str,
    feature_windows: np.ndarray,
    raw_windows: np.ndarray,
    labels: np.ndarray,
    subject_id: np.ndarray,
    trial_id: np.ndarray,
    config: ExperimentConfig,
    output_directory: Path,
) -> dict[str, object]:
    """Run nested, leakage-aware outer LOSO for one model family."""
    family = model_family(model_name)
    splitter = LeaveOneGroupOut()
    splits = list(
        splitter.split(feature_windows, labels, groups=subject_id)
    )
    if config.debug_folds is not None:
        splits = splits[: config.debug_folds]

    fold_results: list[dict[str, np.ndarray]] = []
    selected_epochs: list[int] = []
    histories: list[dict[str, list[float]]] = []
    started = time.time()

    model_directory = output_directory / "models" / model_name
    if config.save_fold_models:
        model_directory.mkdir(parents=True, exist_ok=True)

    for fold, (development, test) in enumerate(splits, start=1):
        fitting, validation = inner_subject_split(
            development,
            subject_id,
            config.inner_validation_subjects,
            config.seed + fold,
        )

        fitting_input, validation_input, feature_shape, raw_shape = prepare_inputs(
            family,
            feature_windows,
            raw_windows,
            labels,
            trial_id,
            fitting,
            validation,
            config.selected_features,
            config.seed + fold,
        )

        tf.keras.backend.clear_session()
        set_global_seed(config.seed + fold)
        temporary_model = compile_model(
            build_selected_model(
                model_name,
                config.number_of_classes,
                feature_shape,
                raw_shape,
            ),
            config.learning_rate,
        )
        history = temporary_model.fit(
            fitting_input,
            labels[fitting],
            validation_data=(validation_input, labels[validation]),
            epochs=config.maximum_epochs,
            batch_size=config.batch_size,
            class_weight=balanced_class_weights(labels[fitting]),
            callbacks=validation_callbacks(),
            verbose=0,
        )
        histories.append(
            {
                key: [float(value) for value in history.history[key]]
                for key in ["accuracy", "val_accuracy", "loss", "val_loss"]
            }
        )
        best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
        selected_epochs.append(best_epoch)

        development_input, test_input, feature_shape, raw_shape = prepare_inputs(
            family,
            feature_windows,
            raw_windows,
            labels,
            trial_id,
            development,
            test,
            config.selected_features,
            config.seed + 1000 + fold,
        )

        tf.keras.backend.clear_session()
        set_global_seed(config.seed + 1000 + fold)
        final_model = compile_model(
            build_selected_model(
                model_name,
                config.number_of_classes,
                feature_shape,
                raw_shape,
            ),
            config.learning_rate,
        )
        final_model.fit(
            development_input,
            labels[development],
            epochs=best_epoch,
            batch_size=config.batch_size,
            class_weight=balanced_class_weights(labels[development]),
            callbacks=[learning_rate_callback()],
            verbose=0,
        )

        window_probability = final_model.predict(test_input, verbose=0)
        fold_results.append(
            aggregate_trial_probabilities(
                window_probability,
                labels[test],
                subject_id[test],
                trial_id[test],
            )
        )

        held_out_subject = int(np.unique(subject_id[test])[0])
        if config.save_fold_models:
            final_model.save(
                model_directory / f"held_out_subject_{held_out_subject:02d}.keras"
            )

        elapsed = time.time() - started
        print(
            f"{model_name}: fold {fold}/{len(splits)}, "
            f"held-out subject {held_out_subject}, best epoch {best_epoch}, "
            f"elapsed {elapsed:.0f}s"
        )

    merged = merge_fold_results(fold_results)
    expected_trials = 480 if config.debug_folds is None else 12 * len(splits)
    if len(np.unique(merged["trial"])) != expected_trials:
        raise RuntimeError(
            f"{model_name}: expected {expected_trials} trial decisions, "
            f"received {len(np.unique(merged['trial']))}."
        )

    return {
        **merged,
        "best_epoch": np.asarray(selected_epochs),
        "history": histories,
        "is_debug_run": config.debug_folds is not None,
    }


def summarize_result(
    result: dict[str, object],
    number_of_classes: int,
) -> dict[str, object]:
    true_labels = np.asarray(result["y_true"])
    predicted_labels = np.asarray(result["y_pred"])
    probabilities = np.asarray(result["y_probability"])
    summary: dict[str, object] = {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "balanced_accuracy": float(
            balanced_accuracy_score(true_labels, predicted_labels)
        ),
        "macro_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            true_labels,
            predicted_labels,
            labels=np.arange(number_of_classes),
        ),
    }
    if number_of_classes == 2:
        summary["stress_f1"] = float(
            f1_score(
                true_labels, predicted_labels, pos_label=1, zero_division=0
            )
        )
        summary["roc_auc"] = float(
            roc_auc_score(true_labels, probabilities[:, 1])
        )
    else:
        summary["stress_f1"] = None
        summary["roc_auc"] = float(
            roc_auc_score(
                true_labels,
                probabilities,
                multi_class="ovr",
                average="macro",
            )
        )
    return summary


def subject_accuracies(
    result: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    subject = np.asarray(result["subject"])
    true_labels = np.asarray(result["y_true"])
    predicted_labels = np.asarray(result["y_pred"])
    unique_subjects = np.unique(subject)
    scores = np.asarray(
        [
            accuracy_score(
                true_labels[subject == participant],
                predicted_labels[subject == participant],
            )
            for participant in unique_subjects
        ]
    )
    return unique_subjects, scores


def subject_cluster_confidence_interval(
    result: dict[str, object],
    repetitions: int = 10000,
    seed: int = 42,
) -> tuple[float, float, float]:
    _, scores = subject_accuracies(result)
    generator = np.random.default_rng(seed)
    bootstrap = generator.choice(
        scores, size=(repetitions, len(scores)), replace=True
    ).mean(axis=1)
    return (
        float(scores.mean()),
        float(np.percentile(bootstrap, 2.5)),
        float(np.percentile(bootstrap, 97.5)),
    )


def paired_subject_permutation(
    result_a: dict[str, object],
    result_b: dict[str, object],
    repetitions: int = 20000,
    seed: int = 42,
) -> tuple[float, float]:
    subjects_a, score_a = subject_accuracies(result_a)
    subjects_b, score_b = subject_accuracies(result_b)
    if not np.array_equal(subjects_a, subjects_b):
        raise ValueError("Paired tests require identical subject ordering.")
    difference = score_a - score_b
    observed = abs(difference.mean())
    generator = np.random.default_rng(seed)
    signs = generator.choice([-1, 1], size=(repetitions, len(difference)))
    null_distribution = np.abs((signs * difference).mean(axis=1))
    p_value = (1 + np.sum(null_distribution >= observed)) / (
        repetitions + 1
    )
    return float(difference.mean()), float(p_value)


def save_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
    model_name: str,
    dpi: int,
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


def aligned_history(
    histories: list[dict[str, list[float]]],
    metric: str,
) -> np.ndarray:
    maximum_length = max(len(history[metric]) for history in histories)
    values = np.full((len(histories), maximum_length), np.nan)
    for index, history in enumerate(histories):
        sequence = np.asarray(history[metric], dtype=float)
        values[index, : len(sequence)] = sequence
    return values


def save_training_curves(
    histories: list[dict[str, list[float]]],
    output_directory: Path,
    model_name: str,
    dpi: int,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    for metric, validation_metric, label in [
        ("accuracy", "val_accuracy", "Accuracy"),
        ("loss", "val_loss", "Loss"),
    ]:
        training = aligned_history(histories, metric)
        validation = aligned_history(histories, validation_metric)
        epochs = np.arange(1, training.shape[1] + 1)

        figure, axis = plt.subplots(figsize=(7, 5))
        training_mean = np.nanmean(training, axis=0)
        training_std = np.nanstd(training, axis=0)
        validation_mean = np.nanmean(validation, axis=0)
        validation_std = np.nanstd(validation, axis=0)
        axis.plot(
            epochs,
            training_mean,
            label=f"Training {label}",
            color="#8B0000",
            linewidth=2,
        )
        axis.fill_between(
            epochs,
            training_mean - training_std,
            training_mean + training_std,
            color="#8B0000",
            alpha=0.12,
        )
        axis.plot(
            epochs,
            validation_mean,
            label=f"Validation {label}",
            color="#7F7F7F",
            linestyle="--",
            linewidth=2,
        )
        axis.fill_between(
            epochs,
            validation_mean - validation_std,
            validation_mean + validation_std,
            color="#7F7F7F",
            alpha=0.12,
        )
        axis.set_xlabel("Epoch")
        axis.set_ylabel(label)
        axis.set_title(f"{model_name.upper()} mean LOSO {label.lower()}")
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        figure.savefig(
            output_directory / f"{metric}_{model_name}.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(figure)


def save_model_result(
    model_name: str,
    result: dict[str, object],
    config: ExperimentConfig,
    output_directory: Path,
) -> dict[str, object]:
    model_output = output_directory / model_name
    model_output.mkdir(parents=True, exist_ok=True)
    summary = summarize_result(result, config.number_of_classes)

    serializable_summary = {
        key: value
        for key, value in summary.items()
        if key != "confusion_matrix"
    }
    mean_accuracy, confidence_low, confidence_high = (
        subject_cluster_confidence_interval(result, seed=config.seed)
    )
    serializable_summary.update(
        {
            "subject_mean_accuracy": mean_accuracy,
            "subject_accuracy_ci_low": confidence_low,
            "subject_accuracy_ci_high": confidence_high,
            "number_of_trial_decisions": int(len(np.asarray(result["y_true"]))),
            "debug_run": bool(result["is_debug_run"]),
        }
    )
    with (model_output / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(serializable_summary, file, indent=2)

    probability = np.asarray(result["y_probability"])
    predictions = pd.DataFrame(
        {
            "subject": np.asarray(result["subject"]).astype(int),
            "trial": np.asarray(result["trial"]).astype(int),
            "y_true": np.asarray(result["y_true"]).astype(int),
            "y_pred": np.asarray(result["y_pred"]).astype(int),
            **{
                f"probability_{class_name.lower().replace('-', '_')}":
                probability[:, class_index]
                for class_index, class_name in enumerate(config.class_names)
            },
        }
    )
    predictions.to_csv(model_output / "trial_predictions.csv", index=False)

    report = classification_report(
        np.asarray(result["y_true"]),
        np.asarray(result["y_pred"]),
        target_names=config.class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(
        model_output / "classification_report.csv"
    )

    subjects, scores = subject_accuracies(result)
    pd.DataFrame(
        {"subject": subjects.astype(int), "accuracy": scores}
    ).to_csv(model_output / "subject_accuracies.csv", index=False)

    pd.DataFrame(
        {"fold": np.arange(1, len(result["best_epoch"]) + 1),
         "best_epoch": np.asarray(result["best_epoch"]).astype(int)}
    ).to_csv(model_output / "selected_epochs.csv", index=False)

    save_confusion_matrix(
        np.asarray(summary["confusion_matrix"]),
        config.class_names,
        model_output / "confusion_matrix.png",
        model_name,
        config.figure_dpi,
    )
    save_training_curves(
        result["history"],
        model_output,
        model_name,
        config.figure_dpi,
    )
    return serializable_summary


def save_binary_comparison_plots(
    results: dict[str, dict[str, object]],
    output_directory: Path,
    dpi: int,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    for model_name, result in results.items():
        true_labels = np.asarray(result["y_true"])
        positive_probability = np.asarray(result["y_probability"])[:, 1]
        false_positive_rate, true_positive_rate, _ = roc_curve(
            true_labels, positive_probability
        )
        auc_value = roc_auc_score(true_labels, positive_probability)
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{model_name.upper()} (AUC={auc_value:.3f})",
            linewidth=2,
        )
    axis.plot([0, 1], [0, 1], color="gray", linestyle="--")
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.set_title("Trial-level ROC curves")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(
        output_directory / "roc_all_models.png", dpi=dpi, bbox_inches="tight"
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 6))
    for model_name, result in results.items():
        true_labels = np.asarray(result["y_true"])
        positive_probability = np.asarray(result["y_probability"])[:, 1]
        precision, recall, _ = precision_recall_curve(
            true_labels, positive_probability
        )
        axis.plot(
            recall,
            precision,
            label=model_name.upper(),
            linewidth=2,
        )
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Trial-level precision-recall curves")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(
        output_directory / "precision_recall_all_models.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_experiment_outputs(
    results: dict[str, dict[str, object]],
    config: ExperimentConfig,
    output_directory: Path,
    dataset_revision: str,
    command_arguments: dict[str, object],
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    for model_name, result in results.items():
        summary = save_model_result(
            model_name, result, config, output_directory
        )
        summary_rows.append({"model": model_name, **summary})
    pd.DataFrame(summary_rows).to_csv(
        output_directory / "model_comparison.csv", index=False
    )

    if len(results) > 1 and config.task == "binary":
        save_binary_comparison_plots(
            results, output_directory, config.figure_dpi
        )

    if "hg-zfn" in results and len(results) > 1 and config.debug_folds is None:
        paired_rows = []
        for competitor, competitor_result in results.items():
            if competitor == "hg-zfn":
                continue
            difference, p_value = paired_subject_permutation(
                results["hg-zfn"], competitor_result, seed=config.seed
            )
            paired_rows.append(
                {
                    "comparison": f"HG-ZFN vs {competitor.upper()}",
                    "mean_accuracy_difference": difference,
                    "p_value": p_value,
                }
            )
        pd.DataFrame(paired_rows).to_csv(
            output_directory / "paired_subject_tests.csv", index=False
        )

    metadata = {
        "dataset_revision": dataset_revision,
        "tensorflow_version": tf.__version__,
        "config": config.__dict__,
        "arguments": command_arguments,
    }
    with (output_directory / "run_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2, default=str)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-aware subject-independent HG-ZFN benchmark on SAM-40."
        )
    )
    parser.add_argument(
        "--model",
        choices=["hg-zfn", "cnn", "dnn", "rnn", "all"],
        default="hg-zfn",
        help="Model to evaluate; default: hg-zfn.",
    )
    parser.add_argument(
        "--task",
        choices=["binary", "4class"],
        default="binary",
        help="Classification target; default: binary Relax vs stress-task.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Existing SAM-40 filtered_data directory. Skips Git cloning.",
    )
    parser.add_argument(
        "--clone-dir",
        type=Path,
        default=Path("data/eeg_stress_detection"),
        help="Repository checkout used when --data-dir is omitted.",
    )
    parser.add_argument(
        "--repo-url",
        default=DEFAULT_REPO_URL,
        help="Prefiltered SAM-40 repository URL.",
    )
    parser.add_argument(
        "--data-commit",
        default=os.environ.get("SAM40_DATA_COMMIT", DEFAULT_DATA_COMMIT),
        help="Reviewed repository commit used for reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_results_hg_zfn"),
        help="Directory for tables, predictions, figures, and optional models.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--inner-val-subjects", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--selected-features", type=int, default=150)
    parser.add_argument("--figure-dpi", type=int, default=300)
    parser.add_argument(
        "--debug-folds",
        type=int,
        default=None,
        help=(
            "Run only the first N outer folds for debugging. Results from this "
            "mode are partial and must not be reported as LOSO performance."
        ),
    )
    parser.add_argument(
        "--save-fold-models",
        action="store_true",
        help="Save one final .keras model for every held-out subject.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.debug_folds is not None and not 1 <= arguments.debug_folds <= 40:
        raise ValueError("--debug-folds must be between 1 and 40.")

    config = ExperimentConfig(
        task=arguments.task,
        seed=arguments.seed,
        maximum_epochs=arguments.max_epochs,
        inner_validation_subjects=arguments.inner_val_subjects,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        selected_features=arguments.selected_features,
        figure_dpi=arguments.figure_dpi,
        debug_folds=arguments.debug_folds,
        save_fold_models=arguments.save_fold_models,
    )
    enable_deterministic_tensorflow()
    set_global_seed(config.seed)

    data_directory, dataset_revision = prepare_dataset(
        arguments.data_dir,
        arguments.clone_dir,
        arguments.repo_url,
        arguments.data_commit,
    )
    print(f"Dataset directory: {data_directory}")
    print(f"Dataset revision: {dataset_revision}")

    raw_trials, trial_labels, trial_subjects, _ = load_sam40_trials(
        data_directory, config.task
    )
    feature_windows, raw_windows, labels, subjects, trials = create_windows(
        raw_trials, trial_labels, trial_subjects
    )
    del raw_trials
    verify_no_outer_leakage(
        feature_windows, labels, subjects, trials
    )

    selected_models = (
        ["hg-zfn", "cnn", "dnn", "rnn"]
        if arguments.model == "all"
        else [arguments.model]
    )
    results: dict[str, dict[str, object]] = {}
    for model_name in selected_models:
        print(f"\nStarting {model_name.upper()} ...")
        results[model_name] = run_subject_independent_loso(
            model_name,
            feature_windows,
            raw_windows,
            labels,
            subjects,
            trials,
            config,
            arguments.output_dir,
        )
        summary = summarize_result(results[model_name], config.number_of_classes)
        print(
            f"{model_name.upper()}: accuracy={summary['accuracy']:.3f}, "
            f"balanced_accuracy={summary['balanced_accuracy']:.3f}, "
            f"macro_f1={summary['macro_f1']:.3f}, "
            f"roc_auc={summary['roc_auc']:.3f}"
        )

    save_experiment_outputs(
        results,
        config,
        arguments.output_dir,
        dataset_revision,
        vars(arguments),
    )
    print(f"\nResults saved to {arguments.output_dir.resolve()}")


if __name__ == "__main__":
    main()
