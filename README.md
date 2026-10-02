# 🧠 EEG Stress Classification — Comparing 5 Models Across Two Datasets

Developed by **ZAHRA — 2026**

This repository contains my experiments adapting and comparing five deep learning architectures for binary stress classification (Relax vs. Stress) from EEG signals, evaluated with a leakage-aware, subject-independent Leave-One-Subject-Out (LOSO) cross-validation scheme (train-only preprocessing, subject-disjoint validation).

Models compared
HG-ZFN — Hybrid Gated Zero-Dimension Feature Network
ZFN — the original, lightweight feature-vector architecture
CNN, DNN, RNN — baseline comparisons

The HG-ZFN/CNN/DNN/RNN architectures and the LOSO pipeline were developed by Laily Ade Oktaviana (original repository); the ZFN architecture was adapted from her EEG_Brainwave repository. The code in this repository is my own adaptation of that pipeline to run all five models on a second dataset and compare the results.

Datasets

The first dataset is SAM-40 (Ghosh et al., 2022), which records 32-channel EEG at 128 Hz from 40 participants across four conditions — relaxation, the Stroop color-word test, a mirror-image recognition task, and an arithmetic task — with three 25-second trials per condition. The second dataset is the more recent Cognitive Load Assessment Through EEG dataset (Nirabi et al., 2025), which records 8-channel OpenBCI EEG from 15 participants performing Stroop and arithmetic tasks at four cognitive load levels; these levels were mapped to the same binary Relax/Stress labeling used for SAM-40 so that results from both datasets could be compared directly under the same five models and the same LOSO evaluation pipeline.

How to Run

pip install numpy pandas scikit-learn mne mne_features tensorflow keras keras-tuner matplotlib seaborn scipy
python 2026_CogLoad.py --model <hg-zfn|cnn|dnn|rnn|zfn|all> --data-dir <path-to-data> --output-dir <output-folder-name>
python 2026_SAM40_zfn.py --model <hg-zfn|cnn|dnn|rnn|zfn|all> --data-dir <path-to-data> --output-dir <output-folder-name> 

Author

Prepared by Zahra Ramadhina [Biomedical Engineering, Telkom University] as part of a university assignment/thesis. Thank You!  
