# 🧠 HG-ZFN for Leakage-Aware EEG Cognitive Stress Classification

Developed by **LAO — 2026**

This repository contains the implementation and evaluation of the **Hybrid Gated Zero-Dimension Feature Network (HG-ZFN)**, a hybrid temporal–spectral neural network developed for binary cognitive stress-task recognition from EEG signals. HG-ZFN extends the previously published Zero-Dimension Feature Network (ZFN) by combining raw EEG waveform learning with adaptively gated spectral-statistical descriptors. The proposed architecture is benchmarked against matched CNN, DNN, and RNN baselines.

The main experiment uses subject-independent **Leave-One-Subject-Out (LOSO)** cross-validation, train-only preprocessing, subject-disjoint validation, and trial-level prediction to reduce data leakage and provide a realistic estimate of performance on unseen participants.

## 📊 Dataset Information

This study uses the publicly available **SAM-40 EEG Stress Dataset** introduced by Ghosh et al. (2022). The dataset contains EEG recordings from **40 participants**, acquired through **32 channels at 128 Hz** while they completed four experimental conditions:

- Relaxation
- Stroop color-word test
- Mirror-image recognition
- Arithmetic task

Each condition contains three 25-second trials per participant, producing **480 EEG trials** in total. In the main binary task, relaxation trials are labeled **Relax**, while Stroop, mirror-image, and arithmetic trials are grouped as **Stress**.

- Dataset: [SAM-40 on Figshare](https://figshare.com/articles/dataset/SAM_40_Dataset_of_40_Subject_EEG_Recordings_to_Monitor_the_Induced-Stress_while_performing_Stroop_Color-Word_Test_Arithmetic_Task_and_Mirror_Image_Recognition_Task/14562090)
- Dataset article: [Ghosh et al., 2022](https://doi.org/10.1016/j.dib.2021.107772)
- Prefiltered EEG source used by the notebook: [wavesresearch/eeg_stress_detection](https://github.com/wavesresearch/eeg_stress_detection)

Please cite the original dataset publication and comply with the dataset license and usage requirements. The dataset should be downloaded from its official source rather than redistributed in this repository.

## 📬 Contact

For questions, research discussions, or collaborations, please contact:

📧 [laoktaviana@gmail.com](mailto:laoktaviana@gmail.com)
