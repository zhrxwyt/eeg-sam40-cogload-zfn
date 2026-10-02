# Catatan/komentar asli dari `2026_CogLoad.py`

- Baris 1: `#!/usr/bin/env python3`
- Baris 52: `# headless backend; avoids Tkinter/Tcl dependency`
- Baris 90: `# NOTE: adapted for the "Cognitive Load Assessment Through EEG" dataset`
- Baris 91: `# (Nirabi et al., Data in Brief, 2025; Mendeley kt38js3jv7) instead of SAM-40.`
- Baris 95: `# OpenBCI Cyton recording rate for this dataset`
- Baris 97: `# 2-second window at 250 Hz (trials are only 10-20s long)`
- Baris 98: `# 1-second hop`
- Baris 101: `# 8-channel OpenBCI Cyton montage used by this dataset (per dataset paper).`
- Baris 111: `# Binary task only: natural (baseline) = Relax(0), any load level = Stress(1).`
- Baris 117: `# 2 tasks (Arithmetic, Stroop) x 4 levels, confirmed by run log`
- Baris 155: `# pragma: no cover - depends on TF build`
- Baris 229: `# Cognitive Load filenames look like "natural-9.txt", "lowlevel-3.txt".`
- Baris 274: `# skip any stray header/metadata line`
- Baris 278: `# trial too short to form even one window`
- Baris 282: `# time x channel, matches SAM-40 loader's orientation`