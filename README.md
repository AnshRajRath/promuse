# ProMuse

A multimodal music recommendation research project focused on combining audio features with higher-level representations for richer recommendation experiments.

## Overview

ProMuse contains preprocessing scripts, notebooks, datasets, and experiment outputs for a music-AI workflow. The audio pipeline extracts features including MFCCs, chroma, spectral centroid, bandwidth, rolloff, spectral contrast, RMS energy, tempo, and duration, then stores processed feature tables for downstream modelling.

## Repository structure

```text
promuse/
├── data/
├── notebooks/
├── outputs/
├── scripts/
│   ├── domadp.py
│   ├── muext.py
│   ├── spext.py
│   └── tda.py
└── requirements.txt
```

## Getting started

```bash
git clone https://github.com/AnshRajRath/promuse.git
cd promuse
python -m venv .venv
pip install numpy pandas librosa
```

Prepare the expected dataset structure under `data/`, then run the relevant preprocessing scripts from `scripts/`.

## Research direction

The broader goal is to explore multimodal representation learning for music recommendation, combining complementary signals instead of relying on a single feature source.

## Author

**Ansh Raj Rath**  
GitHub: [@AnshRajRath](https://github.com/AnshRajRath)
