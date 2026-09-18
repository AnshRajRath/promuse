# ProMuse

> Prosody-aware multimodal music recommendation using speech emotion representations, music audio features, domain adaptation, and cross-modal learning.

## Overview

**ProMuse** explores a simple research question: can the way a person speaks help a recommendation system understand what kind of music may fit their current affective state?

Instead of treating speech and music as isolated modalities, the project builds parallel representation pipelines and brings them into a shared recommendation space. The repository contains dataset exploration, feature extraction, speech-emotion modelling, normalization, domain adaptation experiments, DANN training, and the foundations for a **Dual-Stream Cross-Modal Transformer** architecture.

## Research pipeline

```text
Speech audio
   |
   +--> acoustic / prosodic features
   +--> speech-emotion representation
   +--> normalization + domain adaptation
   |
   v
Speech embedding -------------------+
                                    |
                                    +--> cross-modal alignment
                                    |    and recommendation
                                    |
Music audio                         |
   |                                |
   +--> spectral / rhythmic features+
   +--> music representation
```

The broader design target is a two-stream model in which speech-derived affective information and music-derived audio information are learned separately, then aligned through cross-modal attention.

## What is in this repository

### Notebooks

| Notebook | Purpose |
|---|---|
| `1_1datasetexplore.ipynb` | Dataset exploration |
| `1_2featanalysis.ipynb` | Feature analysis |
| `1_3musicdata.ipynb` | Music-data preparation |
| `2_1speechemotionmodel.ipynb` | Speech-emotion modelling |
| `2_2speechnorm.ipynb` | Speech feature normalization |
| `2_3domainadp.ipynb` | Domain adaptation experiments |
| `2_4domainadv.ipynb` | Domain-adversarial learning |
| `2_5dannlamsweep.ipynb` | DANN lambda sweep |
| `2_6dannfinal.ipynb` | Final DANN experiment |
| `review1.ipynb` | Consolidated review / experiment notebook |

### Scripts

```text
scripts/
├── domadp.py    # domain adaptation utilities
├── muext.py     # music feature extraction
├── spext.py     # speech feature extraction
└── tda.py       # downstream analysis / adaptation utilities
```

## Audio features

The preprocessing pipeline works with a mixture of spectral, temporal, and rhythmic descriptors, including:

- MFCCs
- chroma
- spectral centroid
- spectral bandwidth
- spectral rolloff
- spectral contrast
- RMS energy
- tempo
- duration

These features form the classical audio-analysis layer used alongside learned representations.

## Domain adaptation

Speech-emotion datasets often differ in recording conditions, speakers, microphones, and corpus construction. ProMuse therefore includes **domain-adversarial neural network (DANN)** experiments so that the learned representation focuses less on dataset identity and more on transferable affective information.

The workflow includes:

```text
raw speech features
      |
      v
normalization
      |
      v
domain adaptation / DANN
      |
      v
domain-normalized speech representation
      |
      v
cross-modal recommendation stage
```

## Repository structure

```text
promuse/
├── data/
│   ├── metadata/
│   └── processed/
├── notebooks/
├── outputs/
├── scripts/
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt
```

Raw speech and music datasets are intentionally excluded from normal Git tracking. Large model artifacts are configured for Git LFS where applicable.

## Getting started

```bash
git clone https://github.com/AnshRajRath/promuse.git
cd promuse

python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Place the required datasets under the expected `data/` directories and run the notebooks in numerical order or execute the relevant preprocessing scripts directly.

## Research direction

The project is being developed toward a **Dual-Stream Cross-Modal Transformer** that combines:

1. a speech/prosody stream,
2. a music/audio stream,
3. domain-invariant affective representations,
4. cross-modal attention for alignment,
5. a recommendation layer operating in the shared representation space.

This makes ProMuse less about tagging songs with fixed moods and more about learning relationships between **how a user sounds** and **how music sounds**.

## Reproducibility notes

- Raw datasets are not bundled in the repository.
- Generated caches and temporary notebook artifacts should remain untracked.
- Large learned-model files should use Git LFS.
- Exact dataset paths may need to be adjusted locally depending on where the corpora are stored.

## Author

**Ansh Raj Rath**  
B.Tech CSE (Artificial Intelligence)  
GitHub: [@AnshRajRath](https://github.com/AnshRajRath)
