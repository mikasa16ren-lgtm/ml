# ML Workspace

Empty in Phase 0 by design — populated in Phase 1 (ML).

```
ml/
├── src/                  # importable modules — the real implementation
│   ├── config.py            # single source of truth for every path/hyperparameter
│   ├── utils/                # Kaggle detection, dataset auto-discovery, seeding
│   ├── data/                   # per-dataset discovery + metadata builders, subset/split
│   ├── models/                 # ViT model wrappers (Phase: Image/Signature/Video notebooks)
│   ├── training/                 # training loops
│   ├── evaluation/                # metrics, confusion matrices, classification reports
│   └── xai/                        # Grad-CAM / attention rollout
├── notebooks/            # executable Jupyter notebooks — thin wrappers around src/
│   └── 01_Phase_0_Dataset_Preparation.ipynb   # ✅ implemented
├── datasets/
│   ├── images/       # small subset from the 140k Real and Fake Faces dataset
│   ├── videos/        # small subset from the DFD dataset
│   ├── signatures/     # small subset from the real/fake signature dataset
│   └── metadata/        # CSVs written by Notebook 01 (train/val/test splits, etc.)
├── training/            # (legacy Phase-0 placeholder folder, superseded by src/training/)
└── checkpoints/          # saved fine-tuned ViT weights (.pt/.pth), gitignored
```

### Why both `src/` and `notebooks/`?

Notebooks import everything from `src/` rather than containing the real logic inline.
This means:
- The same dataset-discovery/training/evaluation code runs identically whether it's
  called from a notebook cell or a plain Python script.
- Bugs get fixed in one place (`src/`), not copy-pasted across 6 notebooks.
- Notebooks stay focused on *narrative* (what's happening and why) while `src/`
  holds the *implementation*.

### Notebook 01 — Phase 0: Dataset Preparation (implemented)

Run this first, either in a Kaggle Notebook (with all three datasets added via
"+ Add Input") or locally (after downloading small subsets per the commands below).
It auto-detects the environment, locates all three datasets, builds a balanced
4,000-image subset (2,000 real + 2,000 fake) with an 80/10/10 stratified split,
runs the same discovery pass for signatures (with an honest label-reliability
report — it never assumes unlabeled folders mean what we hope) and videos
(50 real + 50 fake, paths only, no decoding), and saves everything to
`ml/datasets/metadata/*.csv`.

## Dataset sources

- AI vs Human images: https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset
- Deepfake videos: https://www.kaggle.com/competitions/deepfake-detection-challenge/data
- Signatures: https://www.kaggle.com/datasets/emrahaydemr/realfake-signature-datasets

Download **small subsets only** for local development on the constrained laptop
(8 GB RAM, GTX GPU). Use Google Colab/Kaggle GPU for full-scale training and
copy the resulting checkpoint back into `checkpoints/`.

## Constraints to respect in Phase 1

- Pretrained ViT as the base (don't train from scratch)
- 224×224 input resolution
- Batch size 2–4
- 3–5 initial epochs
- Sample frames from video rather than decoding every frame
- Stream/batch-load data — never load a full dataset into RAM
