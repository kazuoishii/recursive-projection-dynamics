# Recursive Projection Dynamics of Neural Representations

Code and reported results for:

**Recursive Projection Dynamics of Neural Representations: Geometry, Dimensionality, and Task Information**  
Kazuo Ishii and Bishnu Prasad Gautam  
NeurIPS 2026 Workshop on Symmetry and Geometry in Neural Representations (NeurReps), Extended Abstract Track.

**arXiv:** pending  
**OpenReview:** submission #47

## Overview

This repository reproduces the CIFAR-10 / ResNet-18 proof-of-concept experiment used to analyze neural representations as trajectories of successive transformations. The analysis jointly tracks:

1. adjacent-layer representation geometry using linear centered kernel alignment (CKA), reported as geometric change `1 - CKA`;
2. participation-ratio effective dimensionality; and
3. linear accessibility of task information using a ridge linear probe.

The main observed transition is `layer3 -> layer4`: geometric change is approximately **0.411**, effective dimensionality decreases from **19.27** to **9.66**, while linear-probe test accuracy increases from **88.27%** to **92.98%**.

## Repository structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── src/
│   ├── train_resnet18.py
│   ├── extract_activations.py
│   ├── compute_cka.py
│   ├── compute_effective_dimension.py
│   ├── compute_linear_probe.py
│   └── make_figure.py
├── results/
│   ├── cka_results.csv
│   ├── effective_dimension_results.csv
│   ├── linear_probe_results.csv
│   └── combined_results.csv
├── figures/
│   ├── representation_dynamics.png
│   └── representation_dynamics.pdf
└── paper/
    └── Recursive_Projection_Dynamics_arXiv.pdf
```

Large model checkpoints and extracted activations are intentionally excluded from version control. They can be regenerated from the scripts below.

## Environment

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The code automatically uses CUDA when available, then Apple MPS, otherwise CPU.

## Reproduction

Run all commands from the repository root.

### 1. Train CIFAR-10 ResNet-18

The paper uses a CIFAR-10 adaptation of ResNet-18: the initial 7x7 stride-2 convolution is replaced by a 3x3 stride-1 convolution, and initial max pooling is removed. Training uses SGD with momentum 0.9, weight decay 5e-4, an initial learning rate of 0.1, cosine annealing, batch size 128, and 30 epochs.

```bash
python src/train_resnet18.py --data-dir data --download
```

This produces `resnet18_final.pt`. The checkpoint used for the reported experiment reached **93.09%** CIFAR-10 test accuracy.

### 2. Extract representations

```bash
python src/extract_activations.py \
  --data-dir data \
  --model resnet18_final.pt \
  --output-dir activations_final
```

Representations are extracted from `conv1`, `layer1`, `layer2`, `layer3`, `layer4`, and `avgpool`. Convolutional feature maps are globally averaged to one feature vector per image. Both the 50,000-image training split and 10,000-image test split are saved.

### 3. Compute adjacent-layer CKA

```bash
python src/compute_cka.py
```

Output: `results/cka_results.csv`.

### 4. Compute effective dimensionality

```bash
python src/compute_effective_dimension.py
```

Output: `results/effective_dimension_results.csv`.

### 5. Fit ridge linear probes

```bash
python src/compute_linear_probe.py
```

The probe is trained on all 50,000 training representations and evaluated on the 10,000 test representations. Ridge parameter: `1.0`.

Output: `results/linear_probe_results.csv`.

### 6. Regenerate Figure 1

```bash
python src/make_figure.py
```

Outputs:

- `figures/representation_dynamics.png`
- `figures/representation_dynamics.pdf`

## Reported results

| Layer | Geometric change from previous layer | Effective dimension | Linear-probe test accuracy |
|---|---:|---:|---:|
| conv1 | — | 2.525 | 32.64% |
| layer1 | 0.3418 | 5.586 | 52.33% |
| layer2 | 0.1335 | 10.289 | 68.63% |
| layer3 | 0.2724 | 19.269 | 88.27% |
| layer4 | 0.4112 | 9.658 | 92.98% |
| avgpool | 0.0000 | 9.658 | 92.98% |

The values above are stored in `results/combined_results.csv`.

## Reproducibility note

Training a neural network can exhibit small platform-dependent variation even with a fixed seed. The CSV files and figure included here are the outputs used for the reported manuscript experiment. Model checkpoints and full activation tensors are not included because of their size.

## Citation

The arXiv identifier will be added after public release.

```bibtex
@misc{ishii2026recursive,
  title  = {Recursive Projection Dynamics of Neural Representations: Geometry, Dimensionality, and Task Information},
  author = {Ishii, Kazuo and Gautam, Bishnu Prasad},
  year   = {2026},
  note   = {Submitted to the NeurIPS 2026 Workshop on Symmetry and Geometry in Neural Representations (NeurReps), Extended Abstract Track}
}
```

## License

The software in this repository is released under the MIT License. The manuscript may be subject to its separate arXiv / publication license.
