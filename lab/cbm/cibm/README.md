# Concepts' Information Bottleneck Models

**[Karim Galliamov](https://scholar.google.com/citations?user=tBhN7ecAAAAJ&hl=en)$^1$, [Syed M Ahsan Kazmi](https://people.uwe.ac.uk/Person/AhsanKazmi)$^2$, [Adil Khan](https://www.hull.ac.uk/staff-directory/adil-khan)$^3$, [Adín Ramírez Rivera](https://www.mn.uio.no/ifi/english/people/aca/adinr/)$^4$**

**${}^1\underset{\text{}}{\text{University of Amsterdam}}$** $\hspace{1em}$
**${}^2\underset{\text{}}{\text{University of the West of England, Bristol}}$** $\hspace{1em}$
**${}^3\underset{\text{}}{\text{University of Hull}}$** $\hspace{1em}$
**${}^4\underset{\text{Department of Informatics}}{\text{University of Oslo}}$**

[![Website](https://img.shields.io/badge/Website-green)](https://dsb-ifi.github.io/cibm/)
[![Paper](https://img.shields.io/badge/Paper-ICLR_2026-blue)](https://openreview.net/forum?id=JGIYfwaNpT)
[![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/dsb-ifi/cibm)

## Abstract

Concept Bottleneck Models (CBMs) aim to deliver interpretable predictions by
routing decisions through a human-understandable concept layer, yet they often
suffer reduced accuracy and concept leakage that undermines faithfulness. We
introduce an explicit Information Bottleneck regularizer on the concept layer that
penalizes $I(X; C)$ while preserving task-relevant information in $I(C; Y)$, encouraging minimal-sufficient concept representations. We derive two practical variants
(a variational objective and an entropy-based surrogate) and integrate them into
standard CBM training without architectural changes or additional supervision.
Evaluated across six CBM families and three benchmarks, the IB-regularized models consistently outperform their vanilla counterparts. Information-plane analyses
further corroborate the intended behavior. These results indicate that enforcing a
minimal-sufficient concept bottleneck improves both predictive performance and
the reliability of concept-level interventions. The proposed regularizer offers a
theoretic-grounded, architecture-agnostic path to more faithful and intervenable
CBMs, resolving prior evaluation inconsistencies by aligning training protocols
and demonstrating robust gains across model families and datasets.


## CIBM: Concepts' Information Bottleneck Models

This repo contains code for **Concepts' Information Bottleneck Models**, accepted at ICLR 2026.

For an introduction to our work, visit the [project webpage](https://dsb-ifi.github.io/cibm/).


## Installation

**1. Clone the repository:**

```bash
git clone https://github.com/dsb-ifi/cibm.git
cd cibm
```

**2. Create a conda environment and install dependencies:**

```bash
conda create -n cibm python=3.10 -y
conda activate cibm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Data

Download the [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) dataset and place it at `../data/CUB/CUB_200_2011/` relative to the repo root. The directory should contain `train.pkl`, `val.pkl`, `test.pkl`, and the `images/` folder.

For AwA2 experiments, download the [xlsa17](https://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip) splits and the [AwA2 predicates](https://cvml.ista.ac.at/AwA2/) and place them at `../data/xlsa17/data/AWA2/` and `../data/AwA2-data/Animals_with_Attributes2/` respectively.


## Training

### Quick start

Train a basic CBM on CUB (backbone frozen, using Inception V3 embeddings):

```bash
python ./src/train.py \
    --model_arch '[2048]' \
    --dataset_name=CUB \
    --is_stochastic=False \
    --train_backbone=False \
    --epochs=100 \
    --lr=0.0003
```

### Reproducing checkpoints

We provide a script that trains a **Basic CBM**, an **IB-CBM (variational)**, and an **IB-CBM (entropy surrogate)** sequentially on CUB-200-2011:

```bash
bash train_cub_demo.sh
```

Checkpoints are saved to `logs/<timestamp>/model.pth`. Each run also produces intervention curves, loss plots, and a summary in its log directory.

### Key flags

| Flag | Description |
|---|---|
| `--is_stochastic` | `True` for stochastic (IB-capable) encoder, `False` for deterministic CBM |
| `--use_HC` | Use entropy-based surrogate $H(C)$ instead of variational MI bound |
| `--beta` | Initial Lagrange multiplier for the IB constraint |
| `--beta_lr` | Learning rate for dual Lagrangian update (set `0.0` to fix $\beta$) |
| `--train_backbone` | Fine-tune the ResNet-50 backbone end-to-end |
| `--dataset_name` | `CUB` or `AWA2_emb` |
| `--log_base` | Base directory for saving logs and checkpoints (default: `logs`) |
| `--measure_intervention` | Run test-time intervention evaluation after training |
| `--measure_robustness` | Run noise-robustness evaluation after training |


## Citation

If you find our work useful, please consider citing our paper.

```bibtex
@inproceedings{galliamov2026,
  title={Concepts' Information Bottleneck Models},
  author={Galliamov, Karim and Kazmi, Syed M Ahsan and Khan, Adil and Ram\'irez Rivera, Ad\'in},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=JGIYfwaNpT}
}
```
