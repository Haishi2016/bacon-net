# Scripts index — `lab/concept-bacon-cbm`

Reference for every script in this folder. Two lines of investigation live here:

1. **Fixed-tree OCBM** — human-authored BACON trees as a CBM head (see [`README.md`](README.md)).
2. **Emergent concepts** — *learn* K unnamed concepts + 10 trainable BACON trees jointly, then reverse-engineer what the concepts mean.

MNIST data auto-downloads to `../../benchmarks/mnist-addition/data`. On Windows invoke Python as `py -3`.

---

## Core model & config

| Script | What it is |
|---|---|
| [`config.py`](config.py) | The 9 human stroke concepts (`CONCEPTS`) + one BACON formula per digit (`DIGIT_RULES`). |
| [`bacon_logic.py`](bacon_logic.py) | AND/OR/NOT DSL parser + `BaconLogicBank` (fixed graded-logic trees). |
| [`model.py`](model.py) | `ConceptCNN` encoder, `ConceptBaconCBM` (fixed-tree head), `binarization_penalty`. |
| [`train.py`](train.py) | Train the fixed-tree OCBM on MNIST (task-only). Saves `checkpoint.pt`. `make_loaders` reused everywhere. |

## Synthetic data / probes

| Script | What it is |
|---|---|
| [`shapes.py`](shapes.py) | Synthetic shape generator: circle, ellipse, line, cross, square, rectangle, triangle, **corner, vee, zigzag** (turns). `generate(..., rotate=False)` for axis-aligned. |
| [`quickdraw.py`](quickdraw.py) | Google QuickDraw doodle loader (range-download + cache). |
| [`photo_sketch.py`](photo_sketch.py) | Photo → Sobel edge-sketch bridge to MNIST format. |

## Zero-shot transfer & concept-alignment evals (fixed-tree OCBM)

| Script | What it does |
|---|---|
| [`eval_shapes.py`](eval_shapes.py) | Apply the frozen "0" tree to synthetic shapes → circle-detection AUC. Has `load_model`, `roc_auc`. |
| [`eval_eight.py`](eval_eight.py) | Frozen "8" tree on stacked-circle scenes (`SCENES`). |
| [`eval_quickdraw.py`](eval_quickdraw.py) | Zero-shot "0" detector on QuickDraw doodles. |
| [`eval_photos.py`](eval_photos.py) | Zero-shot on real object photos (Caltech-101 → edge sketch). |
| [`eval_cream_zeroshot.py`](eval_cream_zeroshot.py) | OCBM vs CREAM zero-shot (defines `CREAMDigit`, `train_cream`). |
| [`eval_zeroshot_compare.py`](eval_zeroshot_compare.py) | **OCBM vs CBM vs CREAM**, "0" & "8" detectors on shapes/quickdraw/scenes (defines `CBMDigit`). |
| [`eval_concept_identification.py`](eval_concept_identification.py) | Theory-grounded concept identification (per-concept ROC-AUC = Somers' D, NMI, Hungarian best-permutation, DCI identity-rate) vs the 9 human strokes, on MNIST + shapes. |

## Emergent-concept discovery (learn concepts, then interpret)

| Script | What it does | Key flags |
|---|---|---|
| [`train_emergent_concepts.py`](train_emergent_concepts.py) | Train CNN → **K unnamed concepts** → **10 separate trainable `binaryTreeLogicNet` trees** (one/digit), task-only. Single run or `--scan` over K with multi-seed mean±std. | `--concepts K` / `--scan --scan-min --scan-max --seeds`; `--weight-mode trainable\|fixed`; `--save PATH`; `--epochs` |
| [`interpret_emergent_concepts.py`](interpret_emergent_concepts.py) | Reverse-engineer a trained model: concept↔stroke AUC naming, per-digit signatures/codes, **shape probe** (incl. turns), and **decoded per-digit trees** (routing, transforms, andness, softmax weights, gradient sensitivity). | `--concepts K --epochs`; `--weight-mode`; `--save PATH`; **`--load PATH`** (analyze a saved model, no retrain) |
| [`compare_weight_modes.py`](compare_weight_modes.py) | Trainable vs fixed node-weights across seeds: accuracy + entanglement metrics (mean \|pairwise concept correlation\|, shape-profile cosine). | `--concepts --seeds --epochs` |
| [`probe_c1_sweep.py`](probe_c1_sweep.py) | Parametric **closure sweeps** to pin a concept's meaning: gapped-circle (closed→open) + bending-line (line→closed polygon). Reads c0/c1/c2 curves. | `--load PATH` |
| [`probe_6v9_morph.py`](probe_6v9_morph.py) | 6-vs-9 tail-**curvature** morph (loop fixed, tail straight→curly); reads concepts + tree-6/tree-9 scores + predicted digit. | `--load PATH` |

### Typical emergent-concept workflow

```powershell
# 1. find the min number of concepts (multi-seed)
py -3 train_emergent_concepts.py --scan --scan-min 2 --scan-max 10 --seeds 5 --epochs 20

# 2. train + save a specific model, decoding its concepts and trees
py -3 interpret_emergent_concepts.py --concepts 3 --epochs 20 --seed 0 --save saved\k3_trainable.pt

# 3. re-analyze a saved model without retraining
py -3 interpret_emergent_concepts.py --load saved\k3_trainable.pt

# 4. pin down a concept's meaning with parametric sweeps
py -3 probe_c1_sweep.py   --load saved\k3_trainable.pt
py -3 probe_6v9_morph.py  --load saved\k3_trainable.pt

# 5. trainable vs fixed weights (entanglement test)
py -3 compare_weight_modes.py --concepts 3 --seeds 5 --epochs 20
```

## Outputs on disk

| Path | Contents |
|---|---|
| `saved/` | Saved emergent models (`k3_trainable.pt`, `k3_fixed.pt`): `{state_dict, K, weight_mode, seed, acc}`. Reload with `--load`. |
| `results/` | Decodes & comparisons (`k3_*_decode.txt`, `scan_5seed.txt`, `weightmode_compare.txt`). |
| `checkpoint.pt` | Fixed-tree OCBM from `train.py`. |

## Sub-directories

| Dir | Contents |
|---|---|
| `cream/` | CREAM reproduction (arXiv:2506.05014): `models.py` (BlackBox/CtrueY/SoftCBM/SoftCBMSC/CREAM/BaconCBM), `run_cream.py`, FashionMNIST concepts. |
| `table/` | Paper-table benchmark suite: one `<model>-<dataset>-accuracy.py` per cell (MNIST/iFMNIST/cFMNIST/CUB/CelebA), shared `_bench.py`/`_fmnist.py`/`_cub.py`/`_celeba.py`, `_gltree.py`, and the leakage suite (`_leakage.py`, `leakage.py`). |
