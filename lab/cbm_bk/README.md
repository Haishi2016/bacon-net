# CBM Benchmarks

A small, extensible harness for benchmarking **Concept Bottleneck Models
(CBMs)** across multiple datasets. Datasets and models are registered
independently, so any `(dataset, model)` pair can be trained and evaluated
through a single config-driven driver.

Currently included:

| Dataset | Config | Model |
| ------- | ------ | ----- |
| CUB-200-2011 | `configs/cub_baseline.yaml` | `baseline_cbm` (joint CBM) |

## Folder layout

```
lab/cbm/
  README.md
  train.py                 # training driver
  evaluate.py              # evaluation driver
  configs/
    cub_baseline.yaml       # one file per (dataset, model) experiment
  cbm_bench/
    registry.py             # name -> factory tables for datasets & models
    config.py               # YAML -> ExperimentConfig
    engine.py               # shared train/eval loops
    metrics.py              # task & concept accuracy
    datasets/
      base.py               # DatasetBundle contract
      cub.py                # CUB-200-2011 loader  (@register_dataset("cub"))
    models/
      base.py               # CBMOutput contract
      baseline_cbm.py        # joint CBM  (@register_model("baseline_cbm"))
```

Dataset image files are **not** stored here. They live in a shared `datasets/`
folder outside `lab/cbm` (default: `lab/datasets/`), keeping large data out of
the experiment code.

## 1. Get the data

Download and extract **CUB-200-2011** into the shared datasets folder so the
final layout matches the default config root (`lab/datasets/CUB_200_2011`):

```powershell
# from the repo root (c:\School\bacon-net)
mkdir lab\datasets -Force
cd lab\datasets

# Download the official release (1.1 GB) from Caltech Vision:
#   https://www.vision.caltech.edu/datasets/cub_200_2011/
# (file: CUB_200_2011.tgz). Then extract it:
tar -xzf CUB_200_2011.tgz
```

Expected resulting structure:

```
lab/datasets/CUB_200_2011/
  images/
  images.txt
  image_class_labels.txt
  train_test_split.txt
  classes.txt
  attributes/
    image_attribute_labels.txt
```

> The official release ships **312** binary attributes (concepts) and **200**
> classes. By default the loader **denoises** concepts to class-level majority
> votes and **filters** rare attributes (`min_class_count: 10` → ~93 concepts),
> matching the standard CBM recipe. Set `concept_mode: instance` and
> `min_class_count: 0` to use the raw per-image 312 attributes instead.

## 2. Configure

Edit `configs/cub_baseline.yaml` (or copy it to a new file). Key fields:

- `dataset.root` — path to the extracted CUB folder (relative paths resolve
  against the config file's directory).
- `model.backbone` — `resnet18` | `resnet34` | `resnet50`.
- `model.bottleneck` — `true` makes the task head read only the concept layer.
- `model.concept_loss_weight` — weight of concept BCE vs. task cross-entropy.
- `train.*` — epochs, batch size, optimizer, learning-rate schedule.

## 3. Install dependencies

```powershell
pip install torch torchvision pyyaml pillow numpy
```

## 4. Train

```powershell
# from the repo root
python lab\cbm\train.py --config lab\cbm\configs\cub_baseline.yaml
```

Outputs are written to `runs/<experiment>/`:

- `best.pt` — checkpoint with the best eval task accuracy
- `last.pt` — final-epoch checkpoint
- `history.json` — per-epoch train/eval metrics

## 5. Evaluate

```powershell
python lab\cbm\evaluate.py `
  --config lab\cbm\configs\cub_baseline.yaml `
  --checkpoint runs\cub_baseline\best.pt
```

Reports test **task accuracy** (class prediction) and **concept accuracy**
(mean per-concept binary accuracy).

## Adding a new dataset

1. Create `cbm_bench/datasets/<name>.py`.
2. Implement a `torch.utils.data.Dataset` yielding `(image, concepts, label)`.
3. Add a factory decorated with `@register_dataset("<name>")` returning a
   `DatasetBundle`.
4. Import the module in `cbm_bench/datasets/__init__.py`.
5. Point a config's `dataset.name` at `<name>`.

## Adding a new model

1. Create `cbm_bench/models/<name>.py`.
2. Implement an `nn.Module` whose `forward` returns a `CBMOutput`
   (`concept_logits`, `class_logits`).
3. Add a factory decorated with `@register_model("<name>")` accepting
   `(cfg, n_concepts, n_classes)`.
4. Import the module in `cbm_bench/models/__init__.py`.
5. Point a config's `model.name` at `<name>`.

No changes to `train.py` / `evaluate.py` are needed — they only talk to the
registries.
