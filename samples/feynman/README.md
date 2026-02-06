## Feynman symbolic regression samples

This folder hosts locally prepared datasets based on the canonical Feynman-I collection (100 equations) that is widely used in recent symbolic regression papers and in SRBench. Each equation is stored in its own subfolder, and within it, multiple noise-level variants are provided with a standardized 75/25 train/test split for comparability.

Folder layout (created by the script):

- `samples/feynman/FeynmanI/<equation_id>/noise_<level>/train.csv`
- `samples/feynman/FeynmanI/<equation_id>/noise_<level>/test.csv`

Where:
- `<equation_id>` is the source file base name (e.g., `feynman_I_6_2`).
- `<level>` is one of: `0.00`, `0.01`, `0.03`, `0.05`, `0.10` (0%–10% additive Gaussian noise on y, scaled by std(y)).

### How to prepare data

1) If you already have the original Feynman-I CSV/text files, run:

```bash
python samples/feynman/download_prepare.py --data-dir /path/to/feynman_raw --output-root /absolute/path/outside/repo
```

2) Otherwise, try automatic download (may require internet access and that the upstream URL is available):

```bash
python samples/feynman/download_prepare.py --auto-download --output-root /absolute/path/outside/repo
```

If automatic download fails, the script will print instructions for manual download and how to point `--data-zip` or `--data-dir` to the archive/location.

Note: The common `Feynman_without_units` archive contains files without extensions (e.g., `I.10.7`). The script detects these and will create sanitized subfolder names like `I_10_7`.

If some files contain only a single numeric column (rare), you can optionally synthesize an index input (x in [0,1]) and treat the column as y:

```bash
python samples/feynman/download_prepare.py --data-dir /path/to/feynman_raw --synthesize-index-for-1col
```

### Defaults and conventions

- Dataset: Feynman-I (100 equations), chosen for broad use in recent literature and SRBench.
- Split: 75% train / 25% test, with a fixed seed for reproducibility.
- Noise: additive Gaussian on the target y only, with σ = level × std(y) computed per equation on the full dataset prior to the split.
- Levels: 0.00, 0.01, 0.03, 0.05, 0.10 (override via `--noise-levels`).

### Storing data outside this repository

- Pass `--output-root` to choose any folder (e.g., a data drive).
- Or set environment variable `BACON_DATA_DIR` and omit `--output-root`; the script will write to `<BACON_DATA_DIR>/feynman/FeynmanI/...`.

### Example

```bash
python samples/feynman/download_prepare.py ^
  --auto-download ^
  --output-root D:/bacon-data/feynman ^
  --noise-levels 0.0 0.01 0.03 0.05 0.10 ^
  --split 0.75 ^
  --seed 42
```

The output can be directly fed into BACON for training/evaluation across noise levels.

