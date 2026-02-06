#!/usr/bin/env python3
"""
Prepare Feynman-I datasets for BACON benchmarking with multiple noise levels and standardized splits.

Usage examples:
  - From existing raw directory of Feynman CSV/TXT files:
      python samples/feynman/download_prepare.py --data-dir /path/to/feynman_raw
  - Try auto-download (best-effort; falls back to manual instructions if unavailable):
      python samples/feynman/download_prepare.py --auto-download
  - Place outputs outside the repo (either pass --output-root, or set BACON_DATA_DIR):
      python samples/feynman/download_prepare.py --data-dir /path/to/feynman_raw --output-root "D:/bacon-data/feynman"
    or set environment variable (used when --output-root not provided):
      set BACON_DATA_DIR=D:/bacon-data
      python samples/feynman/download_prepare.py --data-dir /path/to/feynman_raw
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import random
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import re
import math

try:
    import numpy as np
except Exception:
    np = None  # Fallback to Python's random if NumPy unavailable (slower)

try:
    import urllib.request as urllib_request
except Exception:
    urllib_request = None


DEFAULT_NOISE_LEVELS = [0.00, 0.01, 0.03, 0.05, 0.10]
DEFAULT_SPLIT = 0.75
DEFAULT_SEED = 42

# Best-known upstream sources for the Feynman data.
# Note: These can change; we try them in order. If all fail, we print manual instructions.
POTENTIAL_DATA_URLS = [
    # These URLs are known to change; the script will try each and report failures gracefully.
    # Users can override with --data-zip to a manually downloaded archive.
    # Add or update mirrors here if known.
    # "https://space.mit.edu/home/tegmark/aifeynman/Feynman_csv.zip",  # Example placeholder
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download/prepare Feynman-I dataset with noise variants and standardized splits.")
    src = p.add_argument_group("Source options")
    src.add_argument("--auto-download", action="store_true", help="Attempt to auto-download the dataset archive from known sources.")
    src.add_argument("--data-zip", type=Path, help="Path to a zip archive containing Feynman-I CSV/TXT files.")
    src.add_argument("--data-dir", type=Path, help="Directory containing Feynman-I CSV/TXT files (if already present).")

    out = p.add_argument_group("Output options")
    out.add_argument("--output-root", type=Path, default=None, help="Root output directory. If not set, uses BACON_DATA_DIR env var if present, else 'samples/feynman'.")
    out.add_argument("--dataset-name", type=str, default="FeynmanI", help="Top-level dataset folder name, default 'FeynmanI'.")

    prep = p.add_argument_group("Preparation options")
    prep.add_argument("--noise-levels", type=float, nargs="+", default=DEFAULT_NOISE_LEVELS, help="Noise levels (fraction of std(y)) to apply.")
    prep.add_argument("--split", type=float, default=DEFAULT_SPLIT, help="Train fraction for random split (e.g., 0.75).")
    prep.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for split and noise.")
    prep.add_argument("--max-samples-per-equation", type=int, default=None, help="Optional cap on samples per equation (for speed).")
    prep.add_argument("--overwrite", action="store_true", help="Re-generate outputs even if already present. Default: only process missing.")
    prep.add_argument("--synthesize-index-for-1col", action="store_true", help="If a file has only 1 numeric column, synthesize an index feature in [0,1] as X and treat the column as y.")

    return p.parse_args()


def resolve_output_root(cli_output_root: Optional[Path], dataset_name: str) -> Path:
    """
    Decide where to store outputs:
      1) If --output-root provided, use it.
      2) Else if BACON_DATA_DIR env var present, use <BACON_DATA_DIR>/<dataset_name_parent> (we still append dataset_name later).
      3) Else default to repo-relative 'samples/feynman'.
    """
    if cli_output_root is not None:
        return cli_output_root
    env_root = os.environ.get("BACON_DATA_DIR")
    if env_root:
        return Path(env_root) / "feynman"
    return Path("samples/feynman")


def ensure_output_dirs(root: Path, dataset_name: str) -> Path:
    dataset_dir = root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir


def try_auto_download(tmp_dir: Path) -> Optional[Path]:
    if urllib_request is None or not POTENTIAL_DATA_URLS:
        return None
    for url in POTENTIAL_DATA_URLS:
        try:
            dest = tmp_dir / "feynman.zip"
            print(f"Attempting to download: {url}")
            with urllib_request.urlopen(url, timeout=60) as resp, open(dest, "wb") as f:
                f.write(resp.read())
            # Quick sanity check
            if zipfile.is_zipfile(dest):
                print(f"Downloaded archive to: {dest}")
                return dest
            else:
                print(f"File at {dest} is not a valid zip.")
        except Exception as e:
            print(f"Download failed from {url}: {e}")
    return None


def extract_zip(archive_path: Path, extract_to: Path) -> Path:
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as z:
        z.extractall(extract_to)
    return extract_to


def find_equation_files(root: Path) -> List[Path]:
    """
    Locate candidate data files. Supports:
      - .csv, .txt, .tsv
      - Feynman files with dotted names like 'I.10.7', 'I.15.3x' (may appear to have a numeric extension)
    We assume one equation per file and skip directories.
    """
    candidates: List[Path] = []
    # First, known text extensions
    for ext in ("*.csv", "*.txt", "*.tsv"):
        candidates.extend(root.rglob(ext))
    # Also include typical Feynman entries (e.g., I.10.7, II.11.3, III.9.52, I.15.3x)
    for p in root.rglob("*"):
        if p.is_file():
            name = p.name
            if name.startswith(("I.", "II.", "III.")):
                candidates.append(p)
    # Deduplicate and sort
    uniq = sorted(set(candidates))
    return uniq


def read_table(path: Path) -> Tuple[List[str], List[List[float]]]:
    """
    Read a delimited table; robust to comma/semicolon/tab/space.
    Returns (header, rows). If no header, creates default x1..xk,y.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # First attempt: regex split on commas/semicolons/whitespace
    rows = [re.split(r"[,\s;]+", ln) for ln in lines]

    # Attempt header detection: if all cells convertible to float -> no header
    def is_float(s: str) -> bool:
        try:
            float(s)
            return True
        except Exception:
            return False

    if rows and not all(is_float(c) for c in rows[0] if c != ""):
        header = [c.strip() for c in rows[0]]
        data_rows = rows[1:]
    else:
        data_rows = rows
        if data_rows:
            ncols = len(data_rows[0])
        else:
            ncols = 0
        if ncols >= 2:
            header = [f"x{i+1}" for i in range(ncols - 1)] + ["y"]
        else:
            header = []

    # Convert
    numeric_rows: List[List[float]] = []
    for r in data_rows:
        if not r:
            continue
        try:
            parsed = [float(c) for c in r if c != ""]
            if len(parsed) > 0:
                numeric_rows.append(parsed)
        except Exception:
            # Skip lines that are not numeric
            continue

    return header, numeric_rows


def split_indices(n: int, train_frac: float, seed: int) -> Tuple[List[int], List[int]]:
    idx = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    n_train = int(round(train_frac * n))
    return idx[:n_train], idx[n_train:]


def compute_noise(y: List[float], level: float, seed: int) -> List[float]:
    if level <= 0:
        return y
    if np is not None:
        rng = np.random.default_rng(seed)
        y_arr = np.asarray(y, dtype=float)
        sigma = float(level) * float(y_arr.std(ddof=0)) if y_arr.size else 0.0
        noise = rng.normal(loc=0.0, scale=sigma, size=y_arr.shape)
        return (y_arr + noise).tolist()
    else:
        # Python stdlib fallback
        rnd = random.Random(seed)
        mean = sum(y) / len(y) if y else 0.0
        var = sum((v - mean) ** 2 for v in y) / len(y) if y else 0.0
        std = var ** 0.5
        sigma = level * std
        return [v + rnd.gauss(0.0, sigma) for v in y]


def write_csv(path: Path, header: List[str], rows: Iterable[Iterable[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(header)
        for r in rows:
            w.writerow(r)


def process_equation_file(
    dataset_dir: Path,
    eq_file: Path,
    noise_levels: List[float],
    split: float,
    seed: int,
    max_samples: Optional[int],
    overwrite: bool,
    synthesize_index_for_1col: bool,
) -> None:
    # Determine output dir name up front and optionally skip if already done
    eq_id_full = eq_file.name
    eq_id_sanitized = re.sub(r"[^A-Za-z0-9]+", "_", eq_id_full).strip("_")
    eq_out_dir = dataset_dir / eq_id_sanitized

    if not overwrite:
        # Check if all requested noise levels already exist with both train/test
        all_present = True
        for level in noise_levels:
            level_str = f"noise_{level:.2f}"
            if not (eq_out_dir / level_str / "train.csv").exists() or not (eq_out_dir / level_str / "test.csv").exists():
                all_present = False
                break
        if all_present:
            print(f"Skip (already prepared): {eq_id_sanitized}")
            return

    header, rows = read_table(eq_file)
    if not rows:
        print(f"Skipping empty/non-numeric file: {eq_file}")
        return
    ncols = len(rows[0])
    if ncols < 2:
        if ncols == 1 and synthesize_index_for_1col:
            # Treat the single column as y and synthesize x as normalized index
            y_only = [r[0] for r in rows]
            N = len(y_only)
            if N < 2:
                print(f"Skipping (too few rows to synthesize): {eq_file}")
                return
            x_idx = [i / (N - 1) for i in range(N)]  # [0,1]
            rows = [[x_idx[i], y_only[i]] for i in range(N)]
            ncols = 2
            header = ["x1", "y"]
            print(f"Synthesized index feature for 1-col file: {eq_file}")
        else:
            print(f"Skipping (needs >=2 columns): {eq_file}")
            return
    # Truncate if requested
    if max_samples is not None and len(rows) > max_samples:
        rows = rows[:max_samples]

    # Separate X and y
    X = [r[:-1] for r in rows]
    y = [r[-1] for r in rows]
    n = len(rows)
    train_idx, test_idx = split_indices(n, train_frac=split, seed=seed)

    # eq_out_dir already computed above
    num_x = ncols - 1
    x_header = header[:-1] if header and len(header) == ncols else [f"x{i+1}" for i in range(num_x)]
    y_header = [header[-1]] if header and len(header) == ncols else ["y"]
    full_header = x_header + y_header

    # Build matrices for indexing convenience
    def select_rows(indices: List[int], X_src: List[List[float]], y_src: List[float]) -> List[List[float]]:
        return [X_src[i] + [y_src[i]] for i in indices]

    # Precompute noisy y variants
    noisy_y_by_level = {level: compute_noise(y, level, seed=seed + int(level * 1e6)) for level in noise_levels}

    for level in noise_levels:
        y_noisy = noisy_y_by_level[level]
        train_rows = select_rows(train_idx, X, y_noisy)
        test_rows = select_rows(test_idx, X, y_noisy)
        level_str = f"{level:.2f}"
        out_dir = eq_out_dir / f"noise_{level_str}"
        write_csv(out_dir / "train.csv", full_header, train_rows)
        write_csv(out_dir / "test.csv", full_header, test_rows)
    print(f"Prepared: {eq_id_sanitized} -> levels={noise_levels}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    output_root = resolve_output_root(args.output_root, args.dataset_name)
    dataset_root = ensure_output_dirs(output_root, args.dataset_name)

    # Acquire raw files
    raw_dir: Optional[Path] = None
    cleanup_tmp = False
    try:
        if args.data_dir and args.data_dir.exists():
            raw_dir = args.data_dir
        elif args.data_zip and args.data_zip.exists():
            tmp = Path(tempfile.mkdtemp(prefix="feynman_"))
            extract_zip(args.data_zip, tmp)
            raw_dir = tmp
            cleanup_tmp = True
        elif args.auto_download:
            tmp = Path(tempfile.mkdtemp(prefix="feynman_dl_"))
            archive = try_auto_download(tmp)
            if archive is not None:
                extract_zip(archive, tmp / "extracted")
                raw_dir = tmp / "extracted"
                cleanup_tmp = True
            else:
                raw_dir = None
        else:
            raw_dir = None

        if raw_dir is None:
            print("\nCould not locate raw Feynman-I files automatically.")
            print("Please download the dataset archive manually from an official source (e.g., AI Feynman project page),")
            print("then run one of the following:")
            print("  python samples/feynman/download_prepare.py --data-zip /path/to/Feynman.zip")
            print("  python samples/feynman/download_prepare.py --data-dir /path/to/extracted_feynman")
            sys.exit(1)

        eq_files = find_equation_files(raw_dir)
        if not eq_files:
            print(f"No candidate files found under: {raw_dir}")
            sys.exit(1)

        print(f"Found {len(eq_files)} candidate equation files. Preparing splits and noise variants...")
        for eqf in eq_files:
            process_equation_file(
                dataset_dir=dataset_root,
                eq_file=eqf,
                noise_levels=args.noise_levels,
                split=args.split,
                seed=args.seed,
                max_samples=args.max_samples_per_equation,
                overwrite=args.overwrite,
                synthesize_index_for_1col=args.synthesize_index_for_1col,
            )
        print(f"\nDone. Output at: {dataset_root}")
    finally:
        if cleanup_tmp:
            # Best-effort cleanup
            try:
                import shutil
                shutil.rmtree(str(raw_dir.parent if raw_dir else ""), ignore_errors=True)  # type: ignore[arg-type]
            except Exception:
                pass


if __name__ == "__main__":
    main()

