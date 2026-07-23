"""
Lightweight loader for Google "Quick, Draw!" numpy bitmaps.

Quick, Draw! is a well-known dataset of ~50M doodles of everyday objects across
345 categories (clock, wheel, donut, ladder, envelope, ...).  The `numpy_bitmap`
files are 28x28 grayscale renderings in the SAME format as MNIST (white strokes
on a black background), so the MNIST-trained concept encoder can be applied
directly with no retraining.

We only need a few hundred images per category, so instead of downloading the
full ~40-100 MB `.npy` files we issue an HTTP range request for just the header
plus the first N images (~a few hundred KB) and cache them locally.

    https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/<cat>.npy
"""

from __future__ import annotations

import ast
import os
import urllib.error
import urllib.parse
import urllib.request

import torch

BASE = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"
_MNIST_MEAN = 0.1307
_MNIST_STD = 0.3081
_IMG_BYTES = 28 * 28


def _get_range(url: str, start: int, end: int, retries: int = 3) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    last = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError:
            raise
        except Exception as e:  # transient network / timeout -> retry
            last = e
    raise last


def load_category(name: str, n: int = 500, cache_dir: str = None) -> torch.Tensor:
    """Return the first ``n`` drawings of a category as (m,1,28,28), MNIST-normalized."""
    cache_dir = cache_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "quickdraw_data")
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"{name.replace(' ', '_')}_{n}.pt")
    if os.path.exists(cache):
        return torch.load(cache)

    url = BASE.format(urllib.parse.quote(name))
    head = _get_range(url, 0, 9)
    if head[:6] != b"\x93NUMPY":
        raise ValueError(f"{name}: not a .npy file")
    header_len = int.from_bytes(head[8:10], "little")
    data_off = 10 + header_len

    raw = _get_range(url, 0, data_off + n * _IMG_BYTES - 1)
    meta = ast.literal_eval(raw[10:data_off].decode("latin1").strip())
    total = meta["shape"][0]
    m = min(n, total)
    buf = raw[data_off:data_off + m * _IMG_BYTES]

    t = torch.frombuffer(bytearray(buf), dtype=torch.uint8).float().reshape(m, 1, 28, 28)
    t = t / 255.0
    t = (t - _MNIST_MEAN) / _MNIST_STD
    torch.save(t, cache)
    return t


if __name__ == "__main__":
    for cat in ["clock", "wheel", "donut", "ladder", "envelope"]:
        try:
            t = load_category(cat, n=64)
            raw = t * _MNIST_STD + _MNIST_MEAN
            print(f"{cat:10s} {tuple(t.shape)}  mean {raw.mean():.3f} "
                  f"min {raw.min():.2f} max {raw.max():.2f}")
        except Exception as e:
            print(f"{cat:10s} ERR {e}")
