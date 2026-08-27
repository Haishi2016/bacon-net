"""
Information-leakage measurement across the table models.

For every trained concept model we evaluate three task-accuracy quantities on the
test set, plus the ground-truth concept ceiling:

  ceiling : C_true -> Y accuracy      (info ceiling of the ground-truth concepts)
  full    : task accuracy, side channel ON
  cpath   : task accuracy, concept-only path (side channel OFF)
  cpath_h : concept-only path with HARD (binarized) concepts fed to the same head

From these we report three complementary leakage measures (all in acc points):

  leak_ceiling = max(cpath  - ceiling, 0)   total leak past the concept set's info
  leak_side    =     full   - cpath          explicit side-channel contribution
  leak_soft    =     cpath  - cpath_h         info smuggled in continuous concept
                                              values (soft-vs-hard concept leak)

``leak_soft`` is defined even when the concept set is complete (ceiling = 100),
so it distinguishes the CUB / cFMNIST columns where ``leak_ceiling`` is trivially
zero.  ``ocbm`` rows should show ~0 on all three (leak-free-by-design), while
``cbm`` leaks through soft values and ``cbm+sc`` / ``cream`` leak through the
side channel.
"""

from __future__ import annotations

import statistics

import torch

import _bench                                                   # sets sys.path

# concept-loss weight per model family (matches the per-cell defaults)
LAM = {"cbm": 0.5, "cbm+sc": 0.5, "cream": 1.0, "cream-wo-sc": 1.0,
       "ocbm": 1.0, "ocbm-ft": 1.0}

# models evaluated for leakage (ctruey handled separately as the ceiling)
MODELS = ["blackbox", "cbm", "cbm+sc", "cream-wo-sc", "cream", "ocbm", "ocbm-ft"]

LABELS = {"blackbox": "Black-box", "cbm": "CBM", "cbm+sc": "CBM+SC",
          "cream-wo-sc": "CREAM w/o SC", "cream": "CREAM",
          "ocbm": "OCBM", "ocbm-ft": "OCBM (fine-tuned)"}


@torch.no_grad()
def leak_eval(model, loader, device, has_side, has_c, cacc_fn, unpack):
    """Return dict(full, cpath, cpath_h, cacc) for one trained model.

    ``unpack(batch, device) -> (x, c_true_or_None, y)``.
    ``cacc_fn(cprobs, c_true) -> float`` (mean concept accuracy for the batch).
    """
    model.eval()
    full = cp = cph = n = 0
    cs = cn = 0
    for batch in loader:
        x, ctrue, y = unpack(batch, device)
        logits, cprobs = model(x)
        full += (logits.argmax(1) == y).sum().item()
        n += y.numel()
        lcp = model(x, use_side=False)[0] if has_side else logits
        cp += (lcp.argmax(1) == y).sum().item()
        if has_c and cprobs is not None:
            lph = model(x, use_side=False, harden=True)[0]
            cph += (lph.argmax(1) == y).sum().item()
            cs += cacc_fn(cprobs, ctrue) * y.numel()
            cn += y.numel()
        else:
            cph += (lcp.argmax(1) == y).sum().item()
    return {"full": full / n, "cpath": cp / n, "cpath_h": cph / n,
            "cacc": (cs / cn) if cn else float("nan")}


# --------------------------------------------------------------------------- #
# per-dataset runners
# --------------------------------------------------------------------------- #
def run_fmnist(dataset, iters, epochs=20, seed=0):
    import _fmnist
    from run_cream import (concept_accuracy, eval_ctruey, load_fmnist,
                           train_ctruey, train_model)
    spec = _fmnist.SPECS[dataset]
    device = _bench.get_device()
    tl, vl = load_fmnist(_fmnist._DATA, 256)

    def unpack(batch, dev):
        x, y = batch
        x, y = x.to(dev), y.to(dev)
        return x, spec.concept_targets(y), y

    def cacc_fn(cprobs, ctrue):
        return concept_accuracy(cprobs, ctrue, spec)

    results = {}
    for it in range(iters):
        _bench.set_seed(seed + it)
        ctm = train_ctruey(spec, tl, device)
        ceiling = eval_ctruey(ctm, vl, spec, device)
        for key in MODELS:
            _bench.set_seed(seed + it)
            build, has_c, has_side = _fmnist._MODELS[key]
            model = build(spec)
            train_model(model, tl, spec, device, epochs, LAM.get(key, 1.0), has_c)
            m = leak_eval(model, vl, device, has_side, has_c, cacc_fn, unpack)
            m["ceiling"] = ceiling
            results.setdefault(key, []).append(m)
        _report_iter(dataset, it, iters, ceiling, results)
    return results


def run_cub(iters, epochs=40, seed=0, num_workers=4):
    import _cub
    from torch.utils.data import DataLoader
    from run_cream import concept_accuracy
    device = _bench.get_device()
    spec = _cub.build_spec("positive")
    tl = DataLoader(_cub._CUBImages("train", True), batch_size=64, shuffle=True,
                    num_workers=num_workers, pin_memory=True)
    vl = DataLoader(_cub._CUBImages("test", False), batch_size=128, shuffle=False,
                    num_workers=num_workers, pin_memory=True)

    def unpack(batch, dev):
        img, c, y = batch
        return img.to(dev), c.to(dev), y.to(dev)

    def cacc_fn(cprobs, ctrue):
        return concept_accuracy(cprobs, ctrue, spec)

    results = {}
    for it in range(iters):
        ceiling = _cub._run_ctruey(spec, device, 1, seed=seed + it)[0]
        for key in MODELS:
            _bench.set_seed(seed + it)
            model, has_c, has_side = _cub._build_model(key, spec)
            _cub._train(model, tl, device, epochs, LAM.get(key, 1.0), has_c)
            m = leak_eval(model, vl, device, has_side, has_c, cacc_fn, unpack)
            m["ceiling"] = ceiling
            results.setdefault(key, []).append(m)
        _report_iter("CUB", it, iters, ceiling, results)
    return results


def run_celeba(iters, epochs=40, seed=0, num_workers=6, train_subset=5000):
    import _celeba
    from torch.utils.data import DataLoader
    device = _bench.get_device()
    spec = _celeba.build_spec()
    splits = _celeba._load_index()
    tr_files, tr_c, tr_y = splits[0]
    te_files, te_c, te_y = splits[2]

    def unpack(batch, dev):
        img, c, y = batch
        return img.to(dev), c.to(dev), y.to(dev)

    def cacc_fn(cprobs, ctrue):
        return ((cprobs > 0.5).float() == ctrue).float().mean().item()

    results = {}
    for it in range(iters):
        ceiling = _celeba._run_ctruey(spec, device, splits, 1, seed=seed + it)[0]
        _bench.set_seed(seed + it)
        perm = torch.randperm(len(tr_files))[:train_subset]
        sub_files = [tr_files[i] for i in perm.tolist()]
        tl = DataLoader(_celeba._CelebImages(sub_files, tr_c[perm], tr_y[perm], True),
                        batch_size=256, shuffle=True,
                        num_workers=num_workers, pin_memory=True)
        vl = DataLoader(_celeba._CelebImages(te_files, te_c, te_y, False),
                        batch_size=256, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
        for key in MODELS:
            _bench.set_seed(seed + it)
            model, has_c, has_side = _celeba._build_model(key, spec)
            _celeba._train(model, tl, device, epochs, LAM.get(key, 1.0), has_c)
            m = leak_eval(model, vl, device, has_side, has_c, cacc_fn, unpack)
            m["ceiling"] = ceiling
            results.setdefault(key, []).append(m)
        _report_iter("CelebA", it, iters, ceiling, results)
    return results


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _ms(vals):
    m = statistics.mean(vals)
    s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return m, s


def _report_iter(dataset, it, iters, ceiling, results):
    print(f"  [{dataset} iter {it + 1}/{iters}]  ceiling={ceiling * 100:.2f}  "
          + "  ".join(f"{k}:{results[k][-1]['full'] * 100:.1f}"
                      f"/{results[k][-1]['cpath'] * 100:.1f}"
                      f"/{results[k][-1]['cpath_h'] * 100:.1f}" for k in results))


def summarize(dataset, results):
    """Print the leakage table for one dataset (mean +/- std over seeds)."""
    print(f"\n{'=' * 92}\nLEAKAGE / {dataset}\n{'=' * 92}")
    print(f"{'Model':16s} {'full':>12s} {'cpath':>12s} {'cpath_hard':>12s} "
          f"{'leak_ceil':>11s} {'leak_side':>11s} {'leak_soft':>11s} {'concept':>9s}")
    ceil_m = _ms([r["ceiling"] for r in next(iter(results.values()))])[0]
    print(f"{'C_true->Y':16s} {ceil_m * 100:11.2f}")
    for key in MODELS:
        rs = results.get(key)
        if not rs:
            continue
        full = _ms([r["full"] for r in rs])
        cp = _ms([r["cpath"] for r in rs])
        cph = _ms([r["cpath_h"] for r in rs])
        lc = _ms([max(r["cpath"] - r["ceiling"], 0.0) for r in rs])
        ls = _ms([r["full"] - r["cpath"] for r in rs])
        lf = _ms([r["cpath"] - r["cpath_h"] for r in rs])
        cc = _ms([r["cacc"] for r in rs if r["cacc"] == r["cacc"]]) if any(
            r["cacc"] == r["cacc"] for r in rs) else None
        is_concept = key not in ("blackbox",)

        def cell(ms):
            return f"{ms[0] * 100:6.2f}+/-{ms[1] * 100:4.2f}"

        cstr = cell(cc) if cc else "     --"
        if is_concept:
            print(f"{LABELS[key]:16s} {cell(full):>12s} {cell(cp):>12s} "
                  f"{cell(cph):>12s} {cell(lc):>11s} {cell(ls):>11s} "
                  f"{cell(lf):>11s} {cstr:>9s}")
        else:
            print(f"{LABELS[key]:16s} {cell(full):>12s} {'--':>12s} {'--':>12s} "
                  f"{'--':>11s} {'--':>11s} {'--':>11s} {'--':>9s}")
    print(f"\n  leak_ceiling = max(cpath - ceiling, 0)   [total leak past concept set]")
    print(f"  leak_side    = full - cpath              [side-channel contribution]")
    print(f"  leak_soft    = cpath - cpath_hard        [continuous-value concept leak]")
