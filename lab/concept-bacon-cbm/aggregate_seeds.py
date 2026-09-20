r"""Aggregate a seed-campaign directory into paper-ready mean +/- std.

Parses each ``results/campaign/<id>_..._s<seed>.txt`` log for its final hardened
faithful accuracy (the ``DONE ... acc XX.XX%`` line) and reports mean +/- std
(and n) per configuration (grouping over the ``_s<seed>`` suffix).

    py -3 aggregate_seeds.py results/campaign
"""
import glob
import os
import re
import sys
from collections import defaultdict


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "campaign")
    logs = sorted(glob.glob(os.path.join(root, "*.txt")))
    if not logs:
        raise SystemExit(f"no logs in {root}")
    groups = defaultdict(list)     # config -> list of (seed, acc)
    for path in logs:
        name = os.path.splitext(os.path.basename(path))[0]
        m = re.search(r"_s(\d+)$", name)
        seed = int(m.group(1)) if m else -1
        config = re.sub(r"_s\d+$", "", name)
        acc = None
        # Tee-Object on Windows PowerShell 5.1 writes UTF-16; decode robustly.
        raw = open(path, "rb").read()
        text = (raw.decode("utf-16", errors="ignore")
                if b"\x00" in raw[:200] else raw.decode("utf-8", errors="ignore"))
        for line in text.splitlines():
            mm = re.search(r"DONE.*?acc\s+([\d.]+)%", line)
            if mm:
                acc = float(mm.group(1))
        if acc is None:
            print(f"  [pending/no-acc] {name}")
            continue
        groups[config].append((seed, acc))

    print(f"\n{'config':<34}{'n':>3}{'mean':>9}{'std':>8}   seeds")
    for config in sorted(groups):
        vals = sorted(groups[config])
        accs = [a for _, a in vals]
        n = len(accs)
        mean = sum(accs) / n
        std = (sum((a - mean) ** 2 for a in accs) / n) ** 0.5 if n > 1 else 0.0
        seedstr = " ".join(f"s{s}:{a:.2f}" for s, a in vals)
        print(f"  {config:<32}{n:>3}{mean:>9.2f}{std:>8.2f}   {seedstr}")


if __name__ == "__main__":
    main()
