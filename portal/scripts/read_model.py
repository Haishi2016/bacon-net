"""Read embedded display metadata from a saved BACON model checkpoint.

Uses bacon.utils.read_model_metadata so the single .pth file is the source of
truth for display (feature names, aggregator, accuracy, tree).

Usage:
    python read_model.py --pth <path-to-.pth> [--repo <repo-root>]

Output (stdout): a single JSON object, either the metadata dict or {"error": ...}.
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth", required=True)
    parser.add_argument("--repo", default=str(DEFAULT_REPO))
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass

    sys.path.insert(0, args.repo)

    try:
        from bacon.utils import read_model_metadata
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": f"Could not import bacon: {exc}"}))
        return 1

    try:
        metadata = read_model_metadata(args.pth)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": f"Could not read model: {exc}"}))
        return 1

    print(json.dumps({"metadata": metadata}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
