"""Quality gate for CI: fail the build if the model got worse than agreed minimums.

    python eval/check_gate.py --model-dir artifacts/robust

The minimums are set for the offline TF-IDF backend (what CI uses by default); the SBERT
backend should clear them easily. Raise them as the dataset grows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="artifacts/robust")
    p.add_argument("--min-macro-f1", type=float, default=0.75)
    p.add_argument("--min-auto-accuracy", type=float, default=0.90)
    p.add_argument("--min-auto-rate", type=float, default=0.30)
    p.add_argument("--min-hit-at-3", type=float, default=0.90)
    p.add_argument("--min-typo-accuracy", type=float, default=0.70)
    args = p.parse_args(argv)
    d = Path(args.model_dir)
    m = json.loads((d / "metrics.json").read_text())["test"]
    checks = [("test macro F1", m["macro_f1"], args.min_macro_f1),
              ("accuracy when auto routed", m["auto_accuracy"] or 0.0, args.min_auto_accuracy),
              ("share auto routed", m["auto_rate"], args.min_auto_rate)]
    if (d / "retrieval.json").exists():
        checks.append(("retrieval HitRate@3", json.loads((d / "retrieval.json").read_text())["hit_rate@3"],
                       args.min_hit_at_3))
    if (d / "robustness.json").exists():
        checks.append(("accuracy on typos", json.loads((d / "robustness.json").read_text())["typos"]["accuracy"],
                       args.min_typo_accuracy))
    failed = False
    for name, value, minimum in checks:
        ok = value >= minimum
        failed |= not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {value:.3f} (min {minimum})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
