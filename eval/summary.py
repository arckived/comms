"""Print a short Markdown summary of a trained model's results (used in CI and the README).

    python eval/summary.py artifacts/robust
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def pct(x):
    return "n/a" if x is None else f"{x:.0%}"


def main(model_dir: str) -> None:
    d = Path(model_dir)
    m = json.loads((d / "metrics.json").read_text())
    t = m["test"]
    print(f"Backend **{m['backend']}**, model **{m['best_model']}**, threshold **{m['threshold']}**, "
          f"{m['n_training_rows']} training rows, {m['n_test']} handwritten test requests.\n")
    print("| metric | value |\n|---|---|")
    print(f"| test accuracy | {pct(t['accuracy'])} |")
    print(f"| test macro F1 | {t['macro_f1']:.2f} |")
    print(f"| auto routed | {pct(t['auto_rate'])} |")
    print(f"| accuracy when auto routed | {pct(t['auto_accuracy'])} |")
    print(f"| vague requests sent to review | {pct(t['unclear_sent_to_review'])} |")
    at80 = m["test_at_threshold_0.80"]
    print(f"| at threshold 0.80: auto routed / accuracy | {pct(at80['auto_rate'])} / {pct(at80['auto_accuracy'])} |")
    if (d / "retrieval.json").exists():
        r = json.loads((d / "retrieval.json").read_text())
        print(f"| retrieval HitRate@1 / @3 / MRR | {pct(r['hit_rate@1'])} / {pct(r['hit_rate@3'])} / {r['mrr']:.2f} |")
    if (d / "extraction_rules.json").exists():
        e = json.loads((d / "extraction_rules.json").read_text())
        print(f"| rules: urgency accuracy / deadline match | {pct(e['urgency_accuracy'])} / "
              f"{pct(e['deadline_phrase_match'])} |")
    print("\n**Cross validation (grouped by seed)**\n\n| candidate | accuracy | macro F1 |\n|---|---|---|")
    for name, cv in m["cv"].items():
        print(f"| {name} | {pct(cv['accuracy'])} | {cv['macro_f1']:.2f} |")
    print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "artifacts/robust")
