"""Stress test the classifier on harder versions of the handwritten test requests.

    python eval/robustness.py --models artifacts/base artifacts/robust

Variants
* clean      the test requests as written
* reworded   different phrasing (the robust model is NOT trained on this transform)
* informal   lowercase chat speak, no punctuation (robust model trains on this kind of noise)
* typos      keyboard typos at ~6% of letters (robust model trains on this kind of noise)
* multi_team two requests for different teams joined together; counted correct if the model
             picks either team, and we also report how often it sends these to review

Writes <model_dir>/robustness.json and reports/robustness.md.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score

from comms_triage.config import PROJECT_ROOT, get_settings
from comms_triage.data.io import load_jsonl
from comms_triage.perturb import PERTURBATIONS, multi_team


def make_variants(test: list[dict], seed: int = 99) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    variants = {"clean": [{"text": r["text"], "gold": {r["team"]}, "team": r["team"]} for r in test]}
    for name, fn in PERTURBATIONS.items():
        variants[name] = [{"text": fn(r["text"], rng), "gold": {r["team"]}, "team": r["team"]} for r in test]
    routable = [r for r in test if r["team"] != "Unclear"]
    pairs = []
    for r in routable:
        other = rng.choice([o for o in routable if o["team"] != r["team"]])
        pairs.append({"text": multi_team(r["text"], other["text"], rng), "gold": {r["team"], other["team"]},
                      "team": r["team"]})
    variants["multi_team"] = pairs
    return variants


def score(artifact: dict, rows: list[dict]) -> dict:
    clf, threshold = artifact["classifier"], artifact["threshold"]
    probs = clf.predict_proba([r["text"] for r in rows])
    pred = [clf.classes_[i] for i in probs.argmax(axis=1)]
    conf = probs.max(axis=1)
    correct = np.array([p in r["gold"] for p, r in zip(pred, rows)])
    auto = (conf >= threshold) & (np.array(pred) != "Unclear")
    # Macro F1 against the first listed team (only meaningful for single-team variants).
    f1 = f1_score([r["team"] for r in rows], pred, average="macro")
    return {"n": len(rows), "accuracy": round(float(correct.mean()), 3), "macro_f1": round(float(f1), 3),
            "auto_rate": round(float(auto.mean()), 3),
            "auto_accuracy": round(float(correct[auto].mean()), 3) if auto.any() else None,
            "mean_confidence": round(float(conf.mean()), 3)}


def main(argv=None):
    settings = get_settings()
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=[str(PROJECT_ROOT / "artifacts" / "base"),
                                                   str(PROJECT_ROOT / "artifacts" / "robust")])
    p.add_argument("--test", default=str(settings.data_dir / "splits" / "test.jsonl"))
    p.add_argument("--report", default=str(PROJECT_ROOT / "reports" / "robustness.md"))
    args = p.parse_args(argv)

    test = load_jsonl(args.test)
    variants = make_variants(test)
    results = {}
    for model_dir in args.models:
        artifact = joblib.load(Path(model_dir) / "model.joblib")
        res = {name: score(artifact, rows) for name, rows in variants.items()}
        res["_threshold"] = artifact["threshold"]
        (Path(model_dir) / "robustness.json").write_text(json.dumps(res, indent=2))
        results[Path(model_dir).name] = res

    lines = ["# Robustness results", "",
             f"Test set: {len(test)} handwritten requests. Each variant is generated with a fixed seed.", "",
             "| variant | " + " | ".join(f"{m} acc | {m} auto rate | {m} auto acc" for m in results) + " |",
             "|---|" + "---|---|---|" * len(results)]
    for name in variants:
        cells = []
        for m in results:
            r = results[m][name]
            auto_acc = "n/a" if r["auto_accuracy"] is None else f"{r['auto_accuracy']:.0%}"
            cells += [f"{r['accuracy']:.0%}", f"{r['auto_rate']:.0%}", auto_acc]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "Thresholds: " + ", ".join(f"{m} = {results[m]['_threshold']}" for m in results), "",
              "For multi_team, accuracy counts a prediction of either team as correct."]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
