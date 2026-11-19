"""Train the team classifier.

    python -m comms_triage.classify.train --out artifacts/base
    python -m comms_triage.classify.train --out artifacts/robust --augment-robust

Steps
1. Split handwritten seeds into train and test (test is handwritten only).
2. Expand the train seeds with templates (and LLM generated rows, if present).
3. Optionally add informal and typo variants (--augment-robust).
4. 5 fold cross validation grouped by seed id for every candidate; pick the best macro F1.
5. Choose the confidence threshold on the out of fold predictions.
6. Refit the best candidate on all training data and evaluate once on the test set.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from comms_triage.classify.models import make_candidates
from comms_triage.config import TEAMS, get_settings
from comms_triage.data.augment import expand
from comms_triage.data.io import load_jsonl, write_jsonl
from comms_triage.data.split import split_seeds
from comms_triage.embeddings import make_embedder
from comms_triage.perturb import informal, typos


def routing_decisions(pred: list[str], conf: np.ndarray, threshold: float) -> np.ndarray:
    """True where a request is routed automatically. 'Unclear' always goes to review."""
    return (conf >= threshold) & (np.array(pred) != "Unclear")


def threshold_curve(y: list[str], pred: list[str], conf: np.ndarray) -> list[dict]:
    correct = np.array(y) == np.array(pred)
    rows = []
    for t in np.round(np.arange(0.30, 0.96, 0.05), 2):
        auto = routing_decisions(pred, conf, t)
        rows.append({
            "threshold": float(t),
            "auto_rate": float(auto.mean()),
            "auto_accuracy": float(correct[auto].mean()) if auto.any() else None,
            "n_auto": int(auto.sum()),
        })
    return rows


def choose_threshold(curve: list[dict], target_accuracy: float) -> float:
    """Lowest threshold whose auto-routed accuracy meets the target (routes the most requests)."""
    ok = [r for r in curve if r["auto_accuracy"] is not None and r["auto_accuracy"] >= target_accuracy]
    if ok:
        return min(r["threshold"] for r in ok)
    scored = [r for r in curve if r["auto_accuracy"] is not None]
    return max(scored, key=lambda r: (r["auto_accuracy"], -r["threshold"]))["threshold"] if scored else 0.95


def expected_calibration_error(y: list[str], pred: list[str], conf: np.ndarray, bins: int = 10) -> float:
    correct = (np.array(y) == np.array(pred)).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def robust_variants(rows: list[dict], seed: int = 1234) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for row in rows:
        for name, fn in (("inf", informal), ("typo", typos)):
            out.append({**row, "id": f"{row['id']}_{name}", "text": fn(row["text"], rng),
                        "source": f"augment_{name}", "parent": row.get("parent", row["id"])})
    return out


def build_training_set(train_seeds: list[dict], per_seed: int, llm_file: Path | None,
                       augment_robust: bool, feedback_file: Path | None = None,
                       exclude_texts: set[str] | None = None) -> list[dict]:
    rows = [{**r, "parent": r["id"]} for r in train_seeds]
    if feedback_file and feedback_file.exists():
        # Reviewed requests exported from GET /feedback. Never let a test request leak in.
        exclude_texts = exclude_texts or set()
        rows += [{**r, "parent": r["id"]} for r in load_jsonl(feedback_file)
                 if r.get("team") and r["text"] not in exclude_texts]
    rows += expand(train_seeds, per_seed=per_seed)
    if llm_file and llm_file.exists():
        train_ids = {r["id"] for r in train_seeds}
        llm_rows = [{**r, "parent": r.get("parent", r["id"])} for r in load_jsonl(llm_file)
                    if r.get("parent") in train_ids or "parent" not in r]
        rows += llm_rows
        rows += expand(llm_rows, per_seed=max(1, per_seed // 3), seed=11)
    if augment_robust:
        rows += robust_variants(rows)
    return rows


def evaluate(y_true: list[str], probs: np.ndarray, classes: list[str], threshold: float) -> dict:
    pred = [classes[i] for i in probs.argmax(axis=1)]
    conf = probs.max(axis=1)
    auto = routing_decisions(pred, conf, threshold)
    correct = np.array(y_true) == np.array(pred)
    labels = [t for t in TEAMS if t in set(y_true) | set(pred)]
    unclear = np.array(y_true) == "Unclear"
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "ece": expected_calibration_error(y_true, pred, conf),
        "threshold": threshold,
        "auto_rate": float(auto.mean()),
        "auto_accuracy": float(correct[auto].mean()) if auto.any() else None,
        "review_rate": float(1 - auto.mean()),
        "unclear_sent_to_review": float((~auto[unclear]).mean()) if unclear.any() else None,
        "per_class": classification_report(y_true, pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": {"labels": labels, "matrix": confusion_matrix(y_true, pred, labels=labels).tolist()},
    }


def main(argv: list[str] | None = None) -> dict:
    settings = get_settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default=settings.embedding_backend, choices=["sbert", "tfidf"])
    p.add_argument("--out", default=str(settings.model_dir))
    p.add_argument("--seeds", default=str(settings.data_dir / "seed" / "seed_requests.jsonl"))
    p.add_argument("--llm-data", default=str(settings.data_dir / "synthetic" / "llm_generated.jsonl"))
    p.add_argument("--feedback", default=None, help="JSONL exported from GET /feedback")
    p.add_argument("--per-seed", type=int, default=9)
    p.add_argument("--augment-robust", action="store_true")
    p.add_argument("--target-accuracy", type=float, default=0.95)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--candidates", default="tfidf_lr,emb_lr,emb_svm,emb_ensemble")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    seeds = load_jsonl(args.seeds)
    train_seeds, test = split_seeds(seeds)
    write_jsonl(train_seeds, settings.data_dir / "splits" / "train_seeds.jsonl")
    write_jsonl(test, settings.data_dir / "splits" / "test.jsonl")
    rows = build_training_set(train_seeds, args.per_seed, Path(args.llm_data), args.augment_robust,
                              Path(args.feedback) if args.feedback else None, {r["text"] for r in test})
    write_jsonl(rows, out / "training_rows.jsonl")

    texts = [r["text"] for r in rows]
    labels = [r["team"] for r in rows]
    groups = [r["parent"] for r in rows]
    print(f"train seeds={len(train_seeds)} training rows={len(rows)} test={len(test)} backend={args.backend}")

    embedder = make_embedder(args.backend, settings.sbert_model)
    wanted = args.candidates.split(",")
    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=0)
    cv_results, oof_by_model = {}, {}
    for name in wanted:
        oof = None
        classes = None
        for train_idx, val_idx in cv.split(texts, labels, groups):
            model = make_candidates(embedder)[name]
            model.fit([texts[i] for i in train_idx], [labels[i] for i in train_idx])
            probs = model.predict_proba([texts[i] for i in val_idx])
            if oof is None:
                classes = sorted(set(labels))
                oof = np.zeros((len(texts), len(classes)))
            # Align columns in case a fold missed a class.
            for j, c in enumerate(model.classes_):
                oof[val_idx, classes.index(c)] = probs[:, j]
        pred = [classes[i] for i in oof.argmax(axis=1)]
        cv_results[name] = {"accuracy": float(accuracy_score(labels, pred)),
                            "macro_f1": float(f1_score(labels, pred, average="macro"))}
        oof_by_model[name] = (oof, classes)
        print(f"  cv {name:13s} acc={cv_results[name]['accuracy']:.3f} macro_f1={cv_results[name]['macro_f1']:.3f}")

    best = max(cv_results, key=lambda n: cv_results[n]["macro_f1"])
    oof, classes = oof_by_model[best]
    oof_pred = [classes[i] for i in oof.argmax(axis=1)]
    curve = threshold_curve(labels, oof_pred, oof.max(axis=1))
    threshold = choose_threshold(curve, args.target_accuracy)
    print(f"best={best} threshold={threshold} (target auto accuracy {args.target_accuracy})")

    final = make_candidates(embedder)[best].fit(texts, labels)
    test_probs = final.predict_proba([r["text"] for r in test])
    test_metrics = evaluate([r["team"] for r in test], test_probs, final.classes_, threshold)
    test_at_080 = evaluate([r["team"] for r in test], test_probs, final.classes_, 0.80)

    metrics = {
        "backend": args.backend,
        "sbert_model": settings.sbert_model if args.backend == "sbert" else None,
        "augment_robust": args.augment_robust,
        "n_train_seeds": len(train_seeds),
        "n_training_rows": len(rows),
        "n_test": len(test),
        "cv": cv_results,
        "best_model": best,
        "threshold": threshold,
        "target_auto_accuracy": args.target_accuracy,
        "threshold_curve_oof": curve,
        "test": test_metrics,
        "test_at_threshold_0.80": {k: test_at_080[k] for k in ("auto_rate", "auto_accuracy", "review_rate")},
        "train_seconds": round(time.time() - started, 1),
    }
    joblib.dump({"classifier": final, "threshold": threshold, "backend": args.backend,
                 "best_model": best}, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"test acc={test_metrics['accuracy']:.3f} macro_f1={test_metrics['macro_f1']:.3f} "
          f"auto_rate={test_metrics['auto_rate']:.2f} auto_acc={test_metrics['auto_accuracy']}")
    print(f"saved to {out}")
    return metrics


if __name__ == "__main__":
    main()
