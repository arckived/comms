"""Measure field extraction against the labeled handwritten requests.

    python eval/extraction_eval.py            # rules only
    python eval/extraction_eval.py --llm      # uses the configured LLM (LLM_BASE_URL etc.)

Metrics
* urgency accuracy
* deadline detection: does it find a deadline exactly when the label has one
* deadline phrase match: found phrase overlaps the labeled phrase
* asset hit rate: at least one extracted asset shares a word with a labeled asset
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from comms_triage.config import get_settings
from comms_triage.data.io import load_jsonl
from comms_triage.extract.llm_extract import extract_fields
from comms_triage.llm import LLMClient


def _words(s: str) -> set[str]:
    return set(s.lower().replace(",", " ").split())


def main(argv=None) -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--llm", action="store_true")
    p.add_argument("--split", default="test", choices=["test", "all"])
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    settings = get_settings()
    path = (settings.data_dir / "splits" / "test.jsonl" if args.split == "test"
            else settings.data_dir / "seed" / "seed_requests.jsonl")
    rows = load_jsonl(path)
    llm = LLMClient(settings) if args.llm else None

    n = len(rows)
    urg = det = phrase = phrase_total = asset_hit = asset_total = 0
    methods = {}
    misses = []
    for row in rows:
        f = extract_fields(row["text"], llm)
        methods[f.method] = methods.get(f.method, 0) + 1
        urg += f.urgency == row["urgency"]
        det += bool(f.deadline) == bool(row["deadline"])
        if row["deadline"]:
            phrase_total += 1
            ok = bool(f.deadline) and bool(_words(f.deadline) & _words(row["deadline"]))
            phrase += ok
            if not ok:
                misses.append({"text": row["text"], "label": row["deadline"], "got": f.deadline})
        if row["assets"]:
            asset_total += 1
            labeled = set().union(*(_words(a) for a in row["assets"]))
            asset_hit += any(_words(a) & labeled for a in f.assets)
        if llm is not None:
            time.sleep(0.2)  # stay polite with free tier rate limits

    result = {
        "split": args.split, "n": n, "methods": methods,
        "urgency_accuracy": round(urg / n, 3),
        "deadline_detection_accuracy": round(det / n, 3),
        "deadline_phrase_match": round(phrase / phrase_total, 3) if phrase_total else None,
        "asset_hit_rate": round(asset_hit / asset_total, 3) if asset_total else None,
        "deadline_misses": misses[:10],
    }
    print(json.dumps({k: v for k, v in result.items() if k != "deadline_misses"}, indent=2))
    out = Path(args.out) if args.out else settings.model_dir / f"extraction_{'llm' if args.llm else 'rules'}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
