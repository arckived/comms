"""Retrieval quality on labeled queries: HitRate@1/3/5 and MRR (same metrics as SousChef).

    python eval/retrieval_eval.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from comms_triage.config import get_settings
from comms_triage.data.io import load_jsonl
from comms_triage.retrieve.index import KnowledgeIndex, build_default


def evaluate(kb: KnowledgeIndex, queries: list[dict], ks=(1, 3, 5)) -> dict:
    hits = {k: 0 for k in ks}
    rr_total = 0.0
    failures = []
    for q in queries:
        results = kb.search(q["query"], k=max(ks), kind="doc")
        ranked = [r["doc_id"] for r in results]
        rank = next((i + 1 for i, d in enumerate(ranked) if d in q["relevant"]), None)
        for k in ks:
            hits[k] += rank is not None and rank <= k
        rr_total += 1 / rank if rank else 0.0
        if rank != 1:
            failures.append({"query": q["query"], "relevant": q["relevant"], "top3": ranked[:3]})
    n = len(queries)
    return {"n_queries": n, **{f"hit_rate@{k}": round(hits[k] / n, 3) for k in ks},
            "mrr": round(rr_total / n, 3), "not_first": failures}


def main(argv=None) -> dict:
    settings = get_settings()
    p = argparse.ArgumentParser()
    p.add_argument("--kb", default=str(settings.model_dir / "kb"))
    p.add_argument("--queries", default=str(settings.data_dir / "knowledge" / "retrieval_queries.jsonl"))
    p.add_argument("--backend", default=None, help="rebuild the index with this backend instead of loading")
    args = p.parse_args(argv)
    kb = build_default(settings, args.backend) if args.backend or not Path(args.kb).exists() \
        else KnowledgeIndex.load(Path(args.kb))
    result = evaluate(kb, load_jsonl(args.queries))
    print(json.dumps({k: v for k, v in result.items() if k != "not_first"}, indent=2))
    out = settings.model_dir / "retrieval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
