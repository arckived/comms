"""Send demo traffic to the API so the dashboard has something to show.

    python scripts/simulate.py --n 40

Uses the handwritten test requests (and harder variants of them), then reviews some of the
pending ones with the true label, like a person working through the queue would.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comms_triage.data.io import load_jsonl  # noqa: E402
from comms_triage.perturb import informal, reword, typos  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="http://localhost:8000")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--review-fraction", type=float, default=0.7)
    args = p.parse_args()

    rng = random.Random(3)
    test = load_jsonl(ROOT / "data" / "splits" / "test.jsonl")
    api = httpx.Client(base_url=args.api, timeout=60)
    sent = []
    for _ in range(args.n):
        row = rng.choice(test)
        text = rng.choice([lambda t, r: t, reword, informal, typos])(row["text"], rng)
        req = api.post("/triage", json={"text": text, "source": rng.choice(["slack", "web"])}).json()
        sent.append((req, row["team"]))
    reviewed = 0
    for req, gold in sent:
        if req["status"] == "pending_review" and rng.random() < args.review_fraction:
            team = gold if gold != "Unclear" else rng.choice(["PR", "Social", "Internal Comms"])
            api.post(f"/requests/{req['id']}/review", json={"final_team": team, "reviewer": "simulator"})
            reviewed += 1
    auto = sum(r["route"] == "auto" for r, _ in sent)
    print(f"sent {len(sent)} requests: {auto} auto routed, {len(sent) - auto} to review, {reviewed} reviewed")


if __name__ == "__main__":
    main()
