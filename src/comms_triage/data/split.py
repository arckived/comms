"""Split the handwritten seed requests into train and test.

The test set only ever contains handwritten requests, and synthetic data is generated
only from the train seeds. That way no variant of a test request leaks into training.
"""
from __future__ import annotations

import random
from collections import defaultdict


def split_seeds(seeds: list[dict], test_fraction: float = 0.4, seed: int = 42) -> tuple[list[dict], list[dict]]:
    by_team: dict[str, list[dict]] = defaultdict(list)
    for row in seeds:
        by_team[row["team"]].append(row)

    rng = random.Random(seed)
    train, test = [], []
    for team in sorted(by_team):
        rows = sorted(by_team[team], key=lambda r: r["id"])
        rng.shuffle(rows)
        n_test = max(1, round(len(rows) * test_fraction))
        test.extend(rows[:n_test])
        train.extend(rows[n_test:])
    return train, test
