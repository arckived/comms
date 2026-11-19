import random
from collections import Counter

from comms_triage.config import PROJECT_ROOT, TEAMS
from comms_triage.data.augment import expand
from comms_triage.data.io import load_jsonl
from comms_triage.data.split import split_seeds
from comms_triage.perturb import PERTURBATIONS, multi_team

SEEDS = load_jsonl(PROJECT_ROOT / "data" / "seed" / "seed_requests.jsonl")


def test_seed_file_is_valid():
    assert len(SEEDS) >= 100
    assert len({r["id"] for r in SEEDS}) == len(SEEDS)
    for row in SEEDS:
        assert row["team"] in TEAMS
        assert row["urgency"] in ("low", "medium", "high")
        assert row["text"].strip()


def test_split_is_disjoint_stratified_and_deterministic():
    train, test = split_seeds(SEEDS)
    assert not {r["id"] for r in train} & {r["id"] for r in test}
    assert len(train) + len(test) == len(SEEDS)
    assert set(Counter(r["team"] for r in test)) == set(TEAMS)
    assert [r["id"] for r in split_seeds(SEEDS)[1]] == [r["id"] for r in test]


def test_expand_keeps_labels_and_parent():
    train, _ = split_seeds(SEEDS)
    rows = expand(train[:5], per_seed=4)
    assert len(rows) == 20
    by_id = {r["id"]: r for r in train}
    for row in rows:
        assert row["team"] == by_id[row["parent"]]["team"]
        assert row["text"] != by_id[row["parent"]]["text"]


def test_perturbations_are_deterministic_and_change_text():
    text = "Can someone please draft a press release for the launch? Thanks"
    for name, fn in PERTURBATIONS.items():
        a = fn(text, random.Random(1))
        b = fn(text, random.Random(1))
        assert a == b, name
        assert a != text, name
    joined = multi_team("Need a banner.", "Please post on LinkedIn.", random.Random(0))
    assert "banner" in joined and "linkedin" in joined.lower()


def test_llm_generated_rows_never_copy_test_requests():
    _, test = split_seeds(SEEDS)
    test_texts = {r["text"].lower() for r in test}
    generated = load_jsonl(PROJECT_ROOT / "data" / "synthetic" / "llm_generated.jsonl")
    assert not test_texts & {r["text"].lower() for r in generated}
