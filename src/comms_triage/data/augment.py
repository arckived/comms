"""Template based expansion of the train seeds into a larger synthetic set.

This is the offline, free way to grow the dataset. `generate.py --mode llm` is the
LLM based alternative. Each synthetic row keeps `parent` = the seed id, which is
used as the group for cross validation so siblings never sit in both folds.
"""
from __future__ import annotations

import random
import re

GREETINGS = ["", "", "Hi team, ", "Hey all, ", "Hello! ", "Quick one: ", "Hi comms, ", "Morning! ",
             "Hope you're well. ", "Sorry to bother you, "]
SIGN_OFFS = ["", "", " Thanks!", " Thank you.", " Appreciate it.", " Let me know.", " Thanks in advance!",
             " Happy to jump on a call.", " Any help is great."]
OPENERS = [
    (r"^Can someone\b", ["Could someone", "Is anyone able to", "Can anyone"]),
    (r"^Can we\b", ["Could we", "Are we able to", "Is it possible to"]),
    (r"^Could\b", ["Can", "Would it be possible to have"]),
    (r"^Please\b", ["Can you please", "Could you", "Kindly"]),
    (r"^Need\b", ["We need", "I need", "Looking for"]),
    (r"^We need\b", ["Need", "Our team needs"]),
]
SYNONYMS = {
    "ASAP": ["as soon as possible", "urgently", "right away"],
    "reporter": ["journalist", "writer"],
    "journalist": ["reporter", "writer"],
    "post": ["share", "publish"],
    "draft": ["write", "prepare"],
    "create": ["make", "put together"],
    "make": ["create", "design"],
    "review": ["look over", "check"],
    "announcement": ["news", "update"],
    "employees": ["staff", "everyone"],
    "CEO": ["chief executive"],
    "today": ["by end of day"],
    "help": ["support"],
}


def _swap_opener(text: str, rng: random.Random) -> str:
    for pattern, options in OPENERS:
        if re.search(pattern, text):
            if rng.random() < 0.6:
                text = re.sub(pattern, rng.choice(options), text, count=1)
            break
    return text


def _swap_synonyms(text: str, rng: random.Random, p: float = 0.35) -> str:
    def repl(match: re.Match) -> str:
        word = match.group(0)
        options = SYNONYMS.get(word)
        if options and rng.random() < p:
            return rng.choice(options)
        return word

    keys = sorted(SYNONYMS, key=len, reverse=True)
    pattern = r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b"
    return re.sub(pattern, repl, text)


def variant(text: str, rng: random.Random) -> str:
    out = _swap_opener(text, rng)
    out = _swap_synonyms(out, rng)
    greeting = rng.choice(GREETINGS)
    if greeting and out[:1].isupper() and not out.startswith(("I ", "I'")):
        out = out[0].lower() + out[1:] if rng.random() < 0.3 else out
    return (greeting + out + rng.choice(SIGN_OFFS)).strip()


def expand(seeds: list[dict], per_seed: int = 9, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for row in seeds:
        seen = {row["text"]}
        attempts = 0
        made = 0
        while made < per_seed and attempts < per_seed * 5:
            attempts += 1
            text = variant(row["text"], rng)
            if text in seen:
                continue
            seen.add(text)
            made += 1
            rows.append({**row, "id": f"{row['id']}_t{made}", "text": text, "source": "synthetic_template",
                         "parent": row["id"]})
    return rows
