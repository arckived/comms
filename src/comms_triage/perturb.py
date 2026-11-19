"""Perturbations used to stress test the classifier (the SmallHands idea, applied to text).

Each function takes a request and returns a harder version that a real person might send.
They are deterministic given an `random.Random`, so results are reproducible.
"""
from __future__ import annotations

import random
import re

# Rewording: wrap the request in different phrasing a person might use.
REWORD_TEMPLATES = [
    "I was wondering if {x}",
    "Not sure who owns this, but {x}",
    "Any chance we could sort this out? {x}",
    "Flagging this for whoever handles it: {x}",
    "Got a request from my manager. {x}",
    "{x} Who should I loop in?",
    "Wanted to check: {x}",
]
REWORD_SWAPS = {
    r"\bCan someone\b": "Would somebody",
    r"\bCan we\b": "Would we be able to",
    r"\bPlease\b": "I'd like someone to",
    r"\bNeed\b": "Requesting",
    r"\bneed\b": "require",
    r"\bcan\b": "could",
    r"\bnext week\b": "in the coming week",
    r"\btomorrow\b": "the next day",
}

# Informal: chat speak, lowercase, no punctuation.
INFORMAL_SWAPS = {
    r"\byou\b": "u", r"\bplease\b": "pls", r"\bthanks\b": "thx", r"\btomorrow\b": "tmrw",
    r"\bbecause\b": "bc", r"\bsomeone\b": "some1", r"\bbefore\b": "b4", r"\bwith\b": "w/",
    r"\bare\b": "r", r"\bfor\b": "4",
}
INFORMAL_PREFIX = ["yo ", "hey ", "ok so ", "quick q ", "", "heyy "]
INFORMAL_SUFFIX = [" lol", " ty", " 🙏", "", " thx", " plz"]

KEYBOARD_NEIGHBORS = {
    "a": "qs", "b": "vn", "c": "xv", "d": "sf", "e": "wr", "f": "dg", "g": "fh", "h": "gj", "i": "uo",
    "j": "hk", "k": "jl", "l": "k", "m": "n", "n": "bm", "o": "ip", "p": "o", "q": "w", "r": "et",
    "s": "ad", "t": "ry", "u": "yi", "v": "cb", "w": "qe", "x": "zc", "y": "tu", "z": "x",
}


def reword(text: str, rng: random.Random) -> str:
    out = text
    for pattern, repl in REWORD_SWAPS.items():
        if rng.random() < 0.7:
            out = re.sub(pattern, repl, out)
    template = rng.choice(REWORD_TEMPLATES)
    first = out[0].lower() + out[1:] if template.startswith(("I was", "Not sure", "Wanted")) else out
    return template.format(x=first)


def informal(text: str, rng: random.Random) -> str:
    out = text.lower()
    for pattern, repl in INFORMAL_SWAPS.items():
        out = re.sub(pattern, repl, out)
    out = re.sub(r"[.,!?;:]", "", out)
    return (rng.choice(INFORMAL_PREFIX) + out + rng.choice(INFORMAL_SUFFIX)).strip()


def typos(text: str, rng: random.Random, rate: float = 0.06) -> str:
    chars = list(text)
    out = []
    i = 0
    while i < len(chars):
        c = chars[i]
        if c.isalpha() and rng.random() < rate:
            op = rng.choice(["swap", "drop", "neighbor", "double"])
            if op == "swap" and i + 1 < len(chars) and chars[i + 1].isalpha():
                out.extend([chars[i + 1], c])
                i += 2
                continue
            if op == "drop":
                i += 1
                continue
            if op == "neighbor":
                near = KEYBOARD_NEIGHBORS.get(c.lower())
                out.append(rng.choice(near) if near else c)
                i += 1
                continue
            out.extend([c, c])
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def multi_team(text: str, other_text: str, rng: random.Random) -> str:
    joiner = rng.choice([" Also, ", " And separately, ", " Plus ", " One more thing: "])
    other = other_text[0].lower() + other_text[1:]
    return text.rstrip() + joiner + other


PERTURBATIONS = {"reworded": reword, "informal": informal, "typos": typos}
