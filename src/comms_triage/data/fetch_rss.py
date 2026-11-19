"""Pull a public RSS/Atom feed into a knowledge file the retriever can index.

    python -m comms_triage.data.fetch_rss https://blog.cloudflare.com/rss/ --out data/knowledge/rss_blog.jsonl

Then rebuild the index: python -m comms_triage.retrieve.index
Only use public feeds, and keep the source URL so every answer can cite it.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import re

import feedparser

from comms_triage.data.io import write_jsonl


def clean_html(text: str) -> str:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text or "", flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(html.unescape(text).split())


def fetch(url: str, limit: int = 100) -> list[dict]:
    feed = feedparser.parse(url)
    if feed.bozo and not feed.entries:
        raise SystemExit(f"Could not read feed {url}: {feed.bozo_exception}")
    docs = []
    for entry in feed.entries[:limit]:
        body = ""
        if entry.get("content"):
            body = entry.content[0].get("value", "")
        body = clean_html(body or entry.get("summary", ""))
        link = entry.get("link", "")
        docs.append({
            "id": "rss_" + hashlib.sha1(link.encode()).hexdigest()[:10],
            "title": clean_html(entry.get("title", "Untitled")),
            "date": (entry.get("published") or entry.get("updated") or "")[:32],
            "url": link,
            "text": body[:6000],
        })
    return docs


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("--out", default="data/knowledge/rss_feed.jsonl")
    p.add_argument("--limit", type=int, default=100)
    args = p.parse_args(argv)
    docs = fetch(args.url, args.limit)
    write_jsonl(docs, args.out)
    print(f"wrote {len(docs)} documents to {args.out}")


if __name__ == "__main__":
    main()
