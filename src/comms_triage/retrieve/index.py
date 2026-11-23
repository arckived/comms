"""FAISS index over (1) public announcements / knowledge docs and (2) past requests.

    python -m comms_triage.retrieve.index            # build into MODEL_DIR/kb

Every chunk keeps its source document id, title and URL, so answers can cite where they came from.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import joblib
import numpy as np

from comms_triage.config import get_settings
from comms_triage.data.io import load_jsonl
from comms_triage.embeddings import make_embedder
from comms_triage.text import shorten


def chunk_words(text: str, size: int = 120, overlap: int = 30) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start: start + size]))
        if start + size >= len(words):
            break
        start += size - overlap
    return chunks


def build_entries(knowledge_files: list[Path], past_requests: list[dict]) -> list[dict]:
    entries = []
    for path in knowledge_files:
        for doc in load_jsonl(path):
            for i, chunk in enumerate(chunk_words(doc["text"])):
                entries.append({"kind": "doc", "doc_id": doc["id"], "chunk": i, "title": doc["title"],
                                "url": doc.get("url"), "date": doc.get("date"),
                                "text": f"{doc['title']}. {chunk}"})
    for req in past_requests:
        if req.get("team") == "Unclear":
            continue  # "test" or "can someone help" is never a useful match for anything
        entries.append({"kind": "past_request", "doc_id": req["id"], "chunk": 0,
                        "title": shorten(req["text"], 90), "url": None, "team": req.get("team"),
                        "text": req["text"]})
    return entries


class KnowledgeIndex:
    def __init__(self, embedder, entries: list[dict], index: faiss.Index):
        self.embedder = embedder
        self.entries = entries
        self.index = index

    @classmethod
    def build(cls, entries: list[dict], backend: str, sbert_model: str | None = None) -> "KnowledgeIndex":
        embedder = make_embedder(backend, sbert_model)
        texts = [e["text"] for e in entries]
        embedder.fit(texts)
        vectors = embedder.encode(texts)
        index = faiss.IndexFlatIP(vectors.shape[1])  # inner product on normalized vectors = cosine
        index.add(np.ascontiguousarray(vectors, dtype="float32"))
        return cls(embedder, entries, index)

    def search(self, query: str, k: int = 3, kind: str | None = None, min_score: float = 0.0) -> list[dict]:
        if not self.entries:
            return []
        vector = np.ascontiguousarray(self.embedder.encode([query]), dtype="float32")
        # With a kind filter, rank every entry: otherwise one kind (66 past requests) can fill the
        # candidate list and hide the other (15 announcements) completely.
        n = len(self.entries) if kind else min(len(self.entries), max(k * 4, 20))
        scores, ids = self.index.search(vector, n)
        results, seen = [], set()
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            entry = self.entries[idx]
            key = (entry["kind"], entry["doc_id"])
            if (kind and entry["kind"] != kind) or key in seen or score < min_score:
                continue
            seen.add(key)
            results.append({**{k_: v for k_, v in entry.items() if k_ != "text"},
                            "snippet": entry["text"][:240], "score": round(float(score), 3)})
            if len(results) >= k:
                break
        return results

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        (directory / "entries.json").write_text(json.dumps(self.entries))
        joblib.dump(self.embedder, directory / "embedder.joblib")

    @classmethod
    def load(cls, directory: Path) -> "KnowledgeIndex":
        index = faiss.read_index(str(directory / "index.faiss"))
        entries = json.loads((directory / "entries.json").read_text())
        return cls(joblib.load(directory / "embedder.joblib"), entries, index)


def build_default(settings=None, backend: str | None = None) -> KnowledgeIndex:
    settings = settings or get_settings()
    past_path = settings.data_dir / "splits" / "train_seeds.jsonl"
    past = load_jsonl(past_path) if past_path.exists() else []
    entries = build_entries(settings.resolved_knowledge_files(), past)
    return KnowledgeIndex.build(entries, backend or settings.embedding_backend, settings.sbert_model)


def main(argv=None):
    settings = get_settings()
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default=settings.embedding_backend, choices=["sbert", "tfidf"])
    p.add_argument("--out", default=str(settings.model_dir / "kb"))
    args = p.parse_args(argv)
    kb = build_default(settings, args.backend)
    kb.save(Path(args.out))
    kinds = {}
    for e in kb.entries:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    print(f"indexed {len(kb.entries)} chunks {kinds} with {args.backend} -> {args.out}")


if __name__ == "__main__":
    main()
