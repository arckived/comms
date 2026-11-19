"""Text embedders with one interface: fit(texts) then encode(texts) -> L2 normalized dense array.

* SbertEmbedder: sentence-transformers (all-MiniLM-L6-v2 by default). Needs the model download.
* TfidfEmbedder: word + character TF-IDF reduced with SVD (LSA). Fully offline, used in tests and CI.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self, n_components: int = 256, seed: int = 0):
        self.n_components = n_components
        self.seed = seed
        self.vectorizer = None
        self.svd = None

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        self.vectorizer = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
        ])
        matrix = self.vectorizer.fit_transform(texts)
        n = max(2, min(self.n_components, matrix.shape[0] - 1, matrix.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=n, random_state=self.seed).fit(matrix)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.vectorizer is None:
            raise RuntimeError("TfidfEmbedder must be fit before encode")
        vectors = self.svd.transform(self.vectorizer.transform(texts))
        return normalize(vectors).astype("float32")


# Loaded models and computed vectors are shared across embedder copies, so cross validation
# loads MiniLM once and never embeds the same sentence twice.
_SBERT_MODELS: dict[str, object] = {}
_SBERT_CACHE: dict[tuple[str, str], np.ndarray] = {}


class SbertEmbedder:
    name = "sbert"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            if self.model_name not in _SBERT_MODELS:
                from sentence_transformers import SentenceTransformer  # lazy: it pulls in torch
                _SBERT_MODELS[self.model_name] = SentenceTransformer(self.model_name)
            self._model = _SBERT_MODELS[self.model_name]
        return self._model

    def fit(self, texts: list[str]) -> "SbertEmbedder":
        self._load()
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        texts = list(texts)
        missing = [t for t in dict.fromkeys(texts) if (self.model_name, t) not in _SBERT_CACHE]
        if missing:
            vectors = self._load().encode(missing, batch_size=64, normalize_embeddings=True,
                                          show_progress_bar=False)
            for text, vec in zip(missing, vectors):
                _SBERT_CACHE[(self.model_name, text)] = np.asarray(vec, dtype="float32")
        return np.stack([_SBERT_CACHE[(self.model_name, t)] for t in texts])

    def __deepcopy__(self, memo):
        return SbertEmbedder(self.model_name)

    # The model itself is not pickled; it is reloaded (from the local cache) on first use.
    def __getstate__(self):
        return {"model_name": self.model_name, "_model": None}


def make_embedder(backend: str, sbert_model: str | None = None):
    if backend == "tfidf":
        return TfidfEmbedder()
    if backend == "sbert":
        return SbertEmbedder(sbert_model or "sentence-transformers/all-MiniLM-L6-v2")
    raise ValueError(f"Unknown EMBEDDING_BACKEND {backend!r}; use 'sbert' or 'tfidf'")
