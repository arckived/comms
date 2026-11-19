"""Candidate classifiers. Every candidate exposes fit(texts, labels) and predict_proba(texts).

* tfidf_lr      baseline: raw TF-IDF + logistic regression (no embeddings)
* emb_lr        embeddings + logistic regression
* emb_svm       embeddings + linear SVM with sigmoid calibration
* emb_ensemble  average of the calibrated SVM and XGBoost probabilities (the Mint Sage approach)
"""
from __future__ import annotations

import copy

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC


class TextClassifier:
    """Base class: keeps the label order in `classes_`."""

    name = "base"

    def __init__(self):
        self.classes_: list[str] = []

    def fit(self, texts: list[str], labels: list[str]) -> "TextClassifier":
        raise NotImplementedError

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def predict(self, texts: list[str]) -> list[str]:
        probs = self.predict_proba(texts)
        return [self.classes_[i] for i in probs.argmax(axis=1)]


class TfidfLogReg(TextClassifier):
    name = "tfidf_lr"

    def fit(self, texts, labels):
        self.pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                                  LogisticRegression(C=10, max_iter=2000))
        self.pipe.fit(texts, labels)
        self.classes_ = list(self.pipe.classes_)
        return self

    def predict_proba(self, texts):
        return self.pipe.predict_proba(texts)


class EmbeddingClassifier(TextClassifier):
    """Embeds texts, then runs one or more probabilistic heads and averages them."""

    def __init__(self, embedder, heads: list[str], name: str):
        super().__init__()
        self.embedder = embedder
        self.head_names = heads
        self.name = name
        self.heads = []

    def _make_head(self, kind: str, n_classes: int):
        if kind == "lr":
            return LogisticRegression(C=5, max_iter=3000)
        if kind == "svm":
            return CalibratedClassifierCV(LinearSVC(C=0.5), cv=3, method="sigmoid")
        if kind == "xgb":
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.1, subsample=0.9,
                                 colsample_bytree=0.8, objective="multi:softprob", num_class=n_classes,
                                 eval_metric="mlogloss", n_jobs=2, verbosity=0)
        raise ValueError(kind)

    def fit(self, texts, labels):
        self.classes_ = sorted(set(labels))
        index = {c: i for i, c in enumerate(self.classes_)}
        y = np.array([index[l] for l in labels])
        self.embedder.fit(list(texts))
        X = self.embedder.encode(list(texts))
        self.heads = []
        for kind in self.head_names:
            head = self._make_head(kind, len(self.classes_))
            head.fit(X, y)
            self.heads.append(head)
        return self

    def predict_proba(self, texts):
        X = self.embedder.encode(list(texts))
        probs = [head.predict_proba(X) for head in self.heads]
        return np.mean(probs, axis=0)


def make_candidates(embedder) -> dict[str, TextClassifier]:
    """Fresh, unfitted candidates. The embedder is copied so candidates never share state."""
    return {
        "tfidf_lr": TfidfLogReg(),
        "emb_lr": EmbeddingClassifier(copy.deepcopy(embedder), ["lr"], "emb_lr"),
        "emb_svm": EmbeddingClassifier(copy.deepcopy(embedder), ["svm"], "emb_svm"),
        "emb_ensemble": EmbeddingClassifier(copy.deepcopy(embedder), ["svm", "xgb"], "emb_ensemble"),
    }
