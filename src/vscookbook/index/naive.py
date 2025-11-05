import numpy as np
from .base import BaseIndex

class NaiveIndex(BaseIndex):
    def __init__(self, dim: int):
        self._dim = dim
        self._ids = []
        self._X = None
    def add(self, ids, vectors: np.ndarray):
        V = vectors.astype(np.float32)
        if self._X is None:
            self._X = V
        else:
            self._X = np.vstack([self._X, V])
        self._ids.extend(ids)
    def search(self, query: np.ndarray, k: int = 5):
        if self._X is None or len(self._ids) == 0:
            return []
        q = query.astype(np.float32)
        sims = self._X @ q.T
        order = np.argsort(-sims, axis=0).reshape(-1)[:k]
        scores = sims[order].reshape(-1).tolist()
        return [{"id": self._ids[i], "score": float(scores[idx])} for idx, i in enumerate(order)]
    def dim(self) -> int:
        return self._dim
    def get_ids(self):
        return list(self._ids)
    def get_vectors(self) -> np.ndarray:
        return self._X
