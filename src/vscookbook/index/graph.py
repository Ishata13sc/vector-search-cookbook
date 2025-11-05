import numpy as np
from sklearn.neighbors import NearestNeighbors
from .base import BaseIndex

class GraphKNN(BaseIndex):
    def __init__(self, dim: int, M: int = 10):
        self._dim = dim
        self.M = M
        self._X = None
        self._ids = None
        self.adj = None
        self._built = False
    def add(self, ids, vectors: np.ndarray):
        self._X = vectors.astype(np.float32)
        self._ids = list(ids)
        nn = NearestNeighbors(n_neighbors=min(self.M, len(self._ids)), metric="cosine", algorithm="brute")
        nn.fit(self._X)
        dist, idx = nn.kneighbors(self._X)
        self.adj = idx
        self._built = True
    def _greedy(self, q, start=0, iters=64):
        cur = start
        best = float(self._X[cur] @ q.T)
        for _ in range(iters):
            neigh = self.adj[cur]
            sims = self._X[neigh] @ q.T
            j = int(np.argmax(sims))
            if float(sims[j]) <= best:
                break
            best = float(sims[j])
            cur = int(neigh[j])
        return cur
    def search(self, query: np.ndarray, k: int = 5):
        if not self._built:
            return []
        q = query.astype(np.float32)
        start = 0
        hub = self._greedy(q, start=start)
        cand = np.unique(np.concatenate([self.adj[hub], [hub]]))
        sims = self._X[cand] @ q.T
        ord2 = np.argsort(-sims, axis=0).reshape(-1)[:k]
        scores = sims[ord2].reshape(-1).tolist()
        ids = np.array(self._ids, dtype=object)[cand]
        return [{"id": str(ids[i]), "score": float(scores[idx])} for idx, i in enumerate(ord2)]
    def dim(self) -> int:
        return self._dim
    def get_ids(self):
        return list(self._ids)
    def get_vectors(self) -> np.ndarray:
        return self._X
