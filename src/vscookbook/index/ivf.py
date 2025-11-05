import numpy as np
from sklearn.cluster import KMeans
from .base import BaseIndex

class IVFFlat(BaseIndex):
    def __init__(self, dim: int, nlist: int = 8, nprobe: int = 2, random_state: int = 0):
        self._dim = dim
        self.nlist = nlist
        self.nprobe = nprobe
        self.centroids = None
        self.assign = None
        self._ids = None
        self._X = None
        self._fitted = False
        self._rs = random_state
    def add(self, ids, vectors: np.ndarray):
        self._X = vectors.astype(np.float32)
        self._ids = list(ids)
        km = KMeans(n_clusters=self.nlist, n_init=5, random_state=self._rs)
        self.assign = km.fit_predict(self._X)
        self.centroids = km.cluster_centers_.astype(np.float32)
        self._fitted = True
    def search(self, query: np.ndarray, k: int = 5):
        if not self._fitted:
            return []
        q = query.astype(np.float32)
        cs = self.centroids @ q.T
        order = np.argsort(-cs, axis=0).reshape(-1)[: self.nprobe]
        mask = np.isin(self.assign, order)
        Xc = self._X[mask]
        idc = np.array(self._ids, dtype=object)[mask]
        if Xc.shape[0] == 0:
            return []
        sims = Xc @ q.T
        ord2 = np.argsort(-sims, axis=0).reshape(-1)[:k]
        scores = sims[ord2].reshape(-1).tolist()
        return [{"id": str(idc[i]), "score": float(scores[idx])} for idx, i in enumerate(ord2)]
    def dim(self) -> int:
        return self._dim
    def get_ids(self):
        return list(self._ids)
    def get_vectors(self) -> np.ndarray:
        return self._X
