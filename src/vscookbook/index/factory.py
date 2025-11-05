import numpy as np
from .naive import NaiveIndex
from .ivf import IVFFlat
from .graph import GraphKNN

def create_index(kind: str, X: np.ndarray, ids, **kw):
    dim = X.shape[1]
    if kind == "naive":
        ix = NaiveIndex(dim=dim)
        ix.add(ids, X)
        return ix
    if kind == "ivf":
        nlist = int(kw.get("nlist", 8))
        nprobe = int(kw.get("nprobe", 2))
        ix = IVFFlat(dim=dim, nlist=nlist, nprobe=nprobe)
        ix.add(ids, X)
        return ix
    if kind == "graph":
        M = int(kw.get("M", 10))
        ix = GraphKNN(dim=dim, M=M)
        ix.add(ids, X)
        return ix
    raise ValueError("unknown index kind")
