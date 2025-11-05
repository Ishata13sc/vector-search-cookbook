from typing import List, Dict
import time
import numpy as np
from .metrics import recall_at_k
from .hybrid import fuse_sum

def _avg(xs):
    return float(sum(xs) / max(1, len(xs)))

def bench_index(embedder, index, texts: List[str], ids: List[str], k: int = 5):
    enc_t, srch_t, r1, r5 = [], [], [], []
    for i, q in enumerate(texts):
        t0 = time.perf_counter()
        v = embedder.encode([q])[0]
        t1 = time.perf_counter()
        res = index.search(v, k=k)
        t2 = time.perf_counter()
        enc_t.append((t1 - t0) * 1000.0)
        srch_t.append((t2 - t1) * 1000.0)
        pred = [r["id"] for r in res]
        r1.append(recall_at_k([ids[i]], pred, k=1))
        r5.append(recall_at_k([ids[i]], pred, k=k))
    return {
        "avg_embed_ms": _avg(enc_t),
        "avg_search_ms": _avg(srch_t),
        "recall_at_1": _avg(r1),
        "recall_at_5": _avg(r5)
    }

def bench_hybrid(embedder, index, bm25, texts: List[str], ids: List[str], k: int = 5, alpha: float = 0.5):
    enc_t, fuse_t, r1, r5 = [], [], [], []
    for i, q in enumerate(texts):
        t0 = time.perf_counter()
        v = embedder.encode([q])[0]
        vres = index.search(v, k=max(10, k))
        lres = bm25.search(q, k=max(10, k))
        t1 = time.perf_counter()
        fused = fuse_sum(vres, lres, topk=k, alpha=alpha)
        t2 = time.perf_counter()
        enc_t.append((t1 - t0) * 1000.0)
        fuse_t.append((t2 - t1) * 1000.0)
        pred = [r["id"] for r in fused]
        r1.append(recall_at_k([ids[i]], pred, k=1))
        r5.append(recall_at_k([ids[i]], pred, k=k))
    return {
        "avg_embed_ms": _avg(enc_t),
        "avg_fuse_ms": _avg(fuse_t),
        "recall_at_1": _avg(r1),
        "recall_at_5": _avg(r5),
        "alpha": alpha
    }
