import numpy as np

def fuse_sum(vscores, lscores, topk=5, alpha=0.5):
    s = {}
    for r in vscores:
        s[r["id"]] = s.get(r["id"], 0.0) + alpha * float(r["score"])
    for r in lscores:
        s[r["id"]] = s.get(r["id"], 0.0) + (1 - alpha) * float(r["score"])
    pairs = sorted(s.items(), key=lambda x: -x[1])[:topk]
    return [{"id": i, "score": float(sc)} for i, sc in pairs]
