from pathlib import Path
import json
import numpy as np
from .factory import create_index

def save_index(dirpath: str, kind: str, index_obj):
    d = Path(dirpath)
    d.mkdir(parents=True, exist_ok=True)
    X = index_obj.get_vectors()
    ids = index_obj.get_ids()
    np.save(d / "vectors.npy", X)
    with open(d / "ids.json", "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False)
    with open(d / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"kind": kind}, f)

def load_index(dirpath: str):
    d = Path(dirpath)
    X = np.load(d / "vectors.npy")
    with open(d / "ids.json", "r", encoding="utf-8") as f:
        ids = json.load(f)
    with open(d / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    ix = create_index(meta["kind"], X, ids)
    return meta["kind"], ix

