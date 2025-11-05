from pathlib import Path
import numpy as np
from src.vscookbook.datasets import load_sample_texts
from src.vscookbook.embedder import TextEmbedder
from src.vscookbook.index.naive import NaiveIndex
from src.vscookbook.io import load_npy, load_json
from src.vscookbook.metrics import recall_at_k

root = Path(__file__).resolve().parents[1]
data_dir = root / "data"
embedder = TextEmbedder()
items = load_sample_texts()
texts = [x["text"] for x in items]
ids = [x["id"] for x in items]
E = load_npy(data_dir / "embeddings.npy")
ID = load_json(data_dir / "ids.json")
index = NaiveIndex(dim=E.shape[1])
index.add(ID, E)
Q = embedder.encode(texts)
recalls = []
for i, qv in enumerate(Q):
    res = index.search(qv, k=5)
    pred_ids = [r["id"] for r in res]
    gt = [ids[i]]
    r = recall_at_k(gt, pred_ids, k=5)
    recalls.append(r)
print(round(float(np.mean(recalls)), 4))
