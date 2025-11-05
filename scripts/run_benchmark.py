import json
from pathlib import Path
from src.vscookbook.embedder import TextEmbedder
from src.vscookbook.datasets import load_sample_texts
from src.vscookbook.index.factory import create_index
from src.vscookbook.lexical.bm25 import BM25Index
from src.vscookbook.text import tokenize
from src.vscookbook.bench import bench_index, bench_hybrid

root = Path(__file__).resolve().parents[1]
items = load_sample_texts()
texts = [x["text"] for x in items]
ids = [x["id"] for x in items]
emb = TextEmbedder()
X = emb.encode(texts)
bm25 = BM25Index(texts, ids, tokenize)

results = {}
for kind in ["naive","ivf","graph"]:
    ix = create_index(kind, X, ids)
    results[kind] = bench_index(emb, ix, texts, ids, k=5)
results["hybrid_alpha_0.5"] = bench_hybrid(emb, create_index("naive", X, ids), bm25, texts, ids, k=5, alpha=0.5)

out = root / "data" / "benchmark.json"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(str(out))
