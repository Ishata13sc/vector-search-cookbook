from src.vscookbook.embedder import TextEmbedder
from src.vscookbook.index.factory import create_index
from src.vscookbook.lexical.bm25 import BM25Index
from src.vscookbook.text import tokenize
from src.vscookbook.hybrid import fuse_sum
from src.vscookbook.rerank import Reranker
from pathlib import Path
import json

root = Path(__file__).resolve().parents[1]
ds = root / "data" / "datasets" / "uploaded" / "data.jsonl"
rows = [json.loads(x) for x in open(ds,encoding="utf-8")]
texts = [r["text"] for r in rows]
ids = [r["id"] for r in rows]

embedder = TextEmbedder()
X = embedder.encode(texts)
vec = create_index("ivf", X, ids, nlist=64, nprobe=1)
bm25 = BM25Index(texts, ids, tokenize)
rer = Reranker()

qrels = {
    "hybrid search": ["d4","p1","p2"],
    "transformer architecture": ["d7","p3"],
    "optimasi indeks vektor": ["d5"],
}

def first_pos(results, rels):
    rels = set(rels)
    for i,r in enumerate(results,1):
        if str(r["id"]) in rels:
            return i
    return 999

print("q | pos@hy | pos@rerank(pool=50)")
for q, rel in qrels.items():
    v = embedder.encode([q])[0]
    vres = vec.search(v, k=50)
    lres = bm25.search(q, k=50, tokenizer=tokenize)
    hy = fuse_sum(vres, lres, topk=50, alpha=0.3)
    id2text = {i:t for i,t in zip(ids,texts)}
    for r in hy:
        r["text"] = id2text.get(r["id"],"")
    reranked = rer.rerank(q, hy, k=10)
    print(f"{q} | {first_pos(hy, rel)} | {first_pos(reranked, rel)}")
