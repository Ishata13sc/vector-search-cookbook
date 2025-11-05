from src.vscookbook.embedder import TextEmbedder
from src.vscookbook.index.factory import create_index
from src.vscookbook.lexical.bm25 import BM25Index
from src.vscookbook.text import tokenize
from src.vscookbook.hybrid import fuse_sum
from pathlib import Path
import json, math

root = Path(__file__).resolve().parents[1]
ds = root / "data" / "datasets" / "uploaded" / "data.jsonl"
rows = [json.loads(x) for x in open(ds,encoding="utf-8")]
texts = [r["text"] for r in rows]
ids = [r["id"] for r in rows]

embedder = TextEmbedder()
X = embedder.encode(texts)
vec = create_index("ivf", X, ids, nlist=64, nprobe=1)
bm25 = BM25Index(texts, ids, tokenize)

qrels = {
    "hybrid search": ["d4","p1","p2"],
    "transformer architecture": ["d7","p3"],
    "optimasi indeks vektor": ["d5"],
    "pk-24": ["d10","p5"],
    "rawon kluwek": ["d1","p6","t1"],
}

def recall_at_k(results, rels, k):
    got = set(str(r["id"]) for r in results[:k])
    return len(got & set(rels)) / len(rels)

def mrr_at_k(results, rels, k):
    rels = set(rels)
    for i,r in enumerate(results[:k],1):
        if str(r["id"]) in rels:
            return 1.0/i
    return 0.0

def ndcg_at_k(results, rels, k):
    rels = set(rels)
    dcg = 0.0
    for i,r in enumerate(results[:k],1):
        gain = 1.0 if str(r["id"]) in rels else 0.0
        dcg += gain / math.log2(i+1)
    ideal = sum(1.0 / math.log2(i+1) for i in range(1, min(k,len(rels))+1))
    return dcg/ideal if ideal>0 else 0.0

def run(alpha=0.3, k=10):
    rows=[]
    for q, rel in qrels.items():
        v = embedder.encode([q])[0]
        vres = vec.search(v, k=50)
        lres = bm25.search(q, k=50, tokenizer=tokenize)
        hy = fuse_sum(vres, lres, topk=50, alpha=alpha)
        r = {
            "q": q,
            "vec@1": recall_at_k(vres, rel, 1),
            "vec@5": recall_at_k(vres, rel, 5),
            "vec_mrr": mrr_at_k(vres, rel, k),
            "vec_ndcg": ndcg_at_k(vres, rel, k),
            "hy@1": recall_at_k(hy, rel, 1),
            "hy@5": recall_at_k(hy, rel, 5),
            "hy_mrr": mrr_at_k(hy, rel, k),
            "hy_ndcg": ndcg_at_k(hy, rel, k)
        }
        rows.append(r)
    return rows

out = run(alpha=0.3, k=10)
print("q | vec@1 | hy@1 | vec_mrr | hy_mrr | vec_ndcg | hy_ndcg")
for r in out:
    print(f"{r['q']} | {r['vec@1']:.2f} | {r['hy@1']:.2f} | {r['vec_mrr']:.2f} | {r['hy_mrr']:.2f} | {r['vec_ndcg']:.2f} | {r['hy_ndcg']:.2f}")
