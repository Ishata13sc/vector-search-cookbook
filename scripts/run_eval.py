from src.vscookbook.embedder import TextEmbedder
from src.vscookbook.datasets import load_sample_texts
from src.vscookbook.index.factory import create_index
from src.vscookbook.lexical.bm25 import BM25Index
from src.vscookbook.text import tokenize
from src.vscookbook.hybrid import fuse_sum

items = load_sample_texts()
texts = [x["text"] for x in items]
ids = [x["id"] for x in items]
embedder = TextEmbedder()
X = embedder.encode(texts)
vec = create_index("naive", X, ids)
bm25 = BM25Index(texts, ids, tokenize)

qrels = [
    ("hybrid search", ["d4"]),
    ("transformer architecture", ["d7"]),
    ("optimasi indeks vektor", ["d5"]),
    ("pk-24", ["d10"]),
    ("rawon kluwek", ["d1"]),
]

def recall_at_k(results, truth_ids, k):
    got = set(str(r["id"]) for r in results[:k])
    tru = set(str(t) for t in truth_ids)
    return len(got & tru) / len(tru) if tru else 0.0

rows = []
for q, rel in qrels:
    v = embedder.encode([q])[0]
    vres = vec.search(v, k=25)
    lres = bm25.search(q, k=25, tokenizer=tokenize)
    hy = fuse_sum(vres, lres, topk=10, alpha=0.3)
    r1 = recall_at_k(vres, rel, 1); r5 = recall_at_k(vres, rel, 5)
    h1 = recall_at_k(hy, rel, 1);  h5 = recall_at_k(hy, rel, 5)
    rows.append((q, rel, r1, r5, h1, h5))

print("query | rel | vec@1 | vec@5 | hy@1 | hy@5")
for r in rows:
    print(f"{r[0]} | {r[1]} | {r[2]:.2f} | {r[3]:.2f} | {r[4]:.2f} | {r[5]:.2f}")
