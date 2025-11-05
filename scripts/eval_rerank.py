from src.vscookbook.embedder import TextEmbedder
from src.vscookbook.datasets import load_sample_texts
from src.vscookbook.index.factory import create_index
from src.vscookbook.lexical.bm25 import BM25Index
from src.vscookbook.text import tokenize
from src.vscookbook.hybrid import fuse_sum
from src.vscookbook.rerank import Reranker

items = load_sample_texts()
texts = [x["text"] for x in items]
ids = [x["id"] for x in items]
embedder = TextEmbedder()
X = embedder.encode(texts)
vec = create_index("naive", X, ids)
bm25 = BM25Index(texts, ids, tokenize)
rer = Reranker()

qrels = [
    ("hybrid search", ["d4"]),
    ("transformer architecture", ["d7"]),
    ("optimasi indeks vektor", ["d5"]),
]

def first_pos(results, rels):
    rels = set(rels)
    for i, r in enumerate(results, 1):
        if str(r["id"]) in rels:
            return i
    return 999

print("query | pos@hybrid | pos@rerank")
for q, rel in qrels:
    v = embedder.encode([q])[0]
    vres = vec.search(v, k=25)
    lres = bm25.search(q, k=25, tokenizer=tokenize)
    hy = fuse_sum(vres, lres, topk=25, alpha=0.3)
    for r in hy:
        r["text"] = texts[ids.index(r["id"])] if r["id"] in ids else ""
    reranked = rer.rerank(q, hy, k=10)
    print(f"{q} | {first_pos(hy, rel)} | {first_pos(reranked, rel)}")
