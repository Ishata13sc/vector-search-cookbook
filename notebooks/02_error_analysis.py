from src.vscookbook.embedder import TextEmbedder
from src.vscookbook.datasets import load_sample_texts
from src.vscookbook.index.factory import create_index
from src.vscookbook.lexical.bm25 import BM25Index
from src.vscookbook.text import tokenize
from src.vscookbook.hybrid import fuse_sum

items = load_sample_texts()
texts = [x['text'] for x in items]
ids = [x['id'] for x in items]

embedder = TextEmbedder()
X = embedder.encode(texts)
idx = create_index('naive', X, ids)
bm = BM25Index(texts, ids, tokenize)

def analyze(q, k=5, alpha=0.5):
    v = embedder.encode([q])[0]
    vr = idx.search(v, k=max(10, k))
    lr = bm.search(q, k=max(10, k), tokenizer=tokenize)
    hy = fuse_sum(vr, lr, topk=k, alpha=alpha)
    vids = [str(x['id']) for x in vr[:k]]
    lids = [str(x['id']) for x in lr[:k]]
    hids = [str(x['id']) for x in hy[:k]]
    return {
        'vector': vids,
        'bm25': lids,
        'hybrid': hids,
        'only_vector': list(set(vids) - set(lids)),
        'only_bm25': list(set(lids) - set(vids)),
    }

print(analyze('transformer architecture', k=5, alpha=0.3))
