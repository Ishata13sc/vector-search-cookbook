from src.vscookbook.embedder import TextEmbedder
from src.vscookbook.datasets import load_sample_texts
from src.vscookbook.index.factory import create_index
from src.vscookbook.lexical.bm25 import BM25Index
from src.vscookbook.text import tokenize
from src.vscookbook.hybrid import fuse_sum
import time

items = load_sample_texts()
texts = [x['text'] for x in items]
ids = [x['id'] for x in items]

embedder = TextEmbedder()
X = embedder.encode(texts)
naive = create_index('naive', X, ids)
bm25 = BM25Index(texts, ids, tokenize)

def eval_once(q, k=5, alpha=0.5):
    t0 = time.time()
    v = embedder.encode([q])[0]
    vres = naive.search(v, k=max(10, k))
    t1 = time.time()
    lres = bm25.search(q, k=max(10, k), tokenizer=tokenize)
    t2 = time.time()
    hy = fuse_sum(vres, lres, topk=k, alpha=alpha)
    t3 = time.time()
    return {
        'vector_ms': (t1-t0)*1000,
        'lexical_ms': (t2-t1)*1000,
        'hybrid_ms': (t3-t2)*1000,
        'vector': vres[:k],
        'hybrid': hy
    }

out = eval_once('hybrid search')
print(out)

def recall_at_k(results, truth_ids, k=5):
    got = set([str(r['id']) for r in results[:k]])
    tru = set([str(t) for t in truth_ids])
    return len(got & tru) / len(tru) if tru else 0.0

truth = [ids[0]]
print('recall@k vector, hybrid:', recall_at_k(out['vector'], truth, 5), recall_at_k(out['hybrid'], truth, 5))
