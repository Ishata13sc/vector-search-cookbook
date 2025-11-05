from rank_bm25 import BM25Okapi
from typing import List

class BM25Index:
    def __init__(self, docs: List[str], ids: List[str], tokenizer):
        self.ids = ids if docs else ["d0"]
        self.tokens = [tokenizer(d) or ["_"] for d in (docs if docs else ["_"])]
        self.bm25 = BM25Okapi(self.tokens)
    def search(self, query: str, k: int = 5, tokenizer=None):
        q = tokenizer(query) if tokenizer else query.split()
        scores = self.bm25.get_scores(q or ["_"])
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [{"id": self.ids[i] if i < len(self.ids) else "d0", "score": float(scores[i])} for i in order]
