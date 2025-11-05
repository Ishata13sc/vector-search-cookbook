from typing import List, Dict
import numpy as np
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name, device="cpu")
    def score(self, query: str, texts: List[str]) -> List[float]:
        pairs = [[query, t] for t in texts]
        s = self.model.predict(pairs, convert_to_numpy=True)
        return s.tolist()
    def rerank(self, query: str, items: List[Dict], k: int = 5) -> List[Dict]:
        texts = [it.get("text","") for it in items]
        scores = self.score(query, texts)
        for it, sc in zip(items, scores):
            it["rerank_score"] = float(sc)
        items.sort(key=lambda x: -x["rerank_score"])
        return items[:k]
