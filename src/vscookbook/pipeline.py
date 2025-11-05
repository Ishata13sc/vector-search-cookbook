from typing import List, Dict, Tuple
import numpy as np
from .embedder import TextEmbedder

def build_embeddings(items: List[Dict], model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 64) -> Tuple[np.ndarray, List[str]]:
    embedder = TextEmbedder(model_name=model_name)
    texts = [x["text"] for x in items]
    ids = [x["id"] for x in items]
    X = embedder.encode(texts)
    return X, ids
