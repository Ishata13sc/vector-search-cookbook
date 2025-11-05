import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def recall_at_k(ground_truth_ids, predicted_ids, k: int = 10) -> float:
    gt = set(ground_truth_ids)
    pred = set(predicted_ids[:k])
    if not gt:
        return 0.0
    return len(gt & pred) / len(gt)
