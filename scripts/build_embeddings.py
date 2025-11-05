from pathlib import Path
from src.vscookbook.datasets import load_sample_texts
from src.vscookbook.pipeline import build_embeddings
from src.vscookbook.io import save_npy, save_json

root = Path(__file__).resolve().parents[1]
data_dir = root / "data"
items = load_sample_texts()
X, ids = build_embeddings(items)
save_npy(data_dir / "embeddings.npy", X)
save_json(data_dir / "ids.json", ids)
print(str(data_dir / "embeddings.npy"))
print(str(data_dir / "ids.json"))
