import json
from pathlib import Path
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
p = root / "data" / "benchmark.json"
with open(p, "r", encoding="utf-8") as f:
    data = json.load(f)

labels = list(data.keys())
r1 = [data[k]["recall_at_1"] for k in labels]
r5 = [data[k]["recall_at_5"] for k in labels]

plt.figure()
plt.bar(labels, r1)
plt.title("Recall@1")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(root / "data" / "recall_at_1.png")

plt.figure()
plt.bar(labels, r5)
plt.title("Recall@5")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(root / "data" / "recall_at_5.png")
