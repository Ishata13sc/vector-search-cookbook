from pathlib import Path
import json, csv

def load_jsonl_file(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def load_csv_file(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({"id": str(row.get("id","")), "text": str(row.get("text",""))})
    return rows

def to_items(rows):
    out = []
    for r in rows:
        i = {"id": str(r.get("id","")),"text": str(r.get("text",""))}
        if i["id"] and i["text"]:
            out.append(i)
    return out
