from pathlib import Path
import json, random, string

root = Path(__file__).resolve().parents[1]
ds = root / "data" / "datasets" / "uploaded"
ds.mkdir(parents=True, exist_ok=True)

base = [
    {"id":"d1","text":"Cara membuat rawon khas Surabaya dengan kluwek."},
    {"id":"d4","text":"Hybrid search menggabungkan BM25 dan embedding."},
    {"id":"d5","text":"Optimasi indeks vektor pada CPU dengan memori terbatas."},
    {"id":"d7","text":"Arsitektur transformer dan representasi semantik kalimat."},
    {"id":"d10","text":"Manajemen data lapangan PK-24 dan integrasi ke SIGA."},
]

paraphrase = [
    ("d4","Pencarian hibrida yang memadukan pengindeksan BM25 serta embedding neural."),
    ("d4","Metode hybrid search yang mengombinasikan BM25 dengan representasi vektor."),
    ("d7","Gambaran arsitektur transformer untuk representasi semantik teks."),
    ("d5","Tuning indeks vektor di CPU saat RAM terbatas."),
    ("d10","Workflow data PK-24 terhubung ke SIGA."),
    ("d1","Langkah membuat rawon memakai kluwek ala Surabaya."),
]

def make_typos(t, p=0.08):
    r = []
    for ch in t:
        if random.random()<p and ch.isalpha():
            r.append(random.choice(string.ascii_lowercase))
        else:
            r.append(ch)
    return "".join(r)

def rand_kw_noise(k=12):
    kws = ["search","vector","bm25","index","cpu","gpu","data","system","neural","token","java","python","dataset","cors","http","react","linux","windows","docker","kafka","postgres","faiss","hnsw","ivf","random"]
    return " ".join(random.choices(kws, k=k))

near = [{"id":f"p{i}","text":txt} for i,(rid,txt) in enumerate(paraphrase,1)]
typos = [{"id":f"t{i}","text":make_typos(b["text"],0.12)} for i,b in enumerate(base,1)]
noise = [{"id":f"z{i}","text":rand_kw_noise(random.randint(8,24))} for i in range(800)]

items = base + near + typos + noise
p = ds / "data.jsonl"
with open(p,"w",encoding="utf-8") as f:
    for r in items:
        f.write(json.dumps(r,ensure_ascii=False)+"\n")
print(str(p), len(items))
