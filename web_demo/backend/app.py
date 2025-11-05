from fastapi import FastAPI, Query as Q, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import csv, json, uuid, os, time, hashlib
from typing import List, Dict, Any, Optional
from src.vscookbook.embedder import TextEmbedder
from src.vscookbook.datasets import load_sample_texts
from src.vscookbook.index.factory import create_index
from src.vscookbook.index.serialize import save_index, load_index
from src.vscookbook.lexical.bm25 import BM25Index
from src.vscookbook.text import tokenize
from src.vscookbook.hybrid import fuse_sum
from src.vscookbook.bench import bench_index, bench_hybrid
from src.vscookbook.io import save_npy, load_npy, save_json, load_json
from src.vscookbook.ingest.loader import load_jsonl_file, load_csv_file, to_items
from src.vscookbook.rerank import Reranker
from src.vscookbook.answer import synth_answer

class Query(BaseModel):
    text: str
    k: Optional[int] = 5

app = FastAPI(title="Vector Search Cookbook Demo")
root_dir = Path(__file__).resolve().parents[1]
frontend_dir = root_dir / "frontend"
static_dir = frontend_dir / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

data_root = root_dir.parents[1] / "data"
ds_root = data_root / "datasets"
ix_root = data_root / "indexes"
uploads_dir = data_root / "uploads"
tmp_dir = uploads_dir / "tmp"
ds_root.mkdir(parents=True, exist_ok=True)
ix_root.mkdir(parents=True, exist_ok=True)
uploads_dir.mkdir(parents=True, exist_ok=True)
tmp_dir.mkdir(parents=True, exist_ok=True)

API_KEY = os.getenv("VSC_DEMO_KEY", "").strip()

def require_key(h: Optional[str]):
    if API_KEY and h != API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")

embedder_name = os.getenv("VSC_EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2")
embedder = TextEmbedder(embedder_name) if hasattr(TextEmbedder, "__call__") or True else TextEmbedder()
reranker = Reranker()
dataset_name = "sample"
last_payload: dict = {}
kind_current = "naive"
texts: List[str] = []
ids: List[str] = []
bm25 = None
index = None
X = None

EMB_CACHE: Dict[str, Any] = {}
RERANK_CACHE: Dict[str, Any] = {}
ANSWER_CACHE: Dict[str, Any] = {}

def kkey(*parts):
    raw = "||".join([str(p) for p in parts])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def model_dir_name():
    m = embedder_name.replace("/", "_").replace(":", "_")
    return m

def load_dataset(name: str):
    if name == "sample":
        items = load_sample_texts()
    else:
        p = ds_root / name / "data.jsonl"
        items = load_jsonl_file(p) if p.exists() else []
    return items

def encode_texts(arr: List[str]):
    key = kkey(embedder_name, tuple(arr))
    if key in EMB_CACHE:
        return EMB_CACHE[key]
    v = embedder.encode(arr) if arr else None
    EMB_CACHE[key] = v
    return v

def ensure_embeddings(name: str, items):
    d = ds_root / name / model_dir_name()
    d.mkdir(parents=True, exist_ok=True)
    pE = d / "embeddings.npy"
    pI = d / "ids.json"
    if pE.exists() and pI.exists():
        X = load_npy(pE)
        ids = load_json(pI)
        texts = [x["text"] for x in items] if items else []
        return texts, ids, X
    texts = [x["text"] for x in items]
    ids = [x["id"] for x in items]
    X = encode_texts(texts) if texts else None
    if X is not None:
        save_npy(pE, X)
        save_json(pI, ids)
    return texts, ids, X

def rebuild_state(name: str, kind: str = "naive"):
    global dataset_name, texts, ids, X, bm25, index, kind_current
    items = load_dataset(name)
    texts, ids, X = ensure_embeddings(name, items)
    bm25 = BM25Index(texts, ids, tokenize) if texts else BM25Index([""], ["d0"], tokenize)
    kind_current = kind
    index = create_index(kind_current, X if X is not None else encode_texts([""]), ids if X is not None else ["d0"])
    dataset_name = name

rebuild_state("sample", "naive")

@app.get("/")
def root():
    return FileResponse(frontend_dir / "index.html")

@app.get("/api/cache_info")
def api_cache_info():
    return {"emb_keys": len(EMB_CACHE), "rerank_keys": len(RERANK_CACHE), "answer_keys": len(ANSWER_CACHE)}

@app.post("/api/cache_clear")
def api_cache_clear(x_auth: Optional[str] = Header(default=None)):
    require_key(x_auth)
    EMB_CACHE.clear()
    RERANK_CACHE.clear()
    ANSWER_CACHE.clear()
    return {"ok": True}

# --- existing imports & globals ---
# embedder, dataset_name, texts, ids, X, bm25, index, kind_current

@app.post("/api/set_embedder")
def api_set_embedder(name: str, reembed: bool = False, kind: str = "naive"):
    global embedder
    embedder = TextEmbedder(name)
    if reembed:
        rebuild_state(dataset_name, kind)
        return {"ok": True, "changed": name, "reembed": True, "dataset": dataset_name, "index": kind_current}
    return {"ok": True, "changed": name, "reembed": False}

@app.post("/api/reembed")
def api_reembed(kind: str = "naive"):
    rebuild_state(dataset_name, kind)
    return {"ok": True, "dataset": dataset_name, "index": kind_current, "size": len(ids)}


@app.post("/api/embed")
def api_embed(q: Query):
    vec = encode_texts([q.text])[0]
    return {"dim": int(len(vec)), "embedding_head": [float(x) for x in vec[:8]]}

@app.post("/api/search")
def api_search(q: Query):
    global last_payload
    k = q.k or 5
    v = encode_texts([q.text])
    res = index.search(v[0], k=k)
    id_to_text = {str(i): t for i, t in zip(ids, texts)}
    for r in res:
        r["text"] = id_to_text.get(str(r["id"]), "")
    payload = {"query": q.text, "dataset": dataset_name, "index": kind_current, "results": res}
    last_payload = payload
    return payload

@app.post("/api/reindex")
def api_reindex(kind: str = "naive", nlist: int = 8, nprobe: int = 2, M: int = 10, x_auth: Optional[str] = Header(default=None)):
    require_key(x_auth)
    global index, kind_current
    if kind == "ivf":
        index = create_index("ivf", X, ids, nlist=nlist, nprobe=nprobe)
        kind_current = "ivf"
    elif kind == "graph":
        index = create_index("graph", X, ids, M=M)
        kind_current = "graph"
    else:
        index = create_index("naive", X, ids)
        kind_current = "naive"
    return {"ok": True, "index": kind_current, "params": {"nlist": nlist, "nprobe": nprobe, "M": M}}

@app.get("/api/tune_ivf")
def api_tune_ivf(nlist: int = 32, nprobe: int = 4, k: int = 5):
    t0 = time.time()
    ok = False
    try:
        _ = create_index("ivf", X, ids, nlist=nlist, nprobe=nprobe)
        ok = True
    except Exception as e:
        return {"ok": False, "error": str(e)}
    dt = (time.time() - t0) * 1000.0
    return {"ok": ok, "nlist": nlist, "nprobe": nprobe, "dry_ms": dt}

@app.post("/api/hybrid")
def api_hybrid(q: Query, alpha: float = 0.5):
    global last_payload
    k = q.k or 5
    v = encode_texts([q.text])[0]
    vres = index.search(v, k=max(10, k))
    lres = bm25.search(q.text, k=max(10, k), tokenizer=tokenize)
    fused = fuse_sum(vres, lres, topk=k, alpha=alpha)
    id_to_text = {str(i): t for i, t in zip(ids, texts)}
    for r in fused:
        r["text"] = id_to_text.get(str(r["id"]), "")
    payload = {"query": q.text, "dataset": dataset_name, "index": kind_current, "alpha": alpha, "results": fused}
    last_payload = payload
    return payload

@app.get("/api/benchmark")
def api_benchmark(k: int = 5, alpha: float = 0.5):
    res_idx = bench_index(embedder, index, texts, ids, k=k)
    res_hyb = bench_hybrid(embedder, index, bm25, texts, ids, k=k, alpha=alpha)
    return {"k": k, "dataset": dataset_name, "index_kind": kind_current, "vector": res_idx, "hybrid": res_hyb}

def sniff_csv_header(p: Path, limit: int = 20):
    rows = []
    with p.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        cols = list(rdr.fieldnames or [])
        for i, r in enumerate(rdr):
            if i >= limit:
                break
            rows.append(r)
    return cols, rows

def sniff_jsonl_header(p: Path, limit: int = 20):
    cols = set()
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            try:
                obj = json.loads(line.strip())
            except:
                obj = {}
            rows.append(obj)
            for k in obj.keys():
                cols.add(k)
    return list(cols), rows

@app.post("/api/inspect_file")
async def api_inspect_file(file: UploadFile = File(...)):
    name = (file.filename or "upload")
    token = uuid.uuid4().hex
    p = tmp_dir / f"{token}__{name}"
    with p.open("wb") as f:
        f.write(await file.read())
    lower = name.lower()
    if lower.endswith(".csv"):
        kind = "csv"
        cols, sample = sniff_csv_header(p)
    else:
        kind = "jsonl"
        cols, sample = sniff_jsonl_header(p)
    return {"ok": True, "token": token, "filename": name, "kind": kind, "columns": cols, "sample": sample[:10]}

def map_and_clean(rows: List[Dict[str, Any]], id_col: str, text_col: str, lower: bool = True, dedup: bool = True):
    items = []
    seen = set()
    for r in rows:
        i = str(r.get(id_col, "")).strip()
        t = str(r.get(text_col, "")).strip()
        if lower:
            t = t.lower()
        if not t:
            continue
        key = t if dedup else f"{i}::{t}"
        if dedup and key in seen:
            continue
        seen.add(key)
        if not i:
            i = f"d{len(items)+1}"
        items.append({"id": i, "text": t})
    return items

@app.post("/api/import_uploaded")
async def api_import_uploaded(token: str = Form(...), id_col: str = Form(...), text_col: str = Form(...), name: str = Form("uploaded"), lower_text: bool = Form(True), dedup_text: bool = Form(True), x_auth: Optional[str] = Header(default=None)):
    require_key(x_auth)
    path = None
    for p in tmp_dir.glob(f"{token}__*"):
        path = p
        break
    if path is None or not path.exists():
        return {"ok": False, "reason": "token_not_found"}
    if str(path).lower().endswith(".csv"):
        rows = load_csv_file(path)
    else:
        rows = load_jsonl_file(path)
    mapped = map_and_clean(rows, id_col=id_col, text_col=text_col, lower=lower_text, dedup=dedup_text)
    d = ds_root / name
    d.mkdir(parents=True, exist_ok=True)
    pjsonl = d / "data.jsonl"
    with pjsonl.open("w", encoding="utf-8") as f:
        for r in mapped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    global texts, ids, X
    texts = [x["text"] for x in mapped]
    ids = [x["id"] for x in mapped]
    X = encode_texts(texts) if texts else None
    rebuild_state(name, kind_current)
    return {"ok": True, "dataset": name, "count": len(mapped)}

@app.post("/api/switch_dataset")
def api_switch_dataset(name: str = "sample", kind: str = "naive"):
    rebuild_state(name, kind)
    return {"ok": True, "dataset": dataset_name, "index": kind_current, "size": len(ids)}

@app.post("/api/save_index")
def api_save_index(x_auth: Optional[str] = Header(default=None)):
    require_key(x_auth)
    p = ix_root / kind_current
    save_index(p, kind_current, index)
    return {"ok": True, "path": str(p)}

@app.post("/api/load_index")
def api_load_index(kind: str = "naive", x_auth: Optional[str] = Header(default=None)):
    require_key(x_auth)
    global index, kind_current
    d = ix_root / kind
    kind_loaded, ix = load_index(d)
    index = ix
    kind_current = kind_loaded
    return {"ok": True, "index": kind_current}

@app.get("/api/list_indexes")
def api_list_indexes():
    names = []
    if ix_root.exists():
        for p in ix_root.iterdir():
            if p.is_dir():
                names.append(p.name)
    return {"indexes": names}

@app.post("/api/rerank")
def api_rerank(q: Query, mode: str = "hybrid", alpha: float = 0.5, pool: int = 25):
    global last_payload
    ck = kkey("rerank", dataset_name, embedder_name, mode, alpha, pool, q.text, q.k or 5)
    if ck in RERANK_CACHE:
        ranked = RERANK_CACHE[ck]
    else:
        if mode == "bm25":
            cands = bm25.search(q.text, k=max(pool, q.k or 5), tokenizer=tokenize)
        elif mode == "hybrid":
            v = encode_texts([q.text])[0]
            vres = index.search(v, k=max(pool, q.k or 5))
            lres = bm25.search(q.text, k=max(pool, q.k or 5), tokenizer=tokenize)
            cands = fuse_sum(vres, lres, topk=max(pool, q.k or 5), alpha=alpha)
        else:
            v = encode_texts([q.text])[0]
            cands = index.search(v, k=max(pool, q.k or 5))
        id_to_text = {str(i): t for i, t in zip(ids, texts)}
        for r in cands:
            r["text"] = id_to_text.get(str(r["id"]), "")
        ranked = reranker.rerank(q.text, cands, k=q.k or 5)
        RERANK_CACHE[ck] = ranked
    payload = {"query": q.text, "dataset": dataset_name, "mode": mode, "alpha": alpha, "results": ranked}
    last_payload = payload
    return payload

@app.post("/api/answer")
def api_answer(q: Query, mode: str = "hybrid", alpha: float = 0.3, pool: int = 25, sent_k: int = 3):
    global last_payload
    k = q.k or 5
    ck = kkey("answer", dataset_name, embedder_name, mode, alpha, pool, k, sent_k, q.text)
    if ck in ANSWER_CACHE:
        payload = ANSWER_CACHE[ck]
        last_payload = payload
        return payload
    if mode == "rerank":
        c = api_rerank(q, mode="hybrid", alpha=alpha, pool=pool)["results"]
        picked = c[:k]
    elif mode == "bm25":
        picked = bm25.search(q.text, k=max(pool, k), tokenizer=tokenize)[:k]
    elif mode == "hybrid":
        v = encode_texts([q.text])[0]
        vres = index.search(v, k=max(pool, k))
        lres = bm25.search(q.text, k=max(pool, k), tokenizer=tokenize)
        picked = fuse_sum(vres, lres, topk=max(pool, k), alpha=alpha)[:k]
    else:
        v = encode_texts([q.text])[0]
        picked = index.search(v, k=k)
    id_to_text = {str(i): t for i, t in zip(ids, texts)}
    for r in picked:
        r["text"] = id_to_text.get(str(r["id"]), "")
    ans = synth_answer(q.text, picked, sent_k=sent_k)
    payload = {"query": q.text, "dataset": dataset_name, "mode": mode, "alpha": alpha, "pool": pool, "k": k, "result": ans, "candidates": picked}
    ANSWER_CACHE[ck] = payload
    last_payload = payload
    return payload

@app.get("/api/export_last")
def api_export_last():
    if not last_payload:
        return {"ok": False, "reason": "empty"}
    if "result" in last_payload:
        qtxt = str(last_payload.get("query", ""))
        mode = str(last_payload.get("mode", last_payload.get("index", "-")))
        alpha = last_payload.get("alpha", "")
        pool = last_payload.get("pool", "")
        k = last_payload.get("k", "")
        ans = last_payload["result"].get("answer", "")
        cits = last_payload["result"].get("citations", [])
        lines = ["# Vector Search Cookbook — Export", "", "## Query", "", qtxt, "", "## Params", "", f"- mode: **{mode}**", f"- alpha: **{alpha}**", f"- pool: **{pool}**", f"- k: **{k}**", "", "## Answer", "", ans, "", "## Citations", ""]
        for c in cits:
            cid = str(c.get("id", "-"))
            sn = c.get("snippet", "")
            lines.append(f"- **{cid}**: {sn}")
        content = "\n".join(lines)
        return {"ok": True, "filename": "vscookbook_export.md", "content": content}
    content = json.dumps(last_payload, ensure_ascii=False, indent=2)
    return {"ok": True, "filename": "vscookbook_export.json", "content": content}
