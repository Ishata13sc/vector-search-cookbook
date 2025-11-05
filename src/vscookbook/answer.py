from typing import List, Dict
import re

def split_sentences(t: str):
    return re.split(r'(?<=[.!?])\s+', t.strip())

def select_sentences(query: str, docs: List[Dict], sent_k: int = 3):
    q = set(re.findall(r'\w+', query.lower()))
    cands = []
    for d in docs:
        sid = str(d.get("id"))
        for s in split_sentences(d.get("text","")):
            w = set(re.findall(r'\w+', s.lower()))
            score = len(q & w) / (len(q) + 1e-9)
            cands.append((score, sid, s))
    cands.sort(key=lambda x: -x[0])
    seen = set()
    out = []
    for sc, sid, s in cands:
        if not s or (sid, s) in seen:
            continue
        out.append({"id": sid, "sentence": s, "score": float(sc)})
        seen.add((sid, s))
        if len(out) >= sent_k:
            break
    return out

def synth_answer(query: str, docs: List[Dict], sent_k: int = 3):
    sents = select_sentences(query, docs, sent_k=sent_k)
    text = " ".join(x["sentence"] for x in sents) if sents else ""
    cites = [{"id": x["id"], "snippet": x["sentence"]} for x in sents]
    return {"answer": text, "citations": cites}
