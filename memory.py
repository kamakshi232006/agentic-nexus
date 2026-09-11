#Handles TF-IDF vectorization, cosine similarity ranking, and model memory reconciliation.
import re
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from database import all_mem, add_mem_db, update_mem_db, delete_mem_db

KINDS = ["preference", "fact", "project"]

def _sim(query, texts):
    if not texts:
        return np.array([])
    v = TfidfVectorizer().fit(texts + [query])
    return cosine_similarity(v.transform([query]), v.transform(texts))[0]

def add_mem(kind, content):
    kind = kind if kind in KINDS else "fact"
    rows = all_mem()
    if rows:
        sims = _sim(content, [c for _, _, c in rows])
        if sims.size and sims.max() > 0.75:
            return rows[int(sims.argmax())][0]
    return add_mem_db(kind, content)

def update_mem(i, content):
    update_mem_db(i, content)

def delete_mem(i):
    delete_mem_db(i)

def recall(query, k=4, threshold=0.08):
    rows = all_mem()
    if not rows:
        return []
    sims = _sim(query, [c for _, _, c in rows])
    ranked = sorted(zip(rows, sims), key=lambda x: x[1], reverse=True)
    return [(i, kd, c) for (i, kd, c), s in ranked[:k] if s > threshold]

RECONCILE = """You maintain long-term memory about ONE user.
CURRENT MEMORIES:
{cur}
The user just wrote: "{msg}"
Output ONLY a JSON list of operations (or [] for small talk / one-off questions):
  {{"op":"add","kind":"preference|fact|project","content":"..."}}
  {{"op":"update","id":<id>,"content":"..."}}
  {{"op":"delete","id":<id>}}"""

def reconcile_memory(msg, plain_llm_func, verbose=False):
    cur = "\n".join(f"#{i} [{k}] {c}" for i, k, c in all_mem()) or "(none)"
    raw = plain_llm_func(RECONCILE.format(cur=cur, msg=msg))
    m = re.search(r"\[.*\]", raw, re.S)
    try:
        ops = json.loads(m.group(0)) if m else []
    except Exception:
        ops = []
    touched = []
    for op in ops:
        try:
            if op["op"] == "add":
                touched.append(add_mem(op.get("kind", "fact"), op["content"].strip()))
            elif op["op"] == "update":
                update_mem(int(op["id"]), op["content"].strip()); touched.append(int(op["id"]))
            elif op["op"] == "delete":
                delete_mem(int(op["id"])); touched.append(int(op["id"]))
        except Exception:
            pass
    return touched

def memory_block(recalled):
    prefs = [c for i, k, c in all_mem() if k == "preference"]
    facts = [f"({k}) {c}" for i, k, c in recalled if k != "preference"]
    out = []
    if prefs:
        out.append("User preferences (ALWAYS follow):\n" + "\n".join(f"- {p}" for p in prefs))
    if facts:
        out.append("Relevant about this user:\n" + "\n".join(f"- {f}" for f in facts))
    return "\n\n".join(out)