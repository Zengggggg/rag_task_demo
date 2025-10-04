from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json, os, numpy as np

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def load_kb(folder_path):
    docs = []
    for fn in os.listdir(folder_path):
        if fn.endswith(".json"):
            path = os.path.join(folder_path, fn)
            docs.append(json.load(open(path, encoding="utf-8")))
    return docs

GLOBAL_KB = load_kb("kb/global")
USER_KB = load_kb("kb/user")

def retrieve_docs(event_desc: str, top_k: int = 2):
    """Lấy top tài liệu gần nhất từ User KB + Global KB"""
    query_emb = embedder.encode([event_desc])
    all_docs = GLOBAL_KB + USER_KB
    texts = [" ".join(d.get("event_type", []) + d.get("context_tags", [])) for d in all_docs]
    doc_emb = embedder.encode(texts)
    sims = cosine_similarity(query_emb, doc_emb)[0]
    idx = np.argsort(-sims)[:top_k]
    return [all_docs[i] for i in idx]
