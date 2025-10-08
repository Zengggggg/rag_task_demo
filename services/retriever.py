# services/retriever.py
import os
from typing import List, Dict, Any, Optional

import chromadb
from sentence_transformers import SentenceTransformer
# services/retriever.py
import logging
logger = logging.getLogger("uvicorn.error")

# ====== ENV / DEFAULTS ======
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "global_kb")

# ====== SINGLETONS ======
_embedder: Optional[SentenceTransformer] = None
_client: Optional[chromadb.PersistentClient] = None
_collection = None


def _get_embedder() -> SentenceTransformer:
    """Lazy init embedder (CPU)."""
    global _embedder
    if _embedder is None:
        # Không cần TF/Keras cho sentence-transformers
        os.environ.setdefault("USE_TF", "0")
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _get_collection():
    """Lazy init Chroma collection."""
    global _client, _collection
    if _client is None:
        # Tắt telemetry cho sạch log/dev
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
    if _collection is None:
        _collection = _client.get_or_create_collection(
            CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
        )
    return _collection


def _build_filters(event_input: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    TODO:
    - Xem xét thêm các filter khác (nếu có trong metadata)
    - Xem xét thêm logic kết hợp nhiều doc (top-k) cho context
    - Xem xét thêm logic threshold similarity (nếu cần)
    Hiện tại đang không filter.
    """
    clauses = []

    # chỉ add khi True
    if event_input.get("has_vip") is True:
        clauses.append({"tag_vip": {"$eq": True}})
    if event_input.get("has_sponsor") is True:
        clauses.append({"tag_sponsor": {"$eq": True}})
    if event_input.get("outdoor") is True:
        clauses.append({"tag_outdoor": {"$eq": True}})

    # so sánh lowercase để tránh lệch hoa/thường
    etg = (event_input.get("event_type_guess") or "").strip().lower()
    if etg:
        clauses.append({"event_type_primary_lower": {"$eq": etg}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return None




def retrieve_global_passages(
    query: str,
    top_k: int = 1,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not query or not query.strip():
        return []

    collection = _get_collection()
    embedder = _get_embedder()

    q_emb = embedder.encode([query]).tolist()[0]

    # ===== DEBUG START =====
    print("🧩 Query:", query[:100], flush=True)
    print("🔍 Filters:", filters, flush=True)
    # ===== DEBUG END =====

    # include để lấy thêm distances/scores + docs/metas
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        where=(filters if filters else None),
        include=["documents", "metadatas", "distances"]
    )

    ids = res.get("ids") or []
    if not ids or not ids[0]:
        print("📦 Retrieved IDs: []", flush=True)
        return []

    top_id   = ids[0][0]
    top_dist = (res.get("distances") or [[None]])[0][0]
    top_meta = (res.get("metadatas") or [[{}]])[0][0] or {}
    top_title = top_meta.get("title") or top_meta.get("name") or top_meta.get("slug") or ""

    print(f"🏆 Top Retrieved ID: {top_id} | distance={top_dist}", flush=True)
    if top_title:
        print(f"   ↳ title: {top_title}", flush=True)
    print("📦 Retrieved IDs (sorted):", ids[0], flush=True)

    # Chuẩn hoá output
    out: List[Dict[str, Any]] = []
    docs = res.get("documents") or [[]]
    metas = res.get("metadatas") or [[]]
    dists = res.get("distances") or [[]]

    for i in range(len(ids[0])):
        doc_id = ids[0][i]
        text = docs[0][i] if len(docs[0]) > i else ""
        meta = metas[0][i] if len(metas[0]) > i else {}
        dist = dists[0][i] if len(dists[0]) > i else None
        if isinstance(meta, dict):
            meta.setdefault("doc_id", doc_id)
            if dist is not None:
                meta["distance"] = dist
        out.append({"doc_id": doc_id, "text": text, "metadata": meta})

    return out




def retrieve_docs(event_input: Dict[str, Any], top_k: int = 12) -> List[Dict[str, Any]]:
    """
    API cho pipeline:
      - Lấy query từ description/name
      - Tạo filters từ event_input
      - Query Chroma
    """
    if not isinstance(event_input, dict):
        return []

    query = (event_input.get("description") or event_input.get("name") or "").strip()
    if not query:
        return []

    filters = _build_filters(event_input)
    return retrieve_global_passages(query=query, top_k=top_k, filters=filters)
