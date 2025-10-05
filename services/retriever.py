# services/retriever.py
import os
from typing import List, Dict, Any, Optional

import chromadb
from sentence_transformers import SentenceTransformer


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
    if event_input.get("has_vip"):
        return {"tag_vip": {"$eq": True}}
    if event_input.get("has_sponsor"):
        return {"tag_sponsor": {"$eq": True}}
    if event_input.get("outdoor"):
        return {"tag_outdoor": {"$eq": True}}
    etg = (event_input.get("event_type_guess") or "").strip()
    if etg:
        return {"event_type_primary": {"$eq": etg}}
    return None


def retrieve_global_passages(
    query: str,
    top_k: int = 12,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Tìm văn bản gần nhất trong Global KB (Chroma).
    Trả về: [{ doc_id, text, metadata }]
    """
    if not query or not query.strip():
        return []

    collection = _get_collection()
    embedder = _get_embedder()

    q_emb = embedder.encode([query]).tolist()[0]
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        where=filters or {},
    )

    if not res or not res.get("ids") or not res["ids"]:
        return []

    out: List[Dict[str, Any]] = []
    for i in range(len(res["ids"][0])):
        meta = res["metadatas"][0][i] if res.get("metadatas") else {}
        # Khi ingest, ta upsert ids=doc_id → có thể lấy trực tiếp
        doc_id = res["ids"][0][i]
        text = res["documents"][0][i] if res.get("documents") else ""
        # Gắn doc_id vào metadata cho tiện debug (không bắt buộc)
        if isinstance(meta, dict):
            meta.setdefault("doc_id", doc_id)
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
