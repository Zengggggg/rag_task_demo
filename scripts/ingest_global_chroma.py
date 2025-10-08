import os, json
from sentence_transformers import SentenceTransformer
import chromadb

# ====== Cấu hình ======
DATA_DIR = "./kb/global"        # Thư mục chứa các file JSON tài liệu
CHROMA_DIR = "./chroma_db"           # Thư mục lưu vector database
COLLECTION_NAME = "global_kb"        # Tên collection trong Chroma
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")


# ====== Chuẩn bị ChromaDB và Embedder ======
def get_chroma_collection():
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    return col


def get_embedder():
    os.environ.setdefault("USE_TF", "0")
    return SentenceTransformer(EMBED_MODEL)


# ====== Hàm xử lý ======
def build_metadata(doc: dict):
    ctx_tags = doc.get("context_tags", [])
    etypes = doc.get("event_type", [])
    primary = etypes[0].strip() if etypes else ""

    return {
        "event_type_primary": primary,
        "event_type_primary_lower": primary.lower(),  # 👈 bắt buộc
        "tag_vip": "vip" in ctx_tags,
        "tag_sponsor": "sponsor" in ctx_tags,
        "tag_outdoor": "outdoor" in ctx_tags,
    }
def build_metadata(doc: dict):
    ctx_tags = doc.get("context_tags", [])
    etypes = doc.get("event_type", [])
    primary = etypes[0].strip() if etypes else ""

    return {
        "event_type_primary": primary,
        "event_type_primary_lower": primary.lower(),  # 👈 bắt buộc
        "tag_vip": "vip" in ctx_tags,
        "tag_sponsor": "sponsor" in ctx_tags,
        "tag_outdoor": "outdoor" in ctx_tags,
    }



def ingest():
    print("🚀 Bắt đầu ingest dữ liệu global KB...")
    embedder = get_embedder()
    col = get_chroma_collection()

    all_docs, all_ids, all_metas, all_embs = [], [], [], []

    # ====== Đọc từng file JSON ======
    for file in os.listdir(DATA_DIR):
        if not file.endswith(".json"):
            continue
        path = os.path.join(DATA_DIR, file)
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)

        doc_id = doc.get("doc_id") or os.path.splitext(file)[0]
        meta = build_metadata(doc)

        # Biến từng baseline_task thành 1 đoạn văn độc lập
        tasks = doc.get("baseline_tasks", [])
        text_parts = []
        for t in tasks:
            part = f"{t['name']} ({t['owner_department']}): {t['notes']}"
            text_parts.append(part)
        text_join = "\n".join(text_parts)

        all_ids.append(doc_id)
        all_docs.append(text_join)
        all_metas.append(meta)

    # ====== Tạo embeddings ======
    print(f"🔢 Tạo embeddings cho {len(all_docs)} tài liệu...")
    all_embs = embedder.encode(all_docs, show_progress_bar=True).tolist()

    # ====== Upsert vào Chroma ======
    print(f"📦 Upsert vào collection '{COLLECTION_NAME}' ...")
    col.upsert(ids=all_ids, documents=all_docs, metadatas=all_metas, embeddings=all_embs)

    print("✅ Ingest hoàn tất!")
    print(f"📂 Tổng cộng: {len(all_ids)} tài liệu được thêm vào.")


if __name__ == "__main__":
    ingest()
