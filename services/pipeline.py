from .retriever import retrieve_docs
from .llm_generator import generate_tasks

def run_pipeline(event_input: dict):
    """
    event_input: dict có các trường: name, description, event_type_guess, outdoor, has_sponsor, has_vip, ...
    retrieve_docs(event_input) phải trả List[Dict] dạng:
        { "doc_id": str, "text": str, "metadata": {"similarity": float, ...} }
    và đã được sort sẵn theo độ liên quan (phần tử đầu là top-1).
    """
    # 1) Truy hồi tài liệu
    retrieved = retrieve_docs(event_input) or []

    # 2) Chọn top-1 (áp ngưỡng similarity nếu có)
    MIN_SIM = 0.30  # tuỳ dữ liệu; giảm/tăng nếu cần
    top1 = None
    if retrieved:
        # ưu tiên doc có similarity >= MIN_SIM
        cand = [d for d in retrieved
                if isinstance(d, dict) and (d.get("metadata", {}).get("similarity") or 0) >= MIN_SIM]
        if cand:
            top1 = cand[0]
        else:
            # fallback: lấy phần tử đầu (đã sort)
            top1 = retrieved[0]

    top_context_docs = [top1] if top1 else []  # truyền vào LLM (llm_generator đã hỗ trợ list[dict]/list[str])

    # 3) Gọi LLM sinh tasks chỉ với context top-1
    tasks = generate_tasks(event_input, top_context_docs)

    # 4) Build response: chỉ hiển thị ID của tài liệu liên quan nhất
    top_ids = [top1["doc_id"]] if top1 else []

    return {
        "event": event_input,
        "retrieved_docs": top_ids,  # ← giờ là ["concert_festival"] thay vì cả danh sách
        "tasks": tasks
    }

