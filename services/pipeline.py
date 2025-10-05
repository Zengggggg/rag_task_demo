from .retriever import retrieve_docs
from .llm_generator import generate_tasks

def run_pipeline(event_input: dict):
    # event_input là dict có các trường: name, description, event_type_guess, outdoor, has_sponsor, has_vip, ...
    retrieved = retrieve_docs(event_input)  # ✅ Truyền toàn bộ dict, không chỉ description
    tasks = generate_tasks(event_input, retrieved)

    return {
        "event": event_input,
        "retrieved_docs": [d["doc_id"] for d in retrieved],
        "tasks": tasks
    }
