from .retriever import retrieve_docs
from .llm_generator import generate_tasks

def run_pipeline(event_input: dict):
    desc = event_input.get("description") or event_input.get("name")
    retrieved = retrieve_docs(desc)
    tasks = generate_tasks(event_input, retrieved)
    return {"event": event_input, "retrieved_docs": [d["doc_id"] for d in retrieved], "tasks": tasks}
