from fastapi import FastAPI, Body
from services.pipeline import run_pipeline

app = FastAPI(title="RAG Task Generator Demo")

@app.post("/generate-tasks")
def generate_tasks(event_input: dict = Body(...)):
    """
    Input ví dụ:
    {
      "name": "Ngày hội việc làm K2C7",
      "description": "Sự kiện outdoor có doanh nghiệp và khách mời VIP",
      "event_type_guess": "Career Fair"
    }
    """
    result = run_pipeline(event_input)
    return result
