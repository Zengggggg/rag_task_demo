from fastapi import FastAPI, HTTPException
from models.schemas import EventInput
from services.pipeline import run_pipeline
import logging
logging.basicConfig(
    level=logging.INFO,  # hoặc DEBUG
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("uvicorn.error")  # bám theo logger của uvicorn
logger.setLevel(logging.DEBUG)

app = FastAPI(title="Task Generation RAG API")

@app.post("/generate-tasks")
def generate_tasks(payload: EventInput):
    # Chỉ chấp nhận format phẳng (không có trường 'event')
    # FastAPI sẽ tự 422 nếu client gửi {"event": {...}}
    data = payload.model_dump(exclude_none=True)

    if not (data.get("name") or data.get("description")):
        raise HTTPException(status_code=400, detail="Missing 'name' or 'description'")

    result = run_pipeline(data)
    return result

@app.get("/")
def root():
    return {"message": "RAG Task Generator API is running"}
