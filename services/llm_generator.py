# services/llm_generator.py
import os, json, re
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()  # load .env

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# YÊU CẦU LLM TRẢ VỀ JSON THUẦN (mảng tasks) MAP THEO ATTRIBUTES CỦA BẠN
SYSTEM_PROMPT = """Bạn là trợ lý sinh các đầu việc lớn (big tasks) cho sự kiện.
QUY TẮC:
- Chỉ sinh CÁC TASK LỚN (không sinh subtasks).
- Mỗi task lớn KHÔNG có assigneeId, chỉ có departmentId.
- Timeline chỉ dựa trên task lớn (nếu có estimate/đơn vị).
- Trả về JSON THUẦN theo dạng: {"tasks":[ { ... }, ... ]} (KHÔNG có giải thích).

Schema mỗi task:
{
  "title": "string",
  "description": "string",
  "departmentId": "string",
  "parentId": null,
  "assigneeId": null,
  "status": "pending",
  "estimate": 0,
  "estimateUnit": "day",
  "progressPct": 0
}
Lưu ý: 
- parentId luôn null (vì đây là task lớn).
- assigneeId luôn null (chỉ task nhỏ mới có).
- status mặc định "pending".
- estimate là số nguyên >=0; estimateUnit một trong ["hour","day","week"], mặc định "day".
- progressPct = 0.
"""

# Fallback parser: nếu LLM trả kèm text/fence, bóc JSON ra
def _extract_json(text: str) -> str:
    if not text:
        return ""
    # Ưu tiên code fence ```json ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Nếu có ngoặc mở đầu tiên là { hoặc [, cắt từ đó
    idx = min([i for i in [text.find("{"), text.find("[")] if i != -1], default=-1)
    return text[idx:].strip() if idx >= 0 else text.strip()

def _safe_parse_tasks(raw: str) -> List[Dict[str, Any]]:
    s = _extract_json(raw)
    data = json.loads(s)  # sẽ raise nếu không hợp lệ
    # chấp nhận {"tasks":[...]} hoặc trực tiếp [...]
    if isinstance(data, dict) and "tasks" in data:
        tasks = data["tasks"]
    elif isinstance(data, list):
        tasks = data
    else:
        raise ValueError("JSON không đúng dạng {'tasks':[...]} hoặc [...].")
    # Chuẩn hoá field bắt buộc
    norm = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        norm.append({
            "title": t.get("title", "").strip(),
            "description": t.get("description", "").strip(),
            "departmentId": t.get("departmentId") or t.get("department") or "",
            "parentId": None,                 # big task
            "assigneeId": None,               # big task không có assignee
            "status": t.get("status", "pending"),
            "estimate": int(t.get("estimate", 0) or 0),
            "estimateUnit": t.get("estimateUnit", "day"),
            "progressPct": int(t.get("progressPct", 0) or 0),
        })
    # lọc empty title
    return [x for x in norm if x["title"]]
    

def generate_tasks(event_input: dict, retrieved_docs: list) -> List[Dict[str, Any]]:
    # Ghép ngữ cảnh từ retrieved_docs (nếu có)
    context = "\n".join([d.get("text","") for d in retrieved_docs]) if retrieved_docs else ""
    user_prompt = f"""
SỰ KIỆN:
{json.dumps(event_input, ensure_ascii=False, indent=2)}

THAM CHIẾU TỪ GLOBAL KB:
{context if context.strip() else "(không có)"}

YÊU CẦU:
- Sinh ra 6-10 task lớn phù hợp với sự kiện.
- Mapping departmentId theo nội dung (ví dụ: "Media", "Đối ngoại", "Hậu cần", "Tài chính", "Nội dung", "Kỹ thuật"...).
- Trả về JSON THUẦN đúng schema nêu trên, bọc trong object {{ "tasks": [...] }}.
"""

    try:
        # DÙNG JSON MODE để bắt buộc trả JSON thuần
        resp = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            response_format={"type": "json_object"},  # <<== ép JSON
            max_tokens=800,
        )
        raw = resp.choices[0].message.content or ""
        return _safe_parse_tasks(raw)

    except Exception as e:
        # Fallback lần 2: thử gọi lại KHÔNG response_format (đôi khi model strict quá)
        try:
            resp = client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt + "\n\nChỉ trả về JSON thuần, không kèm giải thích."},
                ],
                temperature=0.4,
                max_tokens=800,
            )
            raw = resp.choices[0].message.content or ""
            return _safe_parse_tasks(raw)
        except Exception as e2:
            # Trả lỗi gọn để bạn thấy nguyên nhân
            return [{
                "title": "Error parsing LLM output",
                "description": f"{e2}",
                "departmentId": "N/A",
                "parentId": None,
                "assigneeId": None,
                "status": "pending",
                "estimate": 0,
                "estimateUnit": "day",
                "progressPct": 0
            }]
