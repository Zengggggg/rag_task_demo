import re, json
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

def clean_json_output(text: str) -> str:
    """
    Loại bỏ ```json ... ``` hoặc ``` ... ``` để còn lại JSON thuần.
    """
    text = text.strip()
    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text

def generate_tasks(event_input, retrieved_docs):
    context = ""
    for d in retrieved_docs:
        context += f"### TEMPLATE {d['doc_id']}\n"
        for t in d.get("baseline_tasks", []):
            context += f"- {t['name']} ({t['owner_department']}): {t.get('notes','')}\n"
        context += "\n"

    prompt = f"""
Bạn là trợ lý lập kế hoạch sự kiện nội bộ FPTU.
Sử dụng thông tin sự kiện và mẫu công việc bên dưới để sinh danh sách task lớn.

SỰ KIỆN:
{json.dumps(event_input, ensure_ascii=False, indent=2)}

CONTEXT:
{context}

Trả về **JSON thuần** (không có markdown, không có ký hiệu ```), chỉ là một mảng:
[
  {{
    "title": "Tên công việc lớn",
    "department": "Ban phụ trách",
    "description": "Mô tả ngắn"
  }}
]
"""

    if DEBUG_MODE:
        print("\n===== PROMPT DEBUG =====")
        print(prompt)
        print("========================\n")

    try:
        response = client.responses.create(
            model=LLM_MODEL,
            input=prompt,
            temperature=0.2
        )
        raw_text = response.output_text.strip()
        cleaned = clean_json_output(raw_text)

        if DEBUG_MODE:
            print("===== RAW LLM OUTPUT =====")
            print(raw_text)
            print("===== CLEANED =====")
            print(cleaned)

        return json.loads(cleaned)

    except Exception as e:
        print("⚠️ Lỗi khi parse LLM output:", e)
        return [{
            "title": "Error parsing LLM output",
            "department": "N/A",
            "description": str(e)
        }]
