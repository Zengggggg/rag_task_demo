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

# ---------- Helpers ----------

def _extract_json(text: str) -> str:
    """Tách JSON top-level sạch từ chuỗi có thể kèm rác (fence/log).
    Dùng đếm ngoặc để cắt đúng phần {...} hoặc [...]. Raise nếu không khớp ngoặc.
    """
    if not text:
        return ""

    # 1) Nếu có code fence ```json ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if m:
        cand = m.group(1).strip()
        if cand:
            text = cand  # tiếp tục cắt bằng đếm ngoặc
        else:
            return ""

    # 2) Vị trí bắt đầu JSON
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        return ""
    start = min(starts)

    # 3) Dò ngoặc khớp
    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_str = False
    esc = False
    end = None

    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

    if end is None:
        # Không khớp ngoặc: coi là truncated -> để caller retry
        raise ValueError("Truncated JSON: unmatched braces/brackets in LLM output.")

    return text[start:end].strip()


def _sanitize_minor(s: str) -> str:
    """Sửa nhẹ: bỏ BOM, cắt rác whitespace sau dấu đóng top-level nếu có."""
    if not s:
        return s
    s = s.lstrip("\ufeff")
    last_curly = s.rfind("}")
    last_brack = s.rfind("]")
    cut = max(last_curly, last_brack)
    if cut != -1:
        s = s[:cut + 1]
    return s


def _safe_parse_tasks(raw: str) -> List[Dict[str, Any]]:
    s = _extract_json(raw)
    s = _sanitize_minor(s)
    if not s or not s.strip():
        raise ValueError("LLM trả về rỗng hoặc không tìm thấy JSON.")

    try:
        data = json.loads(s)
    except json.JSONDecodeError as e:
        pos = e.pos
        ctx = s[max(0, pos - 120): pos + 120]
        raise ValueError(
            f"JSON parse fail: {e.msg} at line {e.lineno}, col {e.colno} (char {pos}). "
            f"Context≈\n...{ctx}..."
        ) from e

    # {"tasks":[...]} hoặc trực tiếp [...]
    if isinstance(data, dict) and "tasks" in data:
        tasks = data["tasks"]
    elif isinstance(data, list):
        tasks = data
    else:
        raise ValueError("JSON không đúng dạng {'tasks':[...]} hoặc [...].")

    norm: List[Dict[str, Any]] = []
    for t in tasks or []:
        if not isinstance(t, dict):
            continue
        title = (t.get("title") or "").strip()
        if not title:
            continue

        dep = t.get("departmentId") or t.get("department") or t.get("dept") or ""

        def _to_int(x, default=0):
            try:
                return int(x)
            except Exception:
                return default

        norm.append({
            "title": title,
            "description": (t.get("description") or "").strip(),
            "departmentId": dep,
            "parentId": None,
            "assigneeId": None,
            "status": (t.get("status") or "pending").strip() or "pending",
            "estimate": _to_int(t.get("estimate"), 0),
            "estimateUnit": (t.get("estimateUnit") or "day"),
            "progressPct": _to_int(t.get("progressPct"), 0),
        })
    return norm


def _build_context(retrieved_docs: list) -> str:
    """Ghép ngữ cảnh từ retrieved_docs (list[str] hoặc list[dict]), giới hạn độ dài."""
    lines = []
    if retrieved_docs:
        for d in retrieved_docs:
            if isinstance(d, str):
                if d.strip():
                    lines.append(d)
            elif isinstance(d, dict):
                txt = d.get("text")
                if isinstance(txt, str) and txt.strip():
                    lines.append(txt)
                else:
                    parts = [str(v) for v in d.values() if isinstance(v, str)]
                    if parts:
                        lines.append(" | ".join(parts))
    return "\n".join(lines)[:6000]


def _call_llm(prompt: str, force_json: bool = True, max_toks: int = 1600, temperature: float = 0.2) -> str:
    """Wrapper gọi LLM 1 lần."""
    kwargs = dict(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_toks,
        top_p=1,
    )
    if force_json:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


# ---------- Public API ----------

def generate_tasks(event_input: dict, retrieved_docs: list) -> List[Dict[str, Any]]:
    # Build context
    context = _build_context(retrieved_docs)

    # Prompt: giới hạn số task & mô tả ngắn để tránh bị cắt
    user_prompt = f"""
SỰ KIỆN:
{json.dumps(event_input, ensure_ascii=False, indent=2)}

THAM CHIẾU TỪ GLOBAL KB:
{context if context.strip() else "(không có)"}

YÊU CẦU:
- Sinh ra 6-8 task lớn phù hợp với sự kiện.
- Mỗi description ≤ 35 từ, rõ ràng, không xuống dòng.
- Mapping departmentId theo nội dung (ví dụ: "Media", "Đối ngoại", "Hậu cần", "Tài chính", "Nội dung", "Kỹ thuật"...).
- Trả về JSON THUẦN đúng schema nêu trên, bọc trong object {{ "tasks": [...] }}.
"""

    try:
        raw = _call_llm(user_prompt, force_json=True, max_toks=1600, temperature=0.2)
        return _safe_parse_tasks(raw)

    except Exception as e1:
        # Retry #1: prompt ngắn hơn + ít task hơn
        short_prompt = user_prompt + "\n\nChỉ trả về JSON thuần (không code fence, không giải thích). Tạo đúng 6 task."
        try:
            raw = _call_llm(short_prompt, force_json=True, max_toks=1200, temperature=0.2)
            return _safe_parse_tasks(raw)
        except Exception as e2:
            # Retry #2: bỏ force_json (một số model strict), vẫn yêu cầu JSON thuần
            try:
                raw = _call_llm(short_prompt, force_json=False, max_toks=1200, temperature=0.2)
                return _safe_parse_tasks(raw)
            except Exception as e3:
                # Trả về 1 task lỗi để quan sát nguyên nhân
                return [{
                    "title": "Error parsing LLM output",
                    "description": f"{e3}",
                    "departmentId": "N/A",
                    "parentId": None,
                    "assigneeId": None,
                    "status": "pending",
                    "estimate": 0,
                    "estimateUnit": "day",
                    "progressPct": 0
                }]
