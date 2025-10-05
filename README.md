# 🤖 Hệ thống AI RAG Task Generator – Module Phân chia công việc sự kiện (Model 1)

---

## 1. Giới thiệu hệ thống

**AI RAG Task Generator** là **Model 1** trong chuỗi mô hình AI phục vụ nền tảng quản lý sự kiện nội bộ **myFEvent** tại Đại học FPT.

Mục tiêu chính:
- Tự động **sinh ra các đầu việc lớn (macro tasks)** cho từng ban (Trưởng ban, TBTC, Media, Đối ngoại, Hậu cần, v.v.)
- Kết hợp **kiến trúc RAG (Retrieval-Augmented Generation)** và **mô hình ngôn ngữ lớn (LLM)** để sinh nội dung phù hợp với loại sự kiện.

Hệ thống hoạt động như một **API nội bộ** dùng để:
- Nhận mô tả sự kiện → truy hồi thông tin sự kiện tương tự từ **Knowledge Base (KB)**  
- Sinh ra danh sách đầu việc chuẩn theo từng ban phụ trách  
- Trả về kết quả JSON dùng cho **timeline generator** (Model 2) hoặc **optimizer** (Model 3)

---

## 2. Cấu trúc Repository
```
rag_task_demo/
├─ main.py # Entry point FastAPI
├─ .env # Chứa OPENAI_API_KEY và cấu hình model
├─ .gitignore # Bỏ qua .env, cache, venv, logs
│
├─ chroma_db/
|
├─ kb/
│ ├─ global/ # Knowledge Base toàn cục (chuẩn loại sự kiện)
│ │ ├─ career_fair.json
│ │ ├─ workshop_ai.json
│ │ └─ concert_festival.json
│ └─ user/ # (Tùy chọn) dữ liệu riêng từng người dùng
│
├─ models/
|  └─ schemas.py # Quy ước dữ liệu trả về
|
├─ scripts/
|  └─ ingest_global_chroma.py # Chunking tài liệu và đưa vào vector DB
|
├─ services/
│ ├─ pipeline.py # Điều phối pipeline (retriever → LLM → parser)
│ ├─ retriever.py # Xử lý embedding & truy hồi tài liệu KB
│ └─ llm_generator.py # Gọi LLM sinh danh sách task
│
└─ requirements.txt
```


---

## 3. Cách chạy dự án (FastAPI)

### **Bước 1 – Cài thư viện**
```powershell
pip install -r requirements.txt
```
### **Bước 2 – Tạo file** `.env`
```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
```
⚠️ Không commit .env lên GitHub.
Đảm bảo .gitignore đã có dòng:
```gitignore
.env
__pycache__/
venv/
*.log
```
### **Bước 3 – Chạy server FastAPI**
```powershell
python -m uvicorn main:app --reload --port 8000
```
Khi chạy thành công:
- Swagger UI: http://127.0.0.1:8000/docs
## 4. Quy trình hoạt động
### **1️⃣ Nhận đầu vào sự kiện**
```json
{
  "name": "Ngày hội việc làm K2C7",
  "description": "Sự kiện ngoài trời có nhiều doanh nghiệp, gian hàng tuyển dụng và khách mời VIP.",
  "event_type_guess": "Career Fair",
  "outdoor": true,
  "has_sponsor": true,
  "has_vip": true
}
```
### **2️⃣ Truy hồi KB tương đồng**
- Dùng SentenceTransformer để embed mô tả sự kiện.

- Tính độ tương đồng cosine với các file KB trong `kb/global/`.
### **3️⃣ Ghép prompt RAG**
- Lấy thông tin từ các file KB phù hợp → đưa vào ngữ cảnh cho mô hình.
### **4️⃣ Sinh danh sách task qua LLM**
- Mô hình (gpt-4o-mini) hoặc các mô hình local sinh các task lớn theo từng ban.
### **5️⃣ Trả về kết quả JSON**
```
{
  "retrieved_docs": ["career_fair", "workshop_ai"],
  "tasks": [
    {
      "title": "Truyền thông & Media",
      "department": "Media",
      "description": "Thiết kế poster, bài đăng mạng xã hội và livestream sự kiện."
    },
    {
      "title": "Đối ngoại & Nhà tài trợ",
      "department": "Đối ngoại",
      "description": "Liên hệ doanh nghiệp, ký hợp đồng booth và tài trợ."
    },
    ...
  ]
}
```
## 5. Tầm quan trọng của các tài liệu (KB)
Knowledge Base (KB) là phần cốt lõi của hệ thống RAG — đóng vai trò như “trí nhớ dài hạn” của AI.<br>
🔹 Chức năng<br>
- Lưu trữ các mẫu sự kiện chuẩn hóa (Career Fair, Workshop, Festival, v.v.)

- Giúp mô hình hiểu rõ cấu trúc công việc, quy trình tổ chức và ban phụ trách

- Cung cấp ngữ cảnh thật để LLM sinh nội dung chính xác, tránh “ảo tưởng” (hallucination)
<br>
🔹 Cấu trúc 1 tài liệu KB mẫu(hiện tại)

```json
{
  "doc_id": "career_fair",
  "event_type": ["Career Fair", "Ngày hội việc làm"],
  "context_tags": ["outdoor", "sponsor", "vip"],
  "baseline_tasks": [
    {
      "name": "Truyền thông & Media",
      "owner_department": "Media",
      "notes": "Thiết kế poster, bài đăng mạng xã hội, livestream"
    },
    {
      "name": "Đối ngoại & Nhà tài trợ",
      "owner_department": "Đối ngoại",
      "notes": "Liên hệ doanh nghiệp, ký hợp đồng booth, tài trợ"
    }
  ]
}
```

- `doc_id`: mã định danh duy nhất

- `event_type`: danh sách tên gọi / alias của loại sự kiện

- `context_tags`: đặc điểm ngữ cảnh (ngoài trời, có tài trợ, có VIP, học thuật, âm nhạc...)

- `baseline_tasks`: danh sách task mẫu mà hệ thống dùng để truyền vào prompt LLM
🔹 Hướng dẫn đóng góp KB
<br>
Các file trong thư mục kb/global/ là nguồn tri thức cốt lõi của hệ thống.
Việc thay đổi cấu trúc của chúng có thể ảnh hưởng trực tiếp đến pipeline, retriever và quá trình sinh đầu việc (LLM generator).

Những thay đổi an toàn :
| Loại thay đổi                                                               | Ảnh hưởng       | Ghi chú                             |
| --------------------------------------------------------------------------- | --------------- | ----------------------------------- |
| Thêm trường mới ở mức ngoài cùng (`difficulty`, `created_by`, `version`, …) | Không ảnh hưởng | Có thể dùng để quản lý metadata     |
| Thêm tag hoặc alias mới trong `context_tags` / `event_type`                 | Không ảnh hưởng | Giúp retriever tìm chính xác hơn    |
| Bổ sung thêm phần tử vào `baseline_tasks`                                   | Không ảnh hưởng | Mô hình có thêm ví dụ để tham chiếu |
Những thay đổi cần điều chỉnh code :
- Nếu bạn đổi tên các trường trong phần `baseline_tasks`
cần cập nhật lại đoạn format trong `llm_generator.py`.

- Thay đổi cấu trúc phức tạp hơn thì cần thay đổi `retriever.py` để chuyển đổi dữ liệu trước khi nhúng (embedding).
## 📍 Team phát triển:
<h1>://</h1>
