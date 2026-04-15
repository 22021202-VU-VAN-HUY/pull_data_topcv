# Job Finder with Chatbot (TopCV Data + RAG)

## 1) Giới thiệu đề tài

**Job Finder with Chatbot** là đồ án xây dựng một nền tảng tìm việc thông minh, trong đó dữ liệu tuyển dụng được thu thập từ TopCV, lưu trữ trên PostgreSQL, sau đó phục vụ cho:
- Tra cứu danh sách công việc theo từ khoá.
- Xem chi tiết việc làm (lương, yêu cầu, quyền lợi, địa điểm, hạn nộp hồ sơ...).
- Quản lý tài khoản ứng viên (đăng ký/đăng nhập, cập nhật hồ sơ, lưu việc làm quan tâm).
- Trò chuyện với chatbot tư vấn tuyển dụng theo ngữ cảnh dữ liệu thật bằng mô hình RAG.
  
Mục tiêu của đề tài là kết hợp **Crawler + CSDL + Web App + AI Chatbot** thành một quy trình thống nhất, có thể mở rộng cho các bài toán gợi ý nghề nghiệp và hỗ trợ ứng tuyển.

---

## 2) Chức năng chính

### 2.1 Thu thập dữ liệu việc làm tự động
- Đọc sitemap TopCV để lấy danh sách URL job.
- Crawl nội dung theo lô (batch), retry khi lỗi.
- Fallback sang crawler headless browser (Playwright) cho trang khó parse.
- Upsert dữ liệu công ty, công việc, địa điểm và các section mô tả vào DB.

### 2.2 Website tìm việc
- Trang chủ hiển thị danh sách việc làm, hỗ trợ tìm kiếm theo từ khoá.
- Sắp xếp ưu tiên các job còn hạn nộp hồ sơ.
- Trang chi tiết công việc hiển thị đầy đủ thông tin tuyển dụng.
- Hiển thị lương đã chuẩn hoá/định dạng từ dữ liệu thô.

### 2.3 Tài khoản người dùng
- Đăng ký, đăng nhập, đăng xuất.
- Quản lý thông tin cá nhân.
- Lưu/bỏ lưu công việc yêu thích (bookmark).
- Xem lại danh sách công việc đã lưu tại trang hồ sơ.

### 2.4 Chatbot tư vấn tuyển dụng (RAG)
- Endpoint chat nhận câu hỏi + lịch sử hội thoại.
- Query parser bóc tách ý định người dùng (`search_jobs`, `ask_detail`, `compare_jobs`, ...).
- Truy xuất ngữ cảnh từ bảng vector (`rag_job_documents`) bằng embedding.
- Kết hợp Gemini để sinh câu trả lời tiếng Việt, giới hạn theo ngữ cảnh dữ liệu việc làm.

### 2.5 Hỗ trợ đánh giá chatbot
- Có bộ câu hỏi kiểm thử và script đánh giá trong thư mục `testing_chatbot/`.
- Phù hợp để benchmark chất lượng chatbot theo từng phiên bản prompt/data.

---

## 3) Công nghệ sử dụng

### Backend & API
- **Python 3.12**
- **Flask** (web server, routing, template rendering)
- **Pydantic** (hỗ trợ validation/schema ở một số thành phần)

### Data & Database
- **PostgreSQL 16**
- **pgvector** (lưu embedding và truy vấn vector)
- **psycopg2** (kết nối DB)

### Crawler
- **requests + BeautifulSoup + lxml** (crawl/parse HTML)
- **Playwright** (headless browser fallback)

### AI / NLP
- **sentence-transformers** (tạo embedding văn bản)
- **Google Gemini API** (`google-generativeai`) cho parser + chat generation

### Front-end
- **HTML/CSS/JavaScript** thuần (Jinja templates)

### Môi trường chạy
- **Docker / Docker Compose** (dịch vụ DB)
- **python-dotenv** (quản lý biến môi trường)



```
# URL mặc định nếu không truyền tham số
TOPCV_BROWSER_JOB_URL="https://www.topcv.vn/viec-lam/your-job.html"

# Tùy chọn Playwright
PLAYWRIGHT_HEADLESS=true
# Tăng nếu hay bị timeout (gợi ý: 45000)
PLAYWRIGHT_NAV_TIMEOUT_MS=20000
PLAYWRIGHT_EXTRA_WAIT_MS=1000
TOPCV_BROWSER_WAIT_SELECTOR="body"
```

2. Chạy crawler headless cho một job:

```
python -m app.topcv.crawl_browser --url "https://www.topcv.vn/viec-lam/your-job.html"
```

Nếu bỏ `--url`, script sẽ dùng giá trị từ `TOPCV_BROWSER_JOB_URL`.
