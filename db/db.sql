-- table companies
CREATE TABLE IF NOT EXISTS companies (
    id          BIGSERIAL PRIMARY KEY,

    name        TEXT NOT NULL,   
    url         TEXT,            
    logo        TEXT,            
    size        TEXT,            --  (vd: "100-499 nhân viên")
    industry    TEXT,            -- vd: khác
    address     TEXT,            

    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_companies_url ON companies(url);


-- table jobs
CREATE TABLE IF NOT EXISTS jobs (
    id                      BIGSERIAL PRIMARY KEY,
    company_id              BIGINT REFERENCES companies(id),  -- FK -> companies.id
    url                     TEXT NOT NULL,
    title                   TEXT NOT NULL,

    salary_min              NUMERIC(18,0),
    salary_max              NUMERIC(18,0),
    salary_currency         VARCHAR(10),
    salary_interval         VARCHAR(20),
    salary_raw_text         TEXT,
    experience_months       INT,
    experience_raw_text     TEXT,
    deadline                TIMESTAMPTZ,
    cap_bac                 TEXT,
    hoc_van                 TEXT,
    so_luong_tuyen          INT,
    hinh_thuc_lam_viec      TEXT,
    hinh_thuc_lam_viec_raw  TEXT,
    so_luong_tuyen_raw      TEXT,

    crawled_at              TIMESTAMPTZ NOT NULL,
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_jobs_url ON jobs(url);
CREATE INDEX IF NOT EXISTS idx_jobs_company_id ON jobs(company_id);
CREATE INDEX IF NOT EXISTS idx_jobs_deadline ON jobs(deadline);
CREATE INDEX IF NOT EXISTS idx_jobs_crawled_at ON jobs(crawled_at);


-- table job_locations
CREATE TABLE IF NOT EXISTS job_locations (
    id              BIGSERIAL PRIMARY KEY,
    job_id          BIGINT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,

    location_text   TEXT NOT NULL,

    is_primary      BOOLEAN NOT NULL DEFAULT FALSE,
    sort_order      INT NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_job_locations_job_id ON job_locations(job_id);


-- table job_sections
CREATE TABLE IF NOT EXISTS job_sections (
    id              BIGSERIAL PRIMARY KEY,
    job_id          BIGINT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,

    section_type    VARCHAR(50) NOT NULL,  -- 'mo_ta_cong_viec', 'yeu_cau_ung_vien',...
    text_content    TEXT,
    html_content    TEXT,

    crawled_at      TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_job_sections_job_id ON job_sections(job_id);
CREATE INDEX IF NOT EXISTS idx_job_sections_type ON job_sections(section_type);

CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS rag_job_documents;

CREATE TABLE IF NOT EXISTS rag_job_documents (
    id              BIGSERIAL PRIMARY KEY,

    job_id          BIGINT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,

    doc_type        VARCHAR(30) NOT NULL,   -- 'job_full', 'job_section', ...
    section_type    VARCHAR(50),            -- 'mo_ta_cong_viec', 'yeu_cau_ung_vien', ...

    chunk_index     INT NOT NULL DEFAULT 0, -- thứ tự chunk trong cùng 1 doc (0,1,2,...)

    content         TEXT NOT NULL,          -- text đã gộp đầy đủ thông tin cho RAG (1 chunk)
    metadata        JSONB,                  -- snapshot đủ thông tin (job_title, company_name, locations, salary_text, deadline,...)

    -- vector embedding (dimension tuỳ model, ví dụ 768 / 1024 / 1536 / 3072)
    embedding_vec   vector(1536),

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Một job không nên có 2 doc trùng hệt loại + section + chunk_index
CREATE UNIQUE INDEX IF NOT EXISTS uq_rag_job_docs_job_doc_chunk
    ON rag_job_documents (job_id, doc_type, section_type, chunk_index);

-- Truy vấn tất cả chunk của 1 job
CREATE INDEX IF NOT EXISTS idx_rag_job_docs_job_id
    ON rag_job_documents (job_id);

-- Lọc theo loại doc (full vs section)
CREATE INDEX IF NOT EXISTS idx_rag_job_docs_doc_type
    ON rag_job_documents (doc_type);

-- Lọc theo section (mô tả, yêu cầu, quyền lợi,...)
CREATE INDEX IF NOT EXISTS idx_rag_job_docs_section_type
    ON rag_job_documents (section_type);

-- Full-text search fallback cho content (nếu dùng PostgreSQL FTS)
CREATE INDEX IF NOT EXISTS idx_rag_job_docs_content_fts
    ON rag_job_documents
    USING GIN (to_tsvector('simple', content));

-- Index cho metadata JSONB (lọc theo industry, location,... nếu cần)
CREATE INDEX IF NOT EXISTS idx_rag_job_docs_metadata_gin
    ON rag_job_documents
    USING GIN (metadata);

-- Index vector cho nearest-neighbor search (cosine similarity)
CREATE INDEX IF NOT EXISTS idx_rag_job_docs_embedding_vec
    ON rag_job_documents
    USING ivfflat (embedding_vec vector_cosine_ops)
    WITH (lists = 100);

-- table user
CREATE TABLE IF NOT EXISTS users (
    id             BIGSERIAL PRIMARY KEY,
    full_name      TEXT        NOT NULL,                 
    email          TEXT        NOT NULL UNIQUE,          
    phone          TEXT,                                 
    password_hash  TEXT        NOT NULL,                 
    is_active      BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_email
    ON users (email);

-- user_job_bookmarks: người dùng đánh dấu công việc quan trọng
CREATE TABLE IF NOT EXISTS user_job_bookmarks (
    user_id    BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    job_id     BIGINT NOT NULL REFERENCES jobs(id)  ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, job_id)
);

CREATE INDEX IF NOT EXISTS idx_user_job_bookmarks_user_id
    ON user_job_bookmarks (user_id);

CREATE INDEX IF NOT EXISTS idx_user_job_bookmarks_job_id
    ON user_job_bookmarks (job_id);

-- chat_sessions: 1 phiên chat của người dùng
CREATE TABLE IF NOT EXISTS chat_sessions (
    id               BIGSERIAL PRIMARY KEY,
    user_id          BIGINT REFERENCES users(id) ON DELETE SET NULL, -- NULL nếu khách vãng lai
    session_token    TEXT UNIQUE,          -- mã session lưu ở client (cookie / sessionStorage)
    is_active        BOOLEAN     NOT NULL DEFAULT TRUE,
    started_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_activity_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    metadata         JSONB                           -- ví dụ: {"browser": "...", "entry_job_id": 123}
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id
    ON chat_sessions (user_id);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_last_activity
    ON chat_sessions (last_activity_at);

-- chat_messages: các tin nhắn trong 1 phiên chat
CREATE TABLE IF NOT EXISTS chat_messages (
    id             BIGSERIAL PRIMARY KEY,
    session_id     BIGINT      NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role           VARCHAR(20) NOT NULL,     -- 'user' | 'assistant' | 'system'
    content        TEXT        NOT NULL,     -- nội dung tin nhắn
    related_job_id BIGINT REFERENCES jobs(id) ON DELETE SET NULL,   -- optional: job liên quan
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id_created_at
    ON chat_messages (session_id, created_at);