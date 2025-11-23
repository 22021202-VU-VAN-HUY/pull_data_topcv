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


-- Bảng document cho RAG
CREATE TABLE IF NOT EXISTS rag_job_documents (
    id              BIGSERIAL PRIMARY KEY,
    job_id          BIGINT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,

    section_type    VARCHAR(50),           -- ví dụ: 'mo_ta_cong_viec', 'full_summary',...
    content         TEXT NOT NULL,         -- text để RAG
    metadata        JSONB,                 -- {"company_name": "...", "cap_bac": "...", ...}

    embedding       BYTEA,                 -- để sau nếu lưu vector trong Postgres

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rag_job_docs_job_id ON rag_job_documents(job_id);
