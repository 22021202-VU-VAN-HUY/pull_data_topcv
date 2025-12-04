import json
from typing import List
from psycopg2.extras import RealDictCursor
from app.topcv.export_job_json import (
    get_connection,
    fetch_job_row,
    fetch_locations,
    fetch_sections,
    build_job_json,
)

def fetch_active_indexed_job_ids(conn, limit: int = 2000) -> List[int]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT j.id
            FROM jobs j
            JOIN rag_job_documents d
              ON d.job_id = j.id
             AND d.doc_type = 'job_overview'
             AND d.chunk_index = 0
            WHERE d.metadata->>'is_active' = 'true'
            ORDER BY j.crawled_at DESC NULLS LAST, j.id DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall() or []
    return [row["id"] for row in rows]


def export_jobs(limit: int = 200, output_file: str = "jobs_for_chatgpt.jsonl") -> None:
    conn = get_connection()
    all_jobs = []
    try:
        job_ids = fetch_active_indexed_job_ids(conn, limit=limit)
        print(f"Found {len(job_ids)} active & indexed job ids to export.")

        for idx, job_id in enumerate(job_ids, start=1):
            row = fetch_job_row(conn, job_id=job_id)
            if not row:
                print(f"[WARN] Job id {job_id} not found in jobs table, skip.")
                continue

            locations = fetch_locations(conn, job_id)
            sections = fetch_sections(conn, job_id)
            job_json = build_job_json(row, locations, sections)

            all_jobs.append(job_json)

            if idx % 20 == 0:
                print(f"Processed {idx}/{len(job_ids)} jobs...")

        # Ghi ra JSON 
        with open(output_file, "w", encoding="utf-8") as f:
            for job in all_jobs:
                f.write(json.dumps(job, ensure_ascii=False) + "\n")

        print(f"Done. Exported {len(all_jobs)} jobs -> {output_file}")
    finally:
        conn.close()


if __name__ == "__main__":
    export_jobs(limit=2000, output_file="create_question/jobs_for_chat.jsonl")
