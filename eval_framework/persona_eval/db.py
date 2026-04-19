from __future__ import annotations
import os
import psycopg2
from psycopg2.extensions import connection


_DATABASE_URL = os.getenv("DATABASE_URL", "")

# NOTE: run_id was changed from UUID to TEXT. If upgrading an existing database,
# run: ALTER TABLE eval_results ALTER COLUMN run_id TYPE TEXT USING run_id::TEXT,
#      ALTER COLUMN run_id SET DEFAULT '';
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS eval_results (
    id          SERIAL PRIMARY KEY,
    run_id      TEXT NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    suite       TEXT NOT NULL,
    model       TEXT NOT NULL,
    persona_id  TEXT NOT NULL,
    dimension_id TEXT NOT NULL,
    dimension_name TEXT NOT NULL,
    passed      BOOLEAN NOT NULL,
    score       REAL NOT NULL,
    details     JSONB,
    errors      JSONB
);
CREATE INDEX IF NOT EXISTS idx_eval_results_run_id ON eval_results (run_id);
CREATE INDEX IF NOT EXISTS idx_eval_results_run_dimension ON eval_results (run_id, dimension_id);
CREATE INDEX IF NOT EXISTS idx_eval_results_persona_id ON eval_results (persona_id);
CREATE INDEX IF NOT EXISTS idx_eval_results_dimension_id ON eval_results (dimension_id);
"""


def get_connection() -> connection:
    if not _DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg2.connect(_DATABASE_URL)


def ensure_schema(conn: connection) -> None:
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
    conn.commit()
