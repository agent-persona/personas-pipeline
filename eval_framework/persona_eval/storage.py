from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Protocol
from persona_eval.schemas import EvalResult


class ResultRecorder(Protocol):
    def record(self, result: EvalResult) -> None: ...
    def record_batch(self, results: list[EvalResult]) -> None: ...


class JsonRecorder:
    """Fallback recorder for local dev — appends to a JSONL file."""

    def __init__(self, path: str | Path = "eval_results.jsonl") -> None:
        self.path = Path(path)

    def record(self, result: EvalResult) -> None:
        with self.path.open("a") as f:
            f.write(result.model_dump_json() + "\n")

    def record_batch(self, results: list[EvalResult]) -> None:
        for r in results:
            self.record(r)

    def load_all(self) -> list[EvalResult]:
        if not self.path.exists():
            return []
        results = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(EvalResult.model_validate_json(line))
        return results


class PostgresRecorder:
    def __init__(self) -> None:
        from persona_eval.db import get_connection, ensure_schema
        self._conn = get_connection()
        ensure_schema(self._conn)

    def _ensure_connection(self) -> None:
        """Reconnect if the database connection is broken."""
        try:
            if self._conn.closed:
                raise Exception("connection closed")
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1")
        except Exception:
            from persona_eval.db import get_connection, ensure_schema
            self._conn = get_connection()
            ensure_schema(self._conn)

    @staticmethod
    def _result_to_row(result: EvalResult) -> tuple:
        return (
            result.suite,
            result.model,
            result.persona_id,
            result.dimension_id,
            result.dimension_name,
            result.passed,
            result.score,
            json.dumps(result.details),
            json.dumps(result.errors),
            result.run_id,
        )

    def record(self, result: EvalResult) -> None:
        self._ensure_connection()
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO eval_results
                  (suite, model, persona_id, dimension_id, dimension_name, passed, score, details, errors, run_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                self._result_to_row(result),
            )
        self._conn.commit()

    def record_batch(self, results: list[EvalResult]) -> None:
        if not results:
            return
        self._ensure_connection()
        from psycopg2.extras import execute_values
        sql = """
            INSERT INTO eval_results
              (suite, model, persona_id, dimension_id, dimension_name, passed, score, details, errors, run_id)
            VALUES %s
        """
        rows = [self._result_to_row(r) for r in results]
        try:
            with self._conn.cursor() as cur:
                execute_values(cur, sql, rows)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise


def get_recorder() -> JsonRecorder | PostgresRecorder:
    if os.getenv("DATABASE_URL"):
        return PostgresRecorder()
    return JsonRecorder()
