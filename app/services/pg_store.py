"""Lưu kho embedding vào PostgreSQL (một dòng JSONB) để không mất dữ liệu khi Render restart.

Dùng khi có biến môi trường DATABASE_URL. Cấu trúc JSON giống hệt file enrollments.json.
Mỗi thao tác mở một kết nối ngắn (Neon free tự ngủ nên không giữ pool).
"""
from __future__ import annotations

from typing import Any

_DDL = """
CREATE TABLE IF NOT EXISTS face_store (
    id         INTEGER PRIMARY KEY,
    data       JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def _connect(url: str):
    import psycopg  # import muộn để test/chạy bằng file JSON không cần psycopg

    return psycopg.connect(url, connect_timeout=20)


def load(url: str) -> dict[str, Any] | None:
    with _connect(url) as conn:
        conn.execute(_DDL)
        row = conn.execute("SELECT data FROM face_store WHERE id = 1").fetchone()
    return None if row is None else row[0]


def save(url: str, data: dict[str, Any]) -> None:
    from psycopg.types.json import Jsonb

    with _connect(url) as conn:
        conn.execute(_DDL)
        conn.execute(
            "INSERT INTO face_store (id, data) VALUES (1, %s) "
            "ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data, updated_at = now()",
            (Jsonb(data),),
        )
