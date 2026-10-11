from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src.core.schemas import Chunk, Document


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Storage:
    """Scoped SQLite records. A connection per operation permits concurrent readers."""

    TABLES = {"documents", "jobs", "sessions", "runs", "cache", "experiments", "versions"}

    def __init__(self, directory: str | Path):
        self.path = Path(directory).resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.path / "state.sqlite3"
        self.lock = threading.RLock()
        with self.connect() as con:
            for table in self.TABLES:
                con.execute(f"CREATE TABLE IF NOT EXISTS {table} (id TEXT, scope TEXT, payload TEXT NOT NULL, updated TEXT NOT NULL, PRIMARY KEY(id,scope))")
            con.execute("CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, scope TEXT, document_id TEXT, payload TEXT NOT NULL)")
            con.execute("CREATE INDEX IF NOT EXISTS chunk_scope ON chunks(scope,document_id)")

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.db_path, timeout=30)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def put(self, table: str, key: str, scope: str, data: dict):
        if table not in self.TABLES:
            raise ValueError("Invalid table")
        with self.lock, self.connect() as con:
            con.execute(f"INSERT INTO {table} VALUES (?,?,?,?) ON CONFLICT(id,scope) DO UPDATE SET payload=excluded.payload,updated=excluded.updated",
                        (key, scope, json.dumps(data, ensure_ascii=False), now()))

    def get(self, table: str, key: str, scope: str) -> dict | None:
        if table not in self.TABLES:
            raise ValueError("Invalid table")
        with self.connect() as con:
            row = con.execute(f"SELECT payload FROM {table} WHERE id=? AND scope=?", (key, scope)).fetchone()
        return json.loads(row[0]) if row else None

    def list(self, table: str, scope: str, limit: int = 1000) -> list[dict]:
        if table not in self.TABLES:
            raise ValueError("Invalid table")
        with self.connect() as con:
            rows = con.execute(f"SELECT payload FROM {table} WHERE scope=? ORDER BY updated DESC LIMIT ?", (scope, limit)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def remove(self, table: str, key: str, scope: str):
        if table not in self.TABLES:
            raise ValueError("Invalid table")
        with self.lock, self.connect() as con:
            con.execute(f"DELETE FROM {table} WHERE id=? AND scope=?", (key, scope))

    def put_document(self, doc: Document):
        self.put("documents", doc.document_id, doc.scope, doc.model_dump())

    def get_document(self, document_id: str, scope: str) -> Document | None:
        data = self.get("documents", document_id, scope)
        return Document.model_validate(data) if data else None

    def list_documents(self, scope: str) -> list[Document]:
        return [Document.model_validate(d) for d in self.list("documents", scope)]

    def replace_chunks(self, document_id: str, scope: str, chunks: list[Chunk]):
        if any(c.scope != scope or c.document_id != document_id for c in chunks):
            raise ValueError("Chunk scope mismatch")
        with self.lock, self.connect() as con:
            con.execute("DELETE FROM chunks WHERE document_id=? AND scope=?", (document_id, scope))
            con.executemany("INSERT INTO chunks VALUES(?,?,?,?)", [(c.chunk_id, scope, document_id, c.model_dump_json()) for c in chunks])

    def commit_document(self, document: Document, chunks: list[Chunk]) -> str:
        """Publish ready metadata, chunks, index version and invalidation atomically."""
        scope, document_id = document.scope, document.document_id
        if any(c.scope != scope or c.document_id != document_id for c in chunks):
            raise ValueError("Chunk scope mismatch")
        version = uuid.uuid4().hex
        for chunk in chunks:
            chunk.index_version = version
        document.status, document.chunks, document.error = "ready", len(chunks), ""
        with self.lock, self.connect() as con:
            con.execute("DELETE FROM chunks WHERE document_id=? AND scope=?", (document_id, scope))
            con.executemany("INSERT INTO chunks VALUES(?,?,?,?)", [(c.chunk_id, scope, document_id, c.model_dump_json()) for c in chunks])
            con.execute("INSERT INTO documents VALUES(?,?,?,?) ON CONFLICT(id,scope) DO UPDATE SET payload=excluded.payload,updated=excluded.updated",
                        (document_id, scope, document.model_dump_json(), now()))
            con.execute("INSERT INTO versions VALUES(?,?,?,?) ON CONFLICT(id,scope) DO UPDATE SET payload=excluded.payload,updated=excluded.updated",
                        ("index", scope, json.dumps({"version": version}), now()))
            con.execute("DELETE FROM cache WHERE scope=?", (scope,))
        return version

    def list_chunks(self, scope: str, document_ids: list[str] | None = None) -> list[Chunk]:
        params: list = [scope]
        sql = "SELECT payload FROM chunks WHERE scope=?"
        if document_ids:
            sql += " AND document_id IN (" + ",".join("?" for _ in document_ids) + ")"
            params += document_ids
        with self.connect() as con:
            rows = con.execute(sql + " ORDER BY id", params).fetchall()
        return [Chunk.model_validate_json(r[0]) for r in rows]

    def index_version(self, scope: str) -> str:
        return (self.get("versions", "index", scope) or {}).get("version", "empty")

    def bump_version(self, scope: str) -> str:
        version = uuid.uuid4().hex
        self.put("versions", "index", scope, {"version": version})
        self.clear_cache(scope)
        return version

    def delete_document(self, document_id: str, scope: str):
        with self.lock, self.connect() as con:
            con.execute("DELETE FROM chunks WHERE document_id=? AND scope=?", (document_id, scope))
            con.execute("DELETE FROM documents WHERE id=? AND scope=?", (document_id, scope))
            con.execute("INSERT INTO versions VALUES(?,?,?,?) ON CONFLICT(id,scope) DO UPDATE SET payload=excluded.payload,updated=excluded.updated",
                        ("index", scope, json.dumps({"version": uuid.uuid4().hex}), now()))
            con.execute("DELETE FROM cache WHERE scope=?", (scope,))

    def clear_cache(self, scope: str):
        with self.lock, self.connect() as con:
            con.execute("DELETE FROM cache WHERE scope=?", (scope,))

    def create_session(self, scope: str, title: str = "新对话") -> dict:
        session = {"id": uuid.uuid4().hex, "title": title[:80], "created_at": now(),
                   "messages": [], "memory": {}, "document_ids": [],
                   "title_custom": title not in {"新对话", "新的科研对话"}}
        self.put("sessions", session["id"], scope, session)
        return session

    def list_messages(self, session_id: str, scope: str) -> list[dict]:
        session = self.get("sessions", session_id, scope)
        return session.get("messages", []) if session else []

    def add_message(self, session_id: str, scope: str, role: str, content: str, **metadata):
        with self.lock:
            session = self.get("sessions", session_id, scope)
            if not session:
                raise KeyError("会话不存在")
            session["messages"].append({"role": role, "content": content, "created_at": now(), **metadata})
            self.put("sessions", session_id, scope, session)

    def get_memory(self, session_id: str, scope: str) -> dict:
        return (self.get("sessions", session_id, scope) or {}).get("memory", {})

    def set_memory(self, session_id: str, scope: str, memory: dict):
        with self.lock:
            session = self.get("sessions", session_id, scope)
            if session:
                session["memory"] = memory
                self.put("sessions", session_id, scope, session)

    def health(self) -> dict:
        with self.connect() as con:
            con.execute("SELECT 1").fetchone()
        return {"ready": True, "backend": "sqlite"}
