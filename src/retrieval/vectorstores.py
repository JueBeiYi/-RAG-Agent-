from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sqlite3
import threading
from pathlib import Path

import numpy as np

from src.core.schemas import Chunk


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def scoped_id(scope: str, chunk_id: str) -> str:
    return hashlib.sha256(scope.encode()).hexdigest()[:24] + ":" + chunk_id


class SQLiteVectors:
    """Exact cosine search for the explicitly labelled lightweight demo backend.

    It also acts as FAISS's transactional vector/metadata journal. Model
    fingerprints are part of every key, so changing an embedding never silently
    reuses incompatible vectors. No user text is needed in this database.
    """

    def __init__(self, directory: Path, fingerprint: str):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "vectors.sqlite3"
        self.fingerprint = fingerprint
        self.lock = threading.RLock()
        self.revision = 0
        with self.connect() as con:
            con.execute("CREATE TABLE IF NOT EXISTS vectors (model TEXT NOT NULL, scope TEXT NOT NULL, id TEXT NOT NULL, document_id TEXT NOT NULL, hash TEXT NOT NULL, dim INTEGER NOT NULL, vector BLOB NOT NULL, PRIMARY KEY(model,scope,id))")
            con.execute("CREATE INDEX IF NOT EXISTS vector_document ON vectors(model,scope,document_id)")

    def connect(self):
        con = sqlite3.connect(self.path, timeout=30)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def existing(self, chunks: list[Chunk]) -> dict[tuple[str, str], str]:
        result = {}
        for scope in {c.scope for c in chunks}:
            with self.connect() as con:
                rows = con.execute("SELECT id,hash FROM vectors WHERE model=? AND scope=?", (self.fingerprint, scope)).fetchall()
            result.update({(scope, identifier): digest for identifier, digest in rows})
        return result

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray):
        if not chunks:
            return
        if vectors.ndim != 2 or len(vectors) != len(chunks) or not np.isfinite(vectors).all():
            raise ValueError("向量维度、数量或有限性校验失败")
        with self.lock, self.connect() as con:
            con.executemany("INSERT INTO vectors VALUES(?,?,?,?,?,?,?) ON CONFLICT(model,scope,id) DO UPDATE SET document_id=excluded.document_id,hash=excluded.hash,dim=excluded.dim,vector=excluded.vector",
                            [(self.fingerprint, c.scope, c.chunk_id, c.document_id, text_hash(c.text),
                              vectors.shape[1], np.asarray(v, dtype="<f4").tobytes()) for c, v in zip(chunks, vectors)])
            self.revision += 1

    def retain(self, scope: str, document_id: str, ids: list[str]):
        keep = set(ids)
        with self.lock, self.connect() as con:
            rows = con.execute("SELECT id FROM vectors WHERE model=? AND scope=? AND document_id=?",
                               (self.fingerprint, scope, document_id)).fetchall()
            con.executemany("DELETE FROM vectors WHERE model=? AND scope=? AND id=?",
                            [(self.fingerprint, scope, r[0]) for r in rows if r[0] not in keep])
            self.revision += 1

    def delete(self, document_id: str, scope: str):
        with self.lock, self.connect() as con:
            con.execute("DELETE FROM vectors WHERE model=? AND scope=? AND document_id=?", (self.fingerprint, scope, document_id))
            self.revision += 1

    def rows(self, scope: str) -> list[tuple[str, str, int, bytes]]:
        with self.connect() as con:
            return con.execute("SELECT id,document_id,dim,vector FROM vectors WHERE model=? AND scope=? ORDER BY id",
                               (self.fingerprint, scope)).fetchall()

    def query(self, query: np.ndarray, scope: str, allowed: set[str], k: int) -> list[tuple[str, float]]:
        rows = [row for row in self.rows(scope) if row[0] in allowed]
        if not rows:
            return []
        if any(row[2] != len(query) for row in rows):
            raise RuntimeError("当前模型与持久化索引维度不一致，需要重新索引")
        vectors = np.stack([np.frombuffer(row[3], dtype="<f4") for row in rows])
        scores = vectors @ query
        ranks = sorted(range(len(rows)), key=lambda i: (-float(scores[i]), rows[i][0]))[:k]
        return [(rows[i][0], float(scores[i])) for i in ranks]

    def health(self):
        with self.connect() as con:
            count = con.execute("SELECT count(*) FROM vectors WHERE model=?", (self.fingerprint,)).fetchone()[0]
        return {"backend": "sqlite", "ready": True, "demo": True, "vectors": count,
                "search": "exact normalized-vector dot product", "fingerprint": self.fingerprint}


class ChromaVectors:
    def __init__(self, directory: Path, fingerprint: str):
        self.directory = directory
        self.fingerprint = fingerprint
        self.client = None
        self.collection = None

    def _load(self):
        if self.collection is None:
            try:
                import chromadb
                from chromadb.config import Settings as ChromaSettings
            except ImportError as exc:
                raise RuntimeError("Chroma 需要安装 requirements-models.txt；demo 可使用 sqlite 向量后端") from exc
            self.directory.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(self.directory),
                                                    settings=ChromaSettings(anonymized_telemetry=False))
            self.collection = self.client.get_or_create_collection("paper_" + self.fingerprint,
                                                                   metadata={"hnsw:space": "cosine"},
                                                                   embedding_function=None)
        return self.collection

    def existing(self, chunks: list[Chunk]) -> dict[tuple[str, str], str]:
        if not chunks:
            return {}
        collection = self._load()
        result = {}
        # Avoid the SQLite bind-parameter limit for large batches.
        for offset in range(0, len(chunks), 500):
            rows = collection.get(ids=[scoped_id(c.scope, c.chunk_id) for c in chunks[offset:offset + 500]],
                                  include=["metadatas"])
            for meta in rows["metadatas"]:
                result[(meta["scope"], meta["chunk_id"])] = meta.get("text_hash", "")
        return result

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray):
        if not chunks:
            return
        collection = self._load()
        batch = min(500, self.client.get_max_batch_size()) if hasattr(self.client, "get_max_batch_size") else 500
        for offset in range(0, len(chunks), batch):
            local = chunks[offset:offset + batch]
            collection.upsert(ids=[scoped_id(c.scope, c.chunk_id) for c in local],
                              embeddings=vectors[offset:offset + batch].tolist(),
                              metadatas=[{"scope": c.scope, "document_id": c.document_id, "chunk_id": c.chunk_id,
                                          "text_hash": text_hash(c.text)} for c in local])

    def retain(self, scope: str, document_id: str, ids: list[str]):
        collection = self._load()
        rows = collection.get(where={"$and": [{"scope": scope}, {"document_id": document_id}]}, include=[])
        keep = {scoped_id(scope, identifier) for identifier in ids}
        remove = [identifier for identifier in rows["ids"] if identifier not in keep]
        for offset in range(0, len(remove), 500):
            collection.delete(ids=remove[offset:offset + 500])

    def delete(self, document_id: str, scope: str):
        self._load().delete(where={"$and": [{"scope": scope}, {"document_id": document_id}]})

    def query(self, query: np.ndarray, scope: str, allowed: set[str], k: int) -> list[tuple[str, float]]:
        if not allowed:
            return []
        collection = self._load()
        if collection.count() == 0:
            return []
        # Pre-filter exact allowed chunks via metadata. Each chunk is verified
        # against scoped SQLite records before it can leave Retriever.search.
        ids = sorted(allowed)
        candidates = []
        for offset in range(0, len(ids), 500):
            local = ids[offset:offset + 500]
            rows = collection.query(query_embeddings=[query.tolist()], n_results=min(k, len(local)),
                                    where={"$and": [{"scope": scope}, {"chunk_id": {"$in": local}}]},
                                    include=["distances", "metadatas"])
            for meta, distance in zip(rows["metadatas"][0], rows["distances"][0]):
                if meta["scope"] == scope and meta["chunk_id"] in allowed:
                    candidates.append((meta["chunk_id"], 1.0 - float(distance)))
        return sorted(candidates, key=lambda item: (-item[1], item[0]))[:k]

    def health(self):
        available = importlib.util.find_spec("chromadb") is not None
        info = {"backend": "chroma", "ready": False, "available": available, "loaded": self.collection is not None,
                "demo": False, "fingerprint": self.fingerprint, "persistent_path": str(self.directory)}
        if available:
            try:
                info.update(ready=True, vectors=self._load().count())
            except Exception as exc:
                info["error"] = str(exc)
        return info


class FaissVectors(SQLiteVectors):
    """FAISS FlatIP, persisted per scope; SQLite is the vector update journal.

    FAISS's flat index does not directly support mutable metadata. A small
    course corpus can safely rebuild its native index after an incremental
    update, while embeddings for unchanged chunks are reused from SQLite.
    """

    def __init__(self, directory: Path, fingerprint: str):
        super().__init__(directory, fingerprint)
        self.directory = directory / "faiss"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, tuple[str, list[str], object]] = {}

    def _index(self, scope: str):
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("FAISS 后端需要 faiss-cpu；安装 requirements-models.txt") from exc
        stem = self.fingerprint + "_" + hashlib.sha256(scope.encode()).hexdigest()[:20]
        index_path, manifest_path = self.directory / (stem + ".faiss"), self.directory / (stem + ".json")
        rows = self.rows(scope)
        if not rows:
            index_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
            self.cache.pop(scope, None)
            return None, []
        signature = hashlib.sha256(b"".join(r[0].encode() + r[3] for r in rows)).hexdigest()
        cached = self.cache.get(scope)
        if cached and cached[0] == signature:
            return cached[2], cached[1]
        dimensions = {r[2] for r in rows}
        if len(dimensions) != 1:
            raise RuntimeError("FAISS 索引含多种向量维度，需要重新索引")
        ids = [r[0] for r in rows]
        index = None
        if index_path.exists() and manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("signature") == signature and manifest.get("ids") == ids:
                    # pathlib handles non-ASCII Windows paths; FAISS native
                    # fopen does not on every wheel/version.
                    index = faiss.deserialize_index(np.frombuffer(index_path.read_bytes(), dtype=np.uint8).copy())
            except (ValueError, OSError, RuntimeError):
                index = None
        if index is None:
            index = faiss.IndexFlatIP(rows[0][2])
            index.add(np.stack([np.frombuffer(r[3], dtype="<f4") for r in rows]).astype(np.float32))
            temporary = index_path.with_suffix(".faiss.tmp")
            temporary.write_bytes(faiss.serialize_index(index).tobytes())
            os.replace(temporary, index_path)
            temporary_manifest = manifest_path.with_suffix(".json.tmp")
            temporary_manifest.write_text(json.dumps({"signature": signature, "ids": ids}), encoding="utf-8")
            os.replace(temporary_manifest, manifest_path)
        self.cache[scope] = (signature, ids, index)
        return index, ids

    def retain(self, scope: str, document_id: str, ids: list[str]):
        super().retain(scope, document_id, ids)
        self.cache.pop(scope, None)
        self._index(scope)

    def delete(self, document_id: str, scope: str):
        super().delete(document_id, scope)
        self.cache.pop(scope, None)
        self._index(scope)

    def query(self, query: np.ndarray, scope: str, allowed: set[str], k: int):
        index, ids = self._index(scope)
        if index is None:
            return []
        if index.d != len(query):
            raise RuntimeError("FAISS 模型向量维度不一致，需要重新索引")
        scores, ranks = index.search(np.asarray([query], dtype=np.float32), len(ids))
        found = [(ids[int(rank)], float(score)) for rank, score in zip(ranks[0], scores[0])
                 if rank >= 0 and ids[int(rank)] in allowed]
        return sorted(found, key=lambda item: (-item[1], item[0]))[:k]

    def health(self):
        info = super().health()
        available = importlib.util.find_spec("faiss") is not None
        info.update(backend="faiss", ready=available, available=available, demo=False,
                    search="FAISS exact FlatIP with scoped metadata journal")
        return info
