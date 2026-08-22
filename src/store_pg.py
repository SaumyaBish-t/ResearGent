from __future__ import annotations

import json
from typing import Any
from src.db import connection
from src.auth.context import get_current_user_id


def _compile_where(where: dict | None, params: list) -> str:
    """
    Compile a ChromaDB-shaped where clause into safe PostgreSQL JSONB extraction SQL.
    Supports equality, $in, $eq, and $ne.
    """
    if not where:
        return ""
    sql_parts = []
    for k, expected in where.items():
        # Validate key is alphanumeric/underscore to prevent injection on JSON path fields
        if not k.isalnum() and "_" not in k:
            raise ValueError(f"Invalid metadata key: {k}")

        if isinstance(expected, dict):
            if "$in" in expected:
                in_list = expected["$in"]
                if not in_list:
                    sql_parts.append("FALSE")
                else:
                    placeholders = ", ".join(["%s"] * len(in_list))
                    sql_parts.append(f"metadata->>'{k}' IN ({placeholders})")
                    params.extend(in_list)
            elif "$eq" in expected:
                sql_parts.append(f"metadata->>'{k}' = %s")
                params.append(expected["$eq"])
            elif "$ne" in expected:
                sql_parts.append(f"metadata->>'{k}' != %s")
                params.append(expected["$ne"])
            else:
                raise ValueError(f"Unsupported where operator: {expected}")
        else:
            sql_parts.append(f"metadata->>'{k}' = %s")
            params.append(expected)
    return " AND " + " AND ".join(sql_parts) if sql_parts else ""


class PgCollection:
    """
    PostgreSQL-backed vector collection implementing the Chroma-shaped API.
    All operations are multi-tenant and scoped to the active current_user_id.
    """

    def __init__(self, name: str):
        self.name = name

    def count(self) -> int:
        user_id = get_current_user_id()
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM chunks WHERE user_id = %s::UUID AND collection = %s;",
                    (user_id, self.name),
                )
                row = cur.fetchone()
                return row.get("count", 0) if row else 0

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        if not (len(ids) == len(embeddings) == len(documents) == len(metadatas)):
            raise ValueError("ids/embeddings/documents/metadatas length mismatch")
        if not ids:
            return

        user_id = get_current_user_id()
        with connection() as conn:
            with conn.cursor() as cur:
                for i in range(len(ids)):
                    # Convert embedding float list to pgvector bracketed string representation
                    emb_str = "[" + ",".join(map(str, embeddings[i])) + "]"
                    meta_json = json.dumps(metadatas[i])
                    cur.execute(
                        """
                        INSERT INTO chunks (id, user_id, collection, embedding, document, metadata)
                        VALUES (%s, %s::UUID, %s, %s::vector, %s, %s::JSONB)
                        ON CONFLICT (id) DO UPDATE SET
                            embedding = EXCLUDED.embedding,
                            document = EXCLUDED.document,
                            metadata = EXCLUDED.metadata;
                        """,
                        (ids[i], user_id, self.name, emb_str, documents[i], meta_json),
                    )

    def delete(self, ids: list[str] | None = None) -> None:
        if not ids:
            return
        user_id = get_current_user_id()
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM chunks WHERE user_id = %s::UUID AND collection = %s AND id = ANY(%s);",
                    (user_id, self.name, ids),
                )

    def get_by_ids(self, ids: list[str]) -> dict[str, list]:
        if not ids:
            return {"ids": [], "documents": [], "metadatas": []}
        user_id = get_current_user_id()
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, document, metadata FROM chunks
                    WHERE user_id = %s::UUID AND collection = %s AND id = ANY(%s);
                    """,
                    (user_id, self.name, ids),
                )
                rows = cur.fetchall()
                out = {"ids": [], "documents": [], "metadatas": []}
                for r in rows:
                    out["ids"].append(r["id"])
                    out["documents"].append(r["document"])
                    meta = r["metadata"]
                    if isinstance(meta, str):
                        meta = json.loads(meta)
                    out["metadatas"].append(meta)
                return out

    def get(
        self,
        where: dict | None = None,
        include: list[str] | None = None,
    ) -> dict[str, list]:
        include = include if include is not None else ["documents", "metadatas"]
        user_id = get_current_user_id()

        query_sql = "SELECT id"
        if "documents" in include:
            query_sql += ", document"
        if "metadatas" in include:
            query_sql += ", metadata"
        query_sql += " FROM chunks WHERE user_id = %s::UUID AND collection = %s"

        params = [user_id, self.name]
        where_sql = _compile_where(where, params)
        query_sql += where_sql

        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query_sql, params)
                rows = cur.fetchall()

                out: dict[str, list] = {"ids": [r["id"] for r in rows]}
                if "documents" in include:
                    out["documents"] = [r["document"] for r in rows]
                if "metadatas" in include:
                    out["metadatas"] = [
                        json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"]
                        for r in rows
                    ]
                return out

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int = 5,
        include: list[str] | None = None,
        where: dict | None = None,
    ) -> dict[str, list[list]]:
        include = include if include is not None else ["documents", "metadatas", "distances"]
        user_id = get_current_user_id()

        sql_params = [user_id, self.name]
        where_sql = _compile_where(where, sql_params)

        ids_out, docs_out, metas_out, dists_out = [], [], [], []

        with connection() as conn:
            with conn.cursor() as cur:
                for q in query_embeddings:
                    # Convert to pgvector float representation
                    q_str = "[" + ",".join(map(str, q)) + "]"
                    query_sql = f"""
                        SELECT id, document, metadata, (embedding <=> %s::vector) as distance
                        FROM chunks
                        WHERE user_id = %s::UUID AND collection = %s{where_sql}
                        ORDER BY distance ASC
                        LIMIT %s;
                    """
                    params = [q_str] + sql_params + [n_results]
                    cur.execute(query_sql, params)
                    rows = cur.fetchall()

                    ids_out.append([r["id"] for r in rows])
                    if "documents" in include:
                        docs_out.append([r["document"] for r in rows])
                    if "metadatas" in include:
                        metas_out.append([
                            json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"]
                            for r in rows
                        ])
                    if "distances" in include:
                        dists_out.append([float(r["distance"]) for r in rows])

        out = {"ids": ids_out}
        if "documents" in include: out["documents"] = docs_out
        if "metadatas" in include: out["metadatas"] = metas_out
        if "distances" in include: out["distances"] = dists_out
        return out
