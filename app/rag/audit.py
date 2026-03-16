import json
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import text

from app.db import engine
from app.config import settings

logger = logging.getLogger(__name__)


def generate_query_id() -> str:
    return str(uuid.uuid4())


def log_query(
    query_id: str,
    question: str,
    normalized_queries: list[str],
    answer: str,
    confidence_level: str,
    retrieved_chunks: list[dict],
    used_chunks: list[dict],
    top_similarity: float | None,
    top_k: int,
    temperature: float,
) -> None:
    """Write a full audit record for every query processed by the system."""
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO query_audit_log (
                        query_id,
                        question,
                        normalized_queries,
                        answer,
                        confidence_level,
                        model_id,
                        embedding_model_id,
                        temperature,
                        top_k,
                        top_similarity,
                        chunks_retrieved,
                        chunks_used,
                        retrieved_chunks,
                        used_chunks,
                        created_at
                    ) VALUES (
                        :query_id,
                        :question,
                        CAST(:normalized_queries AS JSONB),
                        :answer,
                        :confidence_level,
                        :model_id,
                        :embedding_model_id,
                        :temperature,
                        :top_k,
                        :top_similarity,
                        :chunks_retrieved,
                        :chunks_used,
                        CAST(:retrieved_chunks AS JSONB),
                        CAST(:used_chunks AS JSONB),
                        :created_at
                    )
                    """
                ),
                {
                    "query_id": query_id,
                    "question": question,
                    "normalized_queries": json.dumps(normalized_queries),
                    "answer": answer,
                    "confidence_level": confidence_level,
                    "model_id": settings.chat_model_id,
                    "embedding_model_id": settings.embedding_model_id,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_similarity": top_similarity,
                    "chunks_retrieved": len(retrieved_chunks),
                    "chunks_used": len(used_chunks),
                    "retrieved_chunks": json.dumps(
                        _sanitize_chunks(retrieved_chunks)
                    ),
                    "used_chunks": json.dumps(
                        _sanitize_chunks(used_chunks)
                    ),
                    "created_at": datetime.now(timezone.utc),
                },
            )
        logger.info("Audit log written for query_id=%s", query_id)
    except Exception as exc:
        logger.error(
            "Failed to write audit log for query_id=%s: %s",
            query_id,
            exc,
        )


def _sanitize_chunks(chunks: list[dict]) -> list[dict]:
    """Strip embedding vectors from chunks before logging."""
    sanitized = []
    for chunk in chunks:
        entry = {
            "content": chunk.get("content", "")[:500],
            "metadata": chunk.get("metadata", {}),
            "similarity": chunk.get("similarity"),
            "distance": chunk.get("distance"),
            "matched_query": chunk.get("matched_query"),
        }
        sanitized.append(entry)
    return sanitized
