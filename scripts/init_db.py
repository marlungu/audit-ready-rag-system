from sqlalchemy import text
from app.db import engine


def initialize_database() -> None:
    with engine.begin() as connection:

        connection.execute(
            text(
                """
                CREATE EXTENSION IF NOT EXISTS vector;
                """
            )
        )

        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    document_title TEXT NOT NULL,
                    page_number INTEGER,
                    chunk_index INTEGER,
                    section_heading TEXT,
                    content TEXT NOT NULL,
                    embedding VECTOR(1024),
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                );
                """
            )
        )

        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS query_audit_log (
                    id BIGSERIAL PRIMARY KEY,
                    query_id TEXT NOT NULL UNIQUE,
                    question TEXT NOT NULL,
                    normalized_queries JSONB NOT NULL,
                    answer TEXT NOT NULL,
                    confidence_level TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    embedding_model_id TEXT NOT NULL,
                    temperature FLOAT NOT NULL,
                    top_k INTEGER NOT NULL,
                    top_similarity FLOAT,
                    chunks_retrieved INTEGER NOT NULL,
                    chunks_used INTEGER NOT NULL,
                    retrieved_chunks JSONB NOT NULL,
                    used_chunks JSONB NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                );
                """
            )
        )


def create_vector_index() -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding_ivfflat
                ON document_chunks
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 50);
                """
            )
        )

        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_audit_query_id
                ON query_audit_log (query_id);
                """
            )
        )

        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_audit_created_at
                ON query_audit_log (created_at);
                """
            )
        )

        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_audit_confidence
                ON query_audit_log (confidence_level);
                """
            )
        )

        connection.execute(
            text(
                """
                ANALYZE document_chunks;
                """
            )
        )

        print("Indexes created successfully.")



if __name__ == "__main__":
    initialize_database()
    create_vector_index()