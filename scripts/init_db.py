from sqlalchemy import text
from app.db import engine


def initialize_database() -> None:
    with engine.begin() as connection:

        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    document_title TEXT NOT NULL,
                    page_number INTEGER,
                    chunk_index INTEGER,
                    content TEXT NOT NULL,
                    embedding VECTOR(1536)
                );
                """
            )
        )

        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding
                ON document_chunks
                USING ivfflat (embedding vector_cosine_ops);
                """
            )
        )

        print("Database schema initialized successfully.")


if __name__ == "__main__":
    initialize_database()