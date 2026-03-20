import logging

from sqlalchemy import create_engine, text, event
from sqlalchemy.pool import QueuePool
from pgvector.psycopg import register_vector


from app.config import settings

logger = logging.getLogger(__name__)

engine = create_engine(
    settings.postgres_url, 
    future=True,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

@event.listens_for(engine, "connect")
def on_connect(dbapi_connection, connection_record):
    try:
        register_vector(dbapi_connection)
    except Exception:
        logger.debug("pgvector extension not yet available, skipping registration")

def check_database_health() -> dict:
    """Return database health status for the /health endpoint."""
    try:
        with engine.connect() as conn:
            postgres_version = conn.execute(text("SELECT version()")).scalar()
            ext = conn.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            ).scalar()
            chunk_count = conn.execute(
                text("SELECT COUNT(*) FROM document_chunks")
            ).scalar()

        return {
            "status": "healthy",
            "postgres_version": postgres_version,
            "pgvector_enabled": ext == "vector",
            "indexed_chunks": chunk_count,
        }
    except Exception as exc:
        logger.error("Database health check failed: %s", exc)
        return {"status": "unhealthy", "error": str(exc)}


