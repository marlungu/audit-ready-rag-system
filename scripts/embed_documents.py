import logging
import sys

from sqlalchemy import text

from app.db import engine
from app.ingestion.loader import load_pdf_from_s3
from app.ingestion.chunker import chunk_documents
from app.ingestion.quality import validate_chunks, filter_valid_chunks
from app.embeddings.titan_embedder import TitanEmbedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DOCUMENT_KEY = "docs/policy-manual/uscis_policy_manual_full_2026.pdf"


def to_pgvector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(x) for x in vector) + "]"


def main():
    logger.info("Loading pages from S3...")
    pages = load_pdf_from_s3(DOCUMENT_KEY)
    logger.info(f"Loaded %d cleaned pages", len(pages))

    if not pages:
        logger.error("No pages were loaded from S3.")
        sys.exit(1)

    logger.info("Chunking full USCIS Policy Manual...")
    chunks = chunk_documents(pages)
    logger.info("Created %d chunks", len(chunks))

    if not chunks:
        logger.error("No chunks were created.")
        sys.exit(1)
    
     # Quality validation gate
    logger.info("Running quality validation...")
    report = validate_chunks(chunks)
    logger.info(
        "Quality report: %d passed, %d failed (%.1f%% failure rate)",
        report.passed,
        report.failed,
        report.failure_rate * 100,
    )

    if report.failure_rate > 0.20:
        logger.error(
            "Failure rate %.1f%% exceeds 20%% threshold. "
            "Review ingestion pipeline before proceeding.",
            report.failure_rate * 100,
        )
        sys.exit(1)

    valid_chunks = filter_valid_chunks(chunks, report)
    logger.info("%d chunks passed quality gate", len(valid_chunks))

    embedder = TitanEmbedder()

    logger.info("Clearing existing rows from document_chunks...")
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM document_chunks"))

    logger.info("Generating embeddings and inserting chunks...")
    inserted = 0

    with engine.begin() as connection:
        for i, chunk in enumerate(valid_chunks, start=1):
            vector = embedder.embed_text(chunk.page_content)
            vector_literal = to_pgvector_literal(vector)

            connection.execute(
                text(
                    """
                    INSERT INTO document_chunks (
                        document_title,
                        page_number,
                        chunk_index,
                        content,
                        section_heading,
                        embedding
                    )
                    VALUES (
                        :document_title,
                        :page_number,
                        :chunk_index,
                        :content,
                        :section_heading,
                        CAST(:embedding AS vector)
                    )
                    """
                ),
                {
                    "document_title": chunk.metadata["document_title"],
                    "page_number": chunk.metadata["page_number"],
                    "chunk_index": chunk.metadata["chunk_index"],
                    "content": chunk.page_content,
                    "section_heading":
                        chunk.metadata.get("section_heading", ""),
                    "embedding": vector_literal,
                },
            )

            inserted += 1

            if i % 25 == 0 or i == len(valid_chunks):
                logger.info(f"Inserted %d/%d chunks", i, len(valid_chunks))

    logger.info("Done. Inserted %d chunks into document_chunks.", inserted)



if __name__ == "__main__":
    main()