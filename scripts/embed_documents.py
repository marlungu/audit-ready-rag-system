from app.ingestion.loader import load_pdf_from_s3
from app.ingestion.chunker import chunk_documents
from app.embeddings.titan_embedder import TitanEmbedder


def main():

    pages = load_pdf_from_s3(
        "docs/policy-manual/uscis_policy_manual_full_2026.pdf"
    )

    chunks = chunk_documents(pages)

    embedder = TitanEmbedder()

    first_chunk = chunks[0]

    vector = embedder.embed_text(first_chunk.page_content)

    print("Embedding length:", len(vector))


if __name__ == "__main__":
    main()