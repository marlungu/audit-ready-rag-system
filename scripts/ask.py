"""
Interactive CLI for the USCIS Policy RAG system.

Usage:
    python -m scripts.ask
"""

import logging

from app.rag.answer_generator import AnswerGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def main():
    generator = AnswerGenerator()

    print("USCIS Policy Assistant")
    print("Type 'exit' to quit.\n")

    while True:
        print()
        question = input("Ask a USCIS policy question: ").strip()

        if not question:
            continue

        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        response = generator.answer(question)

        # Confidence indicator
        confidence_label = {
            "high": "HIGH",
            "low": "LOW (partial match)",
            "insufficient": "INSUFFICIENT",
        }.get(response.confidence.value, response.confidence.value)

        print(f"\nConfidence: {confidence_label}")
        print("-" * 50)
        print(response.answer)

        # Sources
        print("\nSources")
        print("-" * 50)
        if response.sources:
            seen = set()
            for src in response.sources:
                label = f"{src.document_title}, PDF Page {src.page_number} (sim: {src.similarity:.4f})"
                if label not in seen:
                    seen.add(label)
                    print(f"  - {label}")
        else:
            print("  None")

        # Retrieval metadata
        print("\nRetrieval")
        print("-" * 50)
        print(f"  Chunks retrieved: {response.retrieval.total_chunks_retrieved}")
        print(f"  Chunks sent to model: {response.retrieval.chunks_sent_to_model}")
        if response.retrieval.top_similarity is not None:
            print(f"  Top similarity: {response.retrieval.top_similarity:.4f}")
        print(f"  Expanded queries: {response.retrieval.expanded_queries}")

        # Audit
        print(f"\n  Audit ID: {response.audit.query_id}")
        print(f"  Model: {response.audit.model_id}")
        print("-" * 70)


if __name__ == "__main__":
    main()