from app.rag.answer_generator import AnswerGenerator

PROMPT = "Ask a USCIS policy question: "


def main():
    generator = AnswerGenerator()

    print("USCIS Policy Assistant")
    print("Type 'exit' to quit.\n")

    while True:
        print()
        question = input(PROMPT).strip()

        if not question:
            continue

        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        result = generator.answer(question)

        print("\nAnswer:\n")
        print(result["answer"])
        print(f"\nContext sent to Claude: {len(result['used_chunks'])} chunks")

        print("\nRetrieved Chunks:\n")

        for i, chunk in enumerate(result["retrieved_chunks"], start=1):
            meta = chunk["metadata"]
            preview = chunk["content"][:220].replace("\n", " ")

            similarity = chunk.get("similarity")
            distance = chunk.get("distance")

            print(
                f"{i}. {meta['document_title']} | page {meta['page_number']} | "
                f"chunk {meta['chunk_index']}"
            )

            if similarity is not None and distance is not None:
                print(
                    f"   similarity={similarity:.4f} | distance={distance:.4f}"
                )
                
            print(f"   {preview}...\n")
        
        print("-" * 80)


if __name__ == "__main__":
    main()