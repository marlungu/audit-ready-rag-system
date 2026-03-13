from app.retrieval.vector_search import VectorSearcher
from app.rag.llm_client import BedrockClaudeClient
from app.config import settings


class AnswerGenerator:
    def __init__(self):
        self.searcher = VectorSearcher()
        self.llm = BedrockClaudeClient()

    def build_prompt(self, question: str, k: int | None = None) -> str:
        if k is None:
            k = settings.top_k

        results = self.searcher.search(question, k=k)

        context_blocks = []

        for i, r in enumerate(results, start=1):
            meta = r["metadata"]

            block = (
                f"[Source {i}]\n"
                f"Document: {meta['document_title']}\n"
                f"Page: {meta['page_number']}\n"
                f"Chunk: {meta['chunk_index']}\n"
                f"Content:\n{r['content']}\n"
            )

            context_blocks.append(block)

        context = "\n---\n".join(context_blocks)

        prompt = f"""
You are an assistant that answers questions about USCIS immigration policy.

Use ONLY the information in the sources below.

If the answer cannot be found in the sources, say:
"I do not have enough information from the USCIS Policy Manual."

Do not invent facts.
Do not give legal advice.
Keep the answer clear and direct.
Cite the source numbers you used, like [Source 1].

Sources:
{context}

Question:
{question}

Answer:
"""
        return prompt

    def answer(self, question: str, k: int | None = None) -> dict:
        if k is None:
            k = settings.top_k

        results = self.searcher.search(question, k=k)
        prompt = self.build_prompt(question, k=k)
        answer_text = self.llm.generate(prompt)

        return {
            "question": question,
            "answer": answer_text,
            "sources": [r["metadata"] for r in results],
        }
