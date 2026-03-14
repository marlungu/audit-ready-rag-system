from app.retrieval.vector_search import VectorSearcher
from app.rag.llm_client import BedrockClaudeClient

from app.rag.query_logger import log_query

from app.config import settings


MAX_CONTEXT_CHUNKS = 3
MIN_CONTEXT_SIMILARITY = 0.65


class AnswerGenerator:
    def __init__(self):
        self.searcher = VectorSearcher()
        self.llm = BedrockClaudeClient()


    def _select_context_chunks(self, results: list[dict]) -> list[dict]:
        strong_results = [
            r for r in results
            if (r.get("similarity") is None or r["similarity"] >= MIN_CONTEXT_SIMILARITY)
        ]

        if strong_results:
            return strong_results[:MAX_CONTEXT_CHUNKS]

        return results[:1]
    

    def build_prompt(self, question: str, results: list[dict]) -> tuple[str, list[dict]]:
        used_chunks = self._select_context_chunks(results)
        context_blocks = []

        for r in used_chunks:
            meta = r["metadata"]

            source_label = (
                f"Vol 12 | page {meta['page_number']} | chunk {meta['chunk_index']}"
            )

            block = (
                f"[{source_label}]\n"
                f"Content:\n{r['content']}\n"
            )

            context_blocks.append(block)

        context = "\n---\n".join(context_blocks)

        prompt = f"""
You are an assistant that answers questions about USCIS immigration policy.

Use ONLY the information in the sources below.
Do not use outside knowledge.
If the answer cannot be found in the sources, say exactly:
"I do not have enough information from the USCIS Policy Manual."

Do not invent facts.
Do not guess.
Do not give legal advice.
Keep the answer clear, direct and factual.

When you cite a source, use the exact source label provided in brackets.
Example:
[Vol 12 | page 2330 | chunk 7719]

Do not invent source numbers like [Source 1].

Sources:
{context}

Question:
{question}

Answer:
"""
        return prompt.strip(), used_chunks

    def answer(self, question: str, k: int | None = None) -> dict:
        if k is None:
            k = settings.top_k

        results = self.searcher.search(question, k=k)
        prompt, used_chunks = self.build_prompt(question, results=results)
        answer_text = self.llm.generate(prompt)

        log_query(
            question=question,
            answer=answer_text,
            retrieved_chunks=results,
            top_k=k,
        )

        return {
            "question": question,
            "answer": answer_text,
            "sources": [r["metadata"] for r in used_chunks],
            "retrieved_chunks": used_chunks,
            "used_chunks": used_chunks,
        }
