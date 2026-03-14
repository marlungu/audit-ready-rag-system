from app.retrieval.vector_search import VectorSearcher
from app.rag.llm_client import BedrockClaudeClient

from app.rag.query_logger import log_query

from app.config import settings


MAX_CONTEXT_CHUNKS = 3
MIN_CONTEXT_SIMILARITY = 0.50


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

        return []
    

    def build_prompt(self, question: str, results: list[dict]) -> tuple[str, list[dict]]:
        used_chunks = self._select_context_chunks(results)
        context_blocks = []

        for r in used_chunks:
            meta = r["metadata"]

            source_label = f"{meta['document_title']}, PDF Page {meta['page_number']}"

            block = (
                f"[{source_label}]\n"
                f"Content:\n{r['content']}\n"
            )

            context_blocks.append(block)

        context = "\n---\n".join(context_blocks)

        prompt = f"""
You are a legal research assistant that answers questions using only the USCIS immigration Manual.

Use ONLY the information provided in the retrieved context.
Do not use outside knowledge.

Instructions:
1. Answer the user's question clearly and directly.
2. Start with the answer directly. Do not begin with phrases like "Based on the sources provided" or "Based on the USCIS Policy Manual."
3. Summarize the relevant USCIS policy rules in plain English.
4. Combine overlapping information from multiple chunks.
5. Avoid repeating the same rule multiple times.
6. Use bullet points or numbered lists when helpful.

If the answer cannot be found in the sources, say exactly:
"I do not have enough information from the USCIS Policy Manual."

Rules:
- Do not invent facts.
- Do not guess.
- Provide policy information only. Do not provide personalized legal advice.
- Keep the answer clear, direct and factual, and concise.
- Synthesize repeated facts into one clear statement.
- Do not repeat the same point in different words.

Citations:
When you cite a source, use the exact source label provided in brackets.
Example:
[USCIS Policy Manual Volume 12 2026, PDF Page 2330]

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

        used_chunks = self._select_context_chunks(results)

        if not used_chunks:
            answer_text = "I do not have enough information from the USCIS Policy Manual."

            log_query(
                question=question,
                answer=answer_text,
                retrieved_chunks=results,
                top_k=k,
            )

            return {
                "question": question,
                "answer": answer_text,
                "sources": [],
                "retrieved_chunks": results,
                "used_chunks": [],
            }
        prompt, used_chunks = self.build_prompt(question, results=results)
        answer_text = self.llm.generate(prompt)

        answer_text = answer_text.replace(
    "This is general policy information only and does not constitute personalized legal advice.",
    ""
).strip()

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
