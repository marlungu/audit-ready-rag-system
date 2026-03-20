import logging
from datetime import datetime, timezone

from app.config import settings
from app.models import (
    AuditMetadata,
    ConfidenceLevel,
    QueryResponse,
    RetrievalMetadata,
    SourceReference,
)

from app.rag.audit import generate_query_id, log_query
from app.retrieval.vector_search import VectorSearcher
from app.rag.llm_client import BedrockClaudeClient

logger = logging.getLogger(__name__)

INSUFFICIENT_ANSWER = (
    "I do not have enough information from the USCIS Policy Manual "
    "to answer this question reliably."
)

LOW_CONFIDENCE_PREFIX = (
    "Note: The following answer is based on partially matching policy "
    "sections and may not fully address your question.\n\n"
)

SYSTEM_PROMPT = """\
You are a legal research assistant that answers questions using only the \
USCIS Policy Manual.

Use ONLY the information provided in the retrieved context. Do not use \
outside knowledge.

Instructions:
1. Answer the user's question clearly and directly.
2. Start with the answer. Do not begin with filler phrases.
3. Summarize relevant USCIS policy rules in plain English.
4. Combine overlapping information from multiple chunks.
5. Use bullet points or numbered lists when helpful.
6. Cite sources using the exact label in brackets.

If the answer cannot be found in the sources, say exactly:
"I do not have enough information from the USCIS Policy Manual."

Rules:
- Do not invent facts or guess.
- Provide policy information only, not personalized legal advice.
- Keep the answer clear, direct, factual, and concise.
- Synthesize repeated facts into one clear statement.

Citation format:
[Document Title, PDF Page N]
"""


class AnswerGenerator:
    def __init__(self) -> None:
        self.searcher = VectorSearcher()
        self.llm = BedrockClaudeClient()

    # ------------------------------------------------------------------
    # Confidence classification
    # ------------------------------------------------------------------

    def _classify_confidence(
        self, results: list[dict]
    ) -> tuple[ConfidenceLevel, list[dict]]:
        """Classify retrieval confidence and select usable chunks."""
        high = [
            r for r in results
            if r.get("similarity", 0) >= settings.high_confidence_threshold
        ]
        if high:
            return ConfidenceLevel.HIGH, high[: settings.max_context_chunks]

        low = [
            r for r in results
            if r.get("similarity", 0) >= settings.low_confidence_threshold
        ]
        if low:
            return ConfidenceLevel.LOW, low[: settings.max_context_chunks]

        return ConfidenceLevel.INSUFFICIENT, []

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def build_prompt(self, question: str, chunks: list[dict]) -> str:
        context_blocks = []

        for r in chunks:
            meta = r["metadata"]

            label = f"{meta['document_title']}, PDF Page {meta['page_number']}"
            block = f"[{label}]\nContent:\n{r['content']}\n"
            context_blocks.append(block)
        
        context = "\n---\n".join(context_blocks)

        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Sources:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            f"Answer:"
        )


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(self, question: str, top_k: int | None = None) -> QueryResponse:
        if top_k is None:
            top_k = settings.top_k

        query_id = generate_query_id()
        timestamp = datetime.now(timezone.utc)

        # Retrieve
        results, expanded_queries = self.searcher.search(question, k=top_k)

        # Classify confidence
        confidence, used_chunks = self._classify_confidence(results)
        top_sim = results[0]["similarity"] if results else None

        # Generate answer based on confidence tier
        if confidence == ConfidenceLevel.INSUFFICIENT:
            answer_text = INSUFFICIENT_ANSWER
        else:
            prompt = self.build_prompt(question, used_chunks)
            raw_answer = self.llm.generate(prompt)

            # Clean boilerplate the model sometimes appends
            raw_answer = raw_answer.replace(
                "This is general policy information only and does not "
                "constitute personalized legal advice.",
                "",
            ).strip()

            if confidence == ConfidenceLevel.LOW:
                answer_text = LOW_CONFIDENCE_PREFIX + raw_answer
            else:
                answer_text = raw_answer

        # Audit
        log_query(
            query_id=query_id,
            question=question,
            normalized_queries=expanded_queries,
            answer=answer_text,
            confidence_level=confidence.value,
            retrieved_chunks=results,
            used_chunks=used_chunks,
            top_similarity=top_sim,
            top_k=top_k,
            temperature=settings.temperature,
        )

        # Build response
        sources = [
            SourceReference(
                document_title=c["metadata"]["document_title"],
                page_number=c["metadata"]["page_number"],
                chunk_index=c["metadata"]["chunk_index"],
                similarity=c.get("similarity", 0),
                matched_query=c.get("matched_query", ""),
            )
            for c in used_chunks
        ]

        return QueryResponse(
            question=question,
            answer=answer_text,
            confidence=confidence,
            sources=sources,
            retrieval=RetrievalMetadata(
                total_chunks_retrieved=len(results),
                chunks_sent_to_model=len(used_chunks),
                top_similarity=top_sim,
                confidence_level=confidence,
                expanded_queries=expanded_queries,
            ),
            audit=AuditMetadata(
                query_id=query_id,
                model_id=settings.chat_model_id,
                embedding_model_id=settings.embedding_model_id,
                temperature=settings.temperature,
                timestamp=timestamp,
            ),
        )

