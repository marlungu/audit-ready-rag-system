import logging
from typing import Any
import math

import numpy as np
import psycopg

from app.config import settings
from app.embeddings.titan_embedder import TitanEmbedder
from app.rag.query_rewriter import QueryRewriter

logger = logging.getLogger(__name__)


class VectorSearcher:
    def __init__(self) -> None:
        self.embedder = TitanEmbedder()
        self.rewriter = QueryRewriter()
    
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_vector_literal(values: list[float]) -> str:
        clean_values = []
        for value in values:
            if not math.isfinite(value):
                raise ValueError("Query embedding contains a non-finite value.")
            clean_values.append(f"{value:.12f}")
        return "[" + ",".join(clean_values) + "]"

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        va = np.array(a, dtype=np.float64)
        vb = np.array(b, dtype=np.float64)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)

    
    # ------------------------------------------------------------------
    # MMR re-ranking
    # ------------------------------------------------------------------

    def _mmr_rerank(
        self,
        query_embedding: list[float],
        candidates: list[dict[str, Any]],
        k: int,
        mmr_lambda: float,
    ) -> list[dict[str, Any]]:
        """Maximal Marginal Relevance: select results that are both
        relevant to the query AND diverse from each other.

        MMR score = lambda * sim(doc, query) - (1 - lambda) * max(sim(doc, selected))
        """
        if not candidates:
            return []

        selected: list[dict[str, Any]] = []
        remaining = list(candidates)

        for _ in range(min(k, len(remaining))):
            best_score = -float("inf")
            best_idx = 0

            for idx, candidate in enumerate(remaining):
                relevance = candidate["similarity"]

                if not selected:
                    redundancy = 0.0
                else:
                    redundancy = max(
                        self._cosine_similarity(
                            candidate["embedding"], sel["embedding"]
                        )
                        for sel in selected
                    )

                mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * redundancy

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            selected.append(remaining.pop(best_idx))

        return selected
    
    
    # ------------------------------------------------------------------
    # Database query
    # ------------------------------------------------------------------

    def _fetch_candidates(
        self,
        queries: list[str],
        candidate_pool_size: int,
    ) -> list[dict[str, Any]]:
        """Run vector search for each expanded query, deduplicate, and
        return the merged candidate pool with embeddings attached."""

        sql = """
        SELECT
            document_title,
            page_number,
            chunk_index,
            content,
            embedding::text,
            embedding <=> %s::vector AS distance
        FROM document_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """

        best: dict[tuple[str, int, int], dict[str, Any]] = {}

        with psycopg.connect(settings.psycopg_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SET ivfflat.probes = %s;",
                    (settings.ivfflat_probes,),
                )

                for query_text in queries:
                    embedding = self.embedder.embed_text(query_text)
                    literal = self._to_vector_literal(embedding)

                    cur.execute(
                        sql, (literal, literal, int(candidate_pool_size))
                    )
                    rows = cur.fetchall()

                    for row in rows:
                        distance = float(row[5])
                        similarity = 1 - distance
                        key = (row[0], row[1], row[2])

                        if key not in best or similarity > best[key]["similarity"]:
                            raw_embedding = self._parse_pg_vector(row[4])
                            best[key] = {
                                "content": row[3],
                                "metadata": {
                                    "document_title": row[0],
                                    "page_number": row[1],
                                    "chunk_index": row[2],
                                },
                                "embedding": raw_embedding,
                                "distance": distance,
                                "similarity": similarity,
                                "matched_query": query_text,
                            }

        return sorted(
            best.values(), key=lambda r: r["similarity"], reverse=True
        )

    @staticmethod
    def _parse_pg_vector(text: str) -> list[float]:
        """Parse a pgvector text representation like '[0.1,0.2,...]'."""
        return [float(x) for x in text.strip("[]").split(",")]


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------


    def search(self, query: str, k: int | None = None,) -> tuple[list[dict[str, Any]], list[str]]:
        """Search with LLM query rewriting and MMR re-ranking.

        Returns (results, expanded_queries) so callers can log the
        query expansion for audit.
        """
        if k is None:
            k = settings.top_k

        expanded_queries = self.rewriter.rewrite(query)
        logger.info("Expanded queries: %s", expanded_queries)

        candidate_pool_size = k * 3
        candidates = self._fetch_candidates(expanded_queries, candidate_pool_size)

        if not candidates:
            return [], expanded_queries

        # Use the first query embedding as the reference for MMR
        query_embedding = self.embedder.embed_text(expanded_queries[0])

        reranked = self._mmr_rerank(
            query_embedding=query_embedding,
            candidates=candidates,
            k=k,
            mmr_lambda=settings.mmr_lambda,
        )

        # Strip embeddings from results before returning
        for result in reranked:
            result.pop("embedding", None)

        return reranked, expanded_queries
