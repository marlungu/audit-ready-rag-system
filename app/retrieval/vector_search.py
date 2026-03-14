from typing import List, Dict, Any
import math

import psycopg

from app.config import settings
from app.embeddings.titan_embedder import TitanEmbedder


class VectorSearcher:
    DEFAULT_K = 5

    def __init__(self) -> None:
        self.embedder = TitanEmbedder()

    def _to_vector_literal(self, values: list[float]) -> str:
        clean_values = []
        for value in values:
            if not math.isfinite(value):
                raise ValueError("Query embedding contains a non-finite value.")
            clean_values.append(f"{value:.12f}")
        return "[" + ",".join(clean_values) + "]"


    def _normalize_query(self, query: str) -> str:
        q = query.lower().strip()

        if (
            "become us citizen" in q
            or "become a us citizen" in q
            or "become a u.s. citizen" in q
            or "become an american citizen" in q
            or "become citizen" in q
            or "become a citizen" in q
            or "what does it take to become a citizen" in q
            or "what's going to take in order to become a citizen" in q
            or "requirements to become a citizen" in q
            or "requirements to become a naturalized citizen" in q
            or "u.s. citizen" in q
            or "us citizen" in q
            or "citizenship" in q
            or "naturalization" in q
            or "get citizenship" in q
            or "apply for citizenship" in q
        ):
            return "eligibility requirements for naturalization"
        
        if (
            ("18" in q or "18 years old" in q or "age" in q)
            and (
                "citizen" in q
                or "citizenship" in q
                or "naturalization" in q
            )
        ):
            return "naturalization minimum age requirement"

        if (
            "left the country" in q
            or "leave the country" in q
            or "out of the country" in q
            or "outside the us" in q
            or "outside the u.s." in q
            or "continuous residence" in q
            or "physical presence" in q
        ):
            return "continuous residence requirements for naturalization"

        if (
            "english test" in q
            or "civics test" in q
            or "english requirement" in q
            or "english and civics" in q
            or "english exemption" in q
        ):
            return "english and civics requirements for naturalization"

        if (
            "good moral character" in q
            or "arrested" in q
            or "crime" in q
            or "criminal" in q
            or "lied" in q
            or "false testimony" in q
            or "disqualif" in q
        ):
            return "good moral character requirements for naturalization"

        if (
            "military" in q
            or "armed forces" in q
            or "honorable service" in q
        ):
            return "naturalization through military service"
        
        if "oath of allegiance" in q:
            return "oath of allegiance naturalization requirements"

        # Lawful permanent residence / green card / adjustment of status
        if (
            "legal resident" in q
            or "become a resident" in q
            or "become a legal resident" in q
            or "lawful permanent resident" in q
            or "permanent resident" in q
            or "permanent residence" in q
            or "green card" in q
            or "get a green card" in q
            or "lpr" in q
        ):
            return "how to become a lawful permanent resident green card eligibility"
        if (
            "adjustment of status" in q
            or "adjust status" in q
            or "change my status" in q
            or "become a resident from inside the united states" in q
        ):
            return "adjustment of status eligibility requirements"
        # Admissibility / inadmissibility
        if (
            "inadmissible" in q
            or "inadmissibility" in q
            or "admissibility" in q
            or "can i be denied entry" in q
            or "grounds of inadmissibility" in q
        ):
            return "grounds of inadmissibility and admissibility requirements"

        # Waivers
        if (
            "waiver" in q
            or "forgiveness" in q
            or "waive" in q
        ):
            return "waiver of inadmissibility requirements"

        # Refugees / asylum / humanitarian
        if (
            "asylum" in q
            or "asylee" in q
            or "refugee" in q
            or "humanitarian" in q
        ):
            return "refugee asylee and humanitarian protection eligibility"
         # Temporary protected status
        if (
            "temporary protected status" in q
            or "tps" in q
        ):
            return "temporary protected status eligibility requirements"

        # Nonimmigrants / visas
        if (
            "nonimmigrant" in q
            or "temporary visa" in q
            or "stay temporarily" in q
            or "visitor visa" in q
            or "student visa" in q
            or "work visa" in q
        ):
            return "nonimmigrant classification eligibility requirements"

        # Employment-based immigration
        if (
            "employment based" in q
            or "employment-based" in q
            or "immigrate through work" in q
            or "work based green card" in q
        ):
            return "employment based immigrant eligibility"
        # Family-based immigration
        if (
            "family petition" in q
            or "family based" in q
            or "family-based" in q
            or "petition for my spouse" in q
            or "petition for my parent" in q
            or "petition for my child" in q
        ):
            return "family based immigration eligibility"

        return query.strip()
    

    def _expand_queries(self, query: str) -> list[str]:
        q = query.lower().strip()
        queries = [self._normalize_query(query)]

        if (
            "green card" in q
            or "legal resident" in q
            or "us legal resident" in q
            or "permanent resident" in q
            or "lawful permanent resident" in q
            or "lpr" in q
        ):
            queries.extend([
                "lawful permanent resident eligibility",
                "adjustment of status eligibility requirements",
                "immigrant visa process lawful permanent residence",
                "form I-485 adjustment of status green card",
            ])

        if (
            "become a citizen" in q
            or "citizenship" in q
            or "naturalization" in q
        ):
            queries.extend([
                "general eligibility requirements for naturalization",
                "naturalization requirements lawful permanent resident",
            ])

        if (
            "inadmissible" in q
            or "inadmissibility" in q
            or "denied entry" in q
        ):
            queries.extend([
                "grounds of inadmissibility",
                "inadmissibility under INA 212",
            ])

        if "waiver" in q or "forgiveness" in q:
            queries.extend([
                "waiver of inadmissibility requirements",
                "grounds of inadmissibility waiver eligibility",
            ])

        if (
            "adjustment of status" in q
            or "adjust status" in q
        ):
            queries.extend([
                "adjustment of status eligibility requirements",
                "form I-485 adjustment of status",
            ])

        if "what form" in q and (
            "green card" in q
            or "legal resident" in q
            or "lawful permanent resident" in q
            or "adjustment of status" in q
        ):
            queries.extend([
                "form I-485 application to register permanent residence or adjust status",
                "green card application form I-485",
            ])

        seen = set()
        deduped = []
        for item in queries:
            cleaned = item.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                deduped.append(cleaned)

        return deduped[:4]


    def _build_result(self, row: tuple, matched_query: str) -> Dict[str, Any]:
        distance = float(row[4])

        return {
            "content": row[3],
            "metadata": {
                "document_title": row[0],
                "page_number": row[1],
                "chunk_index": row[2],
            },
            "distance": distance,
            "similarity": 1 - distance,
            "matched_query": matched_query,
        }


    def search(self, query: str, k:int = DEFAULT_K) -> List[Dict[str, Any]]:
        expanded_queries = self._expand_queries(query)

        psycopg_url = settings.postgres_url.replace(
            "postgresql+psycopg://",
            "postgresql://",
            1,
        )

        sql = """
        SELECT
            document_title,
            page_number,
            chunk_index,
            content,
            embedding <=> %s::vector AS distance
        FROM document_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """
        
        best_results: dict[tuple[str, int, int], Dict[str, Any]] = {}

        with psycopg.connect(psycopg_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SET ivfflat.probes = 10;")

                for expanded_query in expanded_queries:
                        query_embedding = self.embedder.embed_text(expanded_query)
                        vector_literal = self._to_vector_literal(query_embedding)

                        cur.execute(sql, (vector_literal, vector_literal, int(k)))
                        rows = cur.fetchall()

                        for row in rows:
                            result = self._build_result(row, expanded_query)
                            meta = result["metadata"]
                            key = (
                                meta["document_title"],
                                meta["page_number"],
                                meta["chunk_index"],
                            )

                            if (
                                key not in best_results
                                or result["similarity"] > best_results[key]["similarity"]
                            ):
                                best_results[key] = result

        ranked_results = sorted(
            best_results.values(),
            key=lambda r: r["similarity"],
            reverse=True,
        )

        return ranked_results[:k]