import json
import logging

from app.rag.llm_client import BedrockClaudeClient

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """You are a query rewriter for a USCIS immigration policy search system.

Given a user question, produce a JSON object with two fields:
- "normalized": a single clear search query rewritten for semantic similarity search against USCIS policy documents.
- "expanded": a list of 1-3 additional search queries that cover related policy areas the user might need.

Rules:
- Rewrite casual or vague language into precise immigration terminology.
- Use terms from the USCIS Policy Manual (naturalization, adjustment of status, lawful permanent resident, inadmissibility, etc).
- Keep each query under 15 words.
- Do not invent policy topics that do not exist.
- Return ONLY valid JSON with no other text.

Examples:

User: "How do I become a US citizen?"
{{"normalized": "eligibility requirements for naturalization", "expanded": ["naturalization application process", "continuous residence requirements for naturalization"]}}

User: "Can I get deported if I have a green card?"
{{"normalized": "grounds for removal of lawful permanent resident", "expanded": ["deportability grounds for LPR", "criminal grounds for removal"]}}

User: "What's a green card?"
{{"normalized": "lawful permanent resident status eligibility", "expanded": ["adjustment of status requirements", "immigrant visa categories"]}}

User question: "{question}"
"""


class QueryRewriter:
    def __init__(self, llm: BedrockClaudeClient | None = None):
        self._llm = llm

    @property
    def llm(self) -> BedrockClaudeClient:
        if self._llm is None:
            self._llm = BedrockClaudeClient()
        return self._llm

    def rewrite(self, question: str) -> list[str]:
        """Rewrite the user question into optimized search queries.

        Returns a list of 2-4 deduplicated query strings, with the
        normalized query first.
        """
        try:
            prompt = REWRITE_PROMPT.format(question=question)
            raw = self.llm.generate(prompt, max_tokens=300, temperature=0.0)
            parsed = self._parse_response(raw)

            queries = [parsed["normalized"]]
            for q in parsed.get("expanded", []):
                cleaned = q.strip()
                if cleaned and cleaned not in queries:
                    queries.append(cleaned)

            return queries[:4]

        except Exception as exc:
            logger.warning(
                "Query rewriter failed, falling back to original: %s", exc
            )
            return [question.strip()]

    def _parse_response(self, raw: str) -> dict:
        """Extract JSON from the LLM response, handling markdown fences."""
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
        return json.loads(text)
