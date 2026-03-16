"""Tests for the query rewriter parsing logic."""

from unittest.mock import MagicMock

from app.rag.query_rewriter import QueryRewriter


class TestQueryRewriterParsing:

    def test_parse_clean_json(self):
        rewriter = QueryRewriter(llm=MagicMock())
        result = rewriter._parse_response(
            '{"normalized": "test query", "expanded": ["query 2"]}'
        )
        assert result["normalized"] == "test query"
        assert result["expanded"] == ["query 2"]

    def test_parse_json_with_markdown_fences(self):
        rewriter = QueryRewriter(llm=MagicMock())
        result = rewriter._parse_response(
            '```json\n{"normalized": "test", "expanded": []}\n```'
        )
        assert result["normalized"] == "test"

    def test_fallback_on_invalid_json(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "not valid json at all"
        rewriter = QueryRewriter(llm=mock_llm)

        queries = rewriter.rewrite("How do I become a citizen?")
        assert queries == ["How do I become a citizen?"]

    def test_deduplication(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            '{"normalized": "naturalization requirements", '
            '"expanded": ["naturalization requirements", "citizenship eligibility"]}'
        )
        rewriter = QueryRewriter(llm=mock_llm)

        queries = rewriter.rewrite("How do I become a citizen?")
        assert len(queries) == len(set(queries))

    def test_max_four_queries(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            '{"normalized": "q1", "expanded": ["q2", "q3", "q4", "q5", "q6"]}'
        )
        rewriter = QueryRewriter(llm=mock_llm)

        queries = rewriter.rewrite("test")
        assert len(queries) <= 4
