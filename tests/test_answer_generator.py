"""Tests for the answer generator logic (confidence, prompt construction)."""

from unittest.mock import MagicMock, patch

from app.models import ConfidenceLevel
from app.rag.answer_generator import AnswerGenerator


def _make_result(similarity: float, content: str = "Policy text") -> dict:
    return {
        "content": content,
        "metadata": {
            "document_title": "USCIS Policy Manual",
            "page_number": 1,
            "chunk_index": 0,
        },
        "similarity": similarity,
        "distance": 1 - similarity,
        "matched_query": "test query",
    }


class TestConfidenceClassification:

    def setup_method(self):
        with patch.object(AnswerGenerator, "__init__", lambda self: None):
            self.gen = AnswerGenerator()
            self.gen.searcher = MagicMock()
            self.gen.llm = MagicMock()

    def test_high_confidence(self):
        results = [_make_result(0.85), _make_result(0.75)]
        level, chunks = self.gen._classify_confidence(results)
        assert level == ConfidenceLevel.HIGH
        assert len(chunks) > 0

    def test_low_confidence(self):
        results = [_make_result(0.60), _make_result(0.55)]
        level, chunks = self.gen._classify_confidence(results)
        assert level == ConfidenceLevel.LOW
        assert len(chunks) > 0

    def test_insufficient_confidence(self):
        results = [_make_result(0.30), _make_result(0.20)]
        level, chunks = self.gen._classify_confidence(results)
        assert level == ConfidenceLevel.INSUFFICIENT
        assert len(chunks) == 0

    def test_empty_results(self):
        level, chunks = self.gen._classify_confidence([])
        assert level == ConfidenceLevel.INSUFFICIENT
        assert chunks == []

    def test_max_context_chunks_respected(self):
        results = [_make_result(0.90) for _ in range(10)]
        level, chunks = self.gen._classify_confidence(results)
        assert level == ConfidenceLevel.HIGH
        # Should be capped by settings.max_context_chunks (default 3)
        assert len(chunks) <= 3


class TestPromptConstruction:

    def setup_method(self):
        with patch.object(AnswerGenerator, "__init__", lambda self: None):
            self.gen = AnswerGenerator()
            self.gen.searcher = MagicMock()
            self.gen.llm = MagicMock()

    def test_prompt_contains_question(self):
        chunks = [_make_result(0.85, "Some policy text about naturalization.")]
        prompt = self.gen.build_prompt("What is naturalization?", chunks)
        assert "What is naturalization?" in prompt

    def test_prompt_contains_context(self):
        chunks = [_make_result(0.85, "Applicants must be 18 years old.")]
        prompt = self.gen.build_prompt("Age requirement?", chunks)
        assert "Applicants must be 18 years old." in prompt

    def test_prompt_contains_citation_format(self):
        chunks = [_make_result(0.85)]
        prompt = self.gen.build_prompt("Test?", chunks)
        assert "USCIS Policy Manual, PDF Page 1" in prompt

    def test_prompt_contains_system_instructions(self):
        chunks = [_make_result(0.85)]
        prompt = self.gen.build_prompt("Test?", chunks)
        assert "Do not invent facts" in prompt