"""Tests for Pydantic request and response models."""

import pytest
from pydantic import ValidationError

from app.models import QueryRequest, ConfidenceLevel


class TestQueryRequest:

    def test_valid_request(self):
        req = QueryRequest(question="What are the requirements?")
        assert req.question == "What are the requirements?"
        assert req.top_k == 5

    def test_custom_top_k(self):
        req = QueryRequest(question="Test?", top_k=10)
        assert req.top_k == 10

    def test_question_too_short(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="Hi")

    def test_top_k_too_high(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="Valid question?", top_k=50)

    def test_top_k_too_low(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="Valid question?", top_k=0)


class TestConfidenceLevel:

    def test_values(self):
        assert ConfidenceLevel.HIGH.value == "high"
        assert ConfidenceLevel.LOW.value == "low"
        assert ConfidenceLevel.INSUFFICIENT.value == "insufficient"