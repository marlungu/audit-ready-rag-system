"""Tests for the chunk quality validation pipeline."""

from langchain_core.documents import Document

from app.ingestion.quality import (
    QualityIssue,
    validate_chunk,
    validate_chunks,
    filter_valid_chunks,
)


def _make_chunk(content: str, **meta_overrides) -> Document:
    meta = {
        "document_title": "USCIS Policy Manual",
        "page_number": 1,
        "chunk_index": 0,
        "section_heading": "Part A | Chapter 1",
        **meta_overrides,
    }
    return Document(page_content=content, metadata=meta)


class TestValidateChunk:

    def test_valid_chunk_passes(self):
        chunk = _make_chunk("This is a valid chunk with sufficient content. " * 5)
        issues = validate_chunk(chunk, index=0)
        assert issues == []

    def test_too_short_chunk(self):
        chunk = _make_chunk("Short.")
        issues = validate_chunk(chunk, index=0)
        assert any(i.issue_type == "too_short" for i in issues)

    def test_empty_chunk(self):
        chunk = _make_chunk("   ")
        issues = validate_chunk(chunk, index=0)
        assert any(i.issue_type == "empty" for i in issues)

    def test_corrupted_repeated_chars(self):
        chunk = _make_chunk("a" * 200)
        issues = validate_chunk(chunk, index=0)
        assert any(i.issue_type == "corrupted" for i in issues)

    def test_low_text_ratio(self):
        chunk = _make_chunk("!@#$%^&*()_+" * 20)
        issues = validate_chunk(chunk, index=0)
        assert any(i.issue_type == "low_text_ratio" for i in issues)

    def test_missing_metadata(self):
        chunk = Document(
            page_content="Valid content with enough characters to pass length check. " * 5,
            metadata={"document_title": "Test"},
        )
        issues = validate_chunk(chunk, index=0)
        assert any(
            i.issue_type == "missing_metadata" and "page_number" in i.detail
            for i in issues
        )

    def test_missing_heading_on_large_chunk(self):
        chunk = _make_chunk(
            "A substantial chunk of text. " * 30,
            section_heading="",
        )
        issues = validate_chunk(chunk, index=0)
        assert any(i.issue_type == "missing_heading" for i in issues)


class TestValidateChunks:

    def test_report_counts(self):
        chunks = [
            _make_chunk("Valid chunk with enough content. " * 5),
            _make_chunk("x"),
            _make_chunk("Another valid chunk with real content. " * 5),
        ]
        report = validate_chunks(chunks)
        assert report.total_chunks == 3
        assert report.passed == 2
        assert report.failed == 1

    def test_failure_rate(self):
        chunks = [_make_chunk("tiny")] * 4
        report = validate_chunks(chunks)
        assert report.failure_rate == 1.0


class TestFilterValidChunks:

    def test_removes_critically_failed(self):
        chunks = [
            _make_chunk("Valid chunk with enough content. " * 5),
            _make_chunk("   "),  # empty -> critical
            _make_chunk("Another valid chunk with real content. " * 5),
        ]
        report = validate_chunks(chunks)
        valid = filter_valid_chunks(chunks, report)
        assert len(valid) == 2

    def test_keeps_non_critical_failures(self):
        # A chunk that is missing heading is non-critical
        chunks = [
            _make_chunk(
                "A substantial chunk of text with enough content. " * 20,
                section_heading="",
            ),
        ]
        report = validate_chunks(chunks)
        valid = filter_valid_chunks(chunks, report)
        assert len(valid) == 1