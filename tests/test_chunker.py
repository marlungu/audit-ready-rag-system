"""Tests for section-aware chunking logic."""

from langchain_core.documents import Document

from app.ingestion.chunker import (
    clean_text,
    is_heading,
    build_sections,
    chunk_documents,
)


class TestCleanText:

    def test_removes_blank_lines(self):
        result = clean_text("line 1\n\n\nline 2")
        assert result == "line 1\nline 2"

    def test_strips_skip_patterns(self):
        text = "Affected Sections\nReal content\nRead More"
        result = clean_text(text)
        assert "Real content" in result
        assert "Affected Sections" not in result
        assert "Read More" not in result

    def test_strips_urls(self):
        text = "Content here\nhttps://example.com/foo\nMore content"
        result = clean_text(text)
        assert "https://" not in result

    def test_empty_input(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""


class TestIsHeading:

    def test_volume_heading(self):
        assert is_heading("Volume 12 - Citizenship & Naturalization") is True

    def test_part_heading(self):
        assert is_heading("Part A - General Requirements") is True

    def test_chapter_heading(self):
        assert is_heading("Chapter 3 - Continuous Residence") is True

    def test_subsection_heading(self):
        assert is_heading("A. Overview") is True

    def test_normal_text(self):
        assert is_heading("The applicant must file Form N-400.") is False


class TestBuildSections:

    def test_groups_under_heading(self):
        pages = [
            Document(
                page_content="Volume 12 Overview\nBody text about citizenship.",
                metadata={"page_number": 1},
            ),
        ]
        sections = build_sections(pages)
        assert len(sections) >= 1
        assert "Volume 12" in sections[0].metadata.get("section_heading", "")

    def test_flushes_on_new_volume(self):
        pages = [
            Document(
                page_content="Volume 12 First\nContent one.",
                metadata={"page_number": 1},
            ),
            Document(
                page_content="Volume 7 Second\nContent two.",
                metadata={"page_number": 2},
            ),
        ]
        sections = build_sections(pages)
        assert len(sections) >= 2


class TestChunkDocuments:

    def test_produces_chunks_with_metadata(self):
        pages = [
            Document(
                page_content="Chapter 1 Overview\n" + ("Policy content. " * 100),
                metadata={
                    "document_title": "USCIS Manual",
                    "page_number": 1,
                    "source_key": "test.pdf",
                    "s3_bucket": "test",
                },
            )
        ]
        chunks = chunk_documents(pages)
        assert len(chunks) > 0
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata

    def test_chunk_overlap(self):
        """When a section is long enough to split, adjacent chunks share words."""
        pages = [
            Document(
                page_content=(
                    "Chapter 1 Overview\n"
                    "The applicant must meet several requirements. " * 200
                ),
                metadata={
                    "document_title": "USCIS Manual",
                    "page_number": 1,
                    "source_key": "test.pdf",
                    "s3_bucket": "test",
                },
            )
        ]
        chunks = chunk_documents(pages)
        # Find two consecutive chunks that are both body text (not just heading)
        body_chunks = [c for c in chunks if len(c.page_content) > 200]
        if len(body_chunks) >= 2:
            c1_words = set(body_chunks[0].page_content.split())
            c2_words = set(body_chunks[1].page_content.split())
            assert len(c1_words & c2_words) > 0