import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


# Pre-compiled heading patterns for document structure detection
RE_VOLUME = re.compile(r"^\s*Volume\s+\d+\b.*$", re.IGNORECASE)
RE_PART = re.compile(r"^\s*Part\s+[A-Z]\b.*$", re.IGNORECASE)
RE_CHAPTER = re.compile(r"^\s*Chapter\s+\d+\b.*$", re.IGNORECASE)
RE_SUBSECTION = re.compile(r"^\s*[A-Z]\.\s+.*$", re.IGNORECASE)

HEADING_PATTERNS = [RE_VOLUME, RE_PART, RE_CHAPTER, RE_SUBSECTION]

SKIP_PATTERNS = [
    re.compile(r"^\s*Affected Sections\s*$", re.IGNORECASE),
    re.compile(r"^\s*Read More\s*$", re.IGNORECASE),
    re.compile(r"^\s*Policy Manual \| USCIS\s*$", re.IGNORECASE),
    re.compile(r"^\s*Search USCIS Policy Manual Search\s*$", re.IGNORECASE),
    re.compile(r"^\s*Current as of.*$", re.IGNORECASE),
    re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{2,4},.*$", re.IGNORECASE),
    re.compile(r"^\s*https?://.*$", re.IGNORECASE),
]


def clean_text(text: str) -> str:
    if not text:
        return ""

    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if any(pattern.match(line) for pattern in SKIP_PATTERNS):
            continue

        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines).strip()


def is_heading(line: str) -> bool:
    return any(pattern.match(line) for pattern in HEADING_PATTERNS)


def build_sections(pages: List[Document]) -> List[Document]:
    sections: List[Document] = []

    current_heading_parts: List[str] = []
    current_body_lines: List[str] = []
    current_metadata = None

    def flush_section():
        nonlocal current_heading_parts, current_body_lines, current_metadata

        if not current_body_lines or not current_metadata:
            return

        heading_text = " | ".join(current_heading_parts).strip()
        body_text = "\n".join(current_body_lines).strip()

        if heading_text:
            full_text = f"{heading_text}\n\n{body_text}"
        else:
            full_text = body_text

        section_doc = Document(
            page_content=full_text,
            metadata={
                **current_metadata,
                "section_heading": heading_text,
            },
        )
        sections.append(section_doc)

        current_body_lines = []

    for page in pages:
        cleaned = clean_text(page.page_content)
        if not cleaned:
            continue

        lines = cleaned.splitlines()

        if current_metadata is None:
            current_metadata = page.metadata.copy()

        for line in lines:
            if is_heading(line):
                if RE_VOLUME.match(line):
                    flush_section()
                    current_heading_parts = [line]
                    current_metadata = page.metadata.copy()
                elif RE_PART.match(line):
                    flush_section()
                    current_heading_parts = [
                        h for h in current_heading_parts
                        if not RE_PART.match(h)
                        and not RE_CHAPTER.match(h)
                        and not RE_SUBSECTION.match(h)
                    ]
                    current_heading_parts.append(line)
                    current_metadata = page.metadata.copy()
                elif RE_CHAPTER.match(line):
                    flush_section()
                    current_heading_parts = [
                        h for h in current_heading_parts
                        if not RE_CHAPTER.match(h)
                        and not RE_SUBSECTION.match(h)
                    ]
                    current_heading_parts.append(line)
                    current_metadata = page.metadata.copy()
                else:
                    flush_section()
                    current_heading_parts = [
                        h for h in current_heading_parts
                        if not RE_SUBSECTION.match(h)
                    ]
                    current_heading_parts.append(line)
                    current_metadata = page.metadata.copy()
            else:
                current_body_lines.append(line)

    flush_section()
    return sections


def chunk_documents(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    
    section_docs = build_sections(pages)
    chunks = splitter.split_documents(section_docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks