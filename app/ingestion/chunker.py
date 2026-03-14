import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


HEADING_PATTERNS = [
    r"^\s*Volume\s+\d+\b.*$",
    r"^\s*Part\s+[A-Z]\b.*$",
    r"^\s*Chapter\s+\d+\b.*$",
    r"^\s*[A-Z]\.\s+.*$",
]


def clean_text(text: str) -> str:
    if not text:
        return ""

    lines = text.splitlines()
    cleaned_lines = []

    skip_patterns = [
        r"^\s*Affected Sections\s*$",
        r"^\s*Read More\s*$",
        r"^\s*Policy Manual \| USCIS\s*$",
        r"^\s*Search USCIS Policy Manual Search\s*$",
        r"^\s*Current as of.*$",
        r"^\s*\d{1,2}/\d{1,2}/\d{2,4},.*$",
        r"^\s*https?://.*$",
    ]

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if any(re.match(pattern, line, re.IGNORECASE) for pattern in skip_patterns):
            continue

        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines).strip()


def is_heading(line: str) -> bool:
    return any(re.match(pattern, line, re.IGNORECASE) for pattern in HEADING_PATTERNS)



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
                if re.match(r"^\s*Volume\s+\d+\b.*$", line, re.IGNORECASE):
                    flush_section()
                    current_heading_parts = [line]
                    current_metadata = page.metadata.copy()
                elif re.match(r"^\s*Part\s+[A-Z]\b.*$", line, re.IGNORECASE):
                    flush_section()
                    current_heading_parts = [
                        h for h in current_heading_parts
                        if not re.match(r"^\s*Part\s+[A-Z]\b.*$", h, re.IGNORECASE)
                        and not re.match(r"^\s*Chapter\s+\d+\b.*$", h, re.IGNORECASE)
                        and not re.match(r"^\s*[A-Z]\.\s+.*$", h, re.IGNORECASE)
                    ]
                    current_heading_parts.append(line)
                    current_metadata = page.metadata.copy()
                elif re.match(r"^\s*Chapter\s+\d+\b.*$", line, re.IGNORECASE):
                    flush_section()
                    current_heading_parts = [
                        h for h in current_heading_parts
                        if not re.match(r"^\s*Chapter\s+\d+\b.*$", h, re.IGNORECASE)
                        and not re.match(r"^\s*[A-Z]\.\s+.*$", h, re.IGNORECASE)
                    ]
                    current_heading_parts.append(line)
                    current_metadata = page.metadata.copy()
                else:
                    flush_section()
                    current_heading_parts = [
                        h for h in current_heading_parts
                        if not re.match(r"^\s*[A-Z]\.\s+.*$", h, re.IGNORECASE)
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