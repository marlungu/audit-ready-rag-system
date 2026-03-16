import logging
import re
from dataclasses import dataclass, field

from langchain_core.documents import Document

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class QualityIssue:
    chunk_index: int
    page_number: int | None
    issue_type: str
    detail: str


@dataclass
class QualityReport:
    total_chunks: int = 0
    passed: int = 0
    failed: int = 0
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        if self.total_chunks == 0:
            return 0.0
        return self.failed / self.total_chunks


def validate_chunk(chunk: Document, index: int) -> list[QualityIssue]:
    """Run all quality checks on a single chunk."""
    issues: list[QualityIssue] = []
    text = chunk.page_content
    meta = chunk.metadata
    page = meta.get("page_number")

    # Length checks
    if len(text.strip()) < settings.min_chunk_length:
        issues.append(
            QualityIssue(
                chunk_index=index,
                page_number=page,
                issue_type="too_short",
                detail=f"Chunk has {len(text.strip())} chars, minimum is {settings.min_chunk_length}.",
            )
        )

    if len(text.strip()) > settings.max_chunk_length:
        issues.append(
            QualityIssue(
                chunk_index=index,
                page_number=page,
                issue_type="too_long",
                detail=f"Chunk has {len(text.strip())} chars, maximum is {settings.max_chunk_length}.",
            )
        )

    # Empty or whitespace-only content
    if not text.strip():
        issues.append(
            QualityIssue(
                chunk_index=index,
                page_number=page,
                issue_type="empty",
                detail="Chunk contains no meaningful text.",
            )
        )
        return issues

    # Repeated character detection (corrupted extraction)
    if re.search(r"(.)\1{20,}", text):
        issues.append(
            QualityIssue(
                chunk_index=index,
                page_number=page,
                issue_type="corrupted",
                detail="Chunk contains long runs of repeated characters.",
            )
        )

    # High ratio of non-alphanumeric characters (garbled PDF extraction)
    alpha_chars = sum(1 for c in text if c.isalnum() or c.isspace())
    if len(text) > 0 and alpha_chars / len(text) < 0.50:
        issues.append(
            QualityIssue(
                chunk_index=index,
                page_number=page,
                issue_type="low_text_ratio",
                detail=f"Only {alpha_chars / len(text):.0%} alphanumeric content.",
            )
        )

    # Heading integrity: chunk should preserve section context
    heading = meta.get("section_heading", "")
    if not heading and len(text.strip()) > 300:
        issues.append(
            QualityIssue(
                chunk_index=index,
                page_number=page,
                issue_type="missing_heading",
                detail="Large chunk has no section heading context.",
            )
        )

    # Metadata completeness
    required_fields = ["document_title", "page_number", "chunk_index"]
    for field_name in required_fields:
        if field_name not in meta or meta[field_name] is None:
            issues.append(
                QualityIssue(
                    chunk_index=index,
                    page_number=page,
                    issue_type="missing_metadata",
                    detail=f"Missing required metadata field: {field_name}",
                )
            )

    return issues


def validate_chunks(chunks: list[Document]) -> QualityReport:
    """Run quality validation across all chunks and return a report."""
    report = QualityReport(total_chunks=len(chunks))

    for i, chunk in enumerate(chunks):
        issues = validate_chunk(chunk, index=i)
        if issues:
            report.failed += 1
            report.issues.extend(issues)
        else:
            report.passed += 1

    logger.info(
        "Quality validation complete: %d/%d passed (%.1f%% failure rate)",
        report.passed,
        report.total_chunks,
        report.failure_rate * 100,
    )

    if report.issues:
        by_type: dict[str, int] = {}
        for issue in report.issues:
            by_type[issue.issue_type] = by_type.get(issue.issue_type, 0) + 1
        for issue_type, count in sorted(by_type.items()):
            logger.warning("  %s: %d occurrence(s)", issue_type, count)

    return report


def filter_valid_chunks(
    chunks: list[Document],
    report: QualityReport,
) -> list[Document]:
    """Remove chunks that failed critical quality checks."""
    critical_types = {"empty", "corrupted", "low_text_ratio"}
    failed_indices = {
        issue.chunk_index
        for issue in report.issues
        if issue.issue_type in critical_types
    }

    valid = [
        chunk for i, chunk in enumerate(chunks) if i not in failed_indices
    ]
    removed = len(chunks) - len(valid)

    if removed > 0:
        logger.info(
            "Filtered out %d critically failed chunks, %d remaining.",
            removed,
            len(valid),
        )

    return valid
