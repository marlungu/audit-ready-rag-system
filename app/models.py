from __future__ import annotations

import enum
from datetime import datetime

from pydantic import BaseModel, Field


class ConfidenceLevel(str, enum.Enum):
    HIGH = "high"
    LOW = "low"
    INSUFFICIENT = "insufficient"


class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    timestamp: datetime


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The immigration policy question to answer.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve.",
    )


class SourceReference(BaseModel):
    document_title: str
    page_number: int
    chunk_index: int
    similarity: float
    matched_query: str


class RetrievalMetadata(BaseModel):
    total_chunks_retrieved: int
    chunks_sent_to_model: int
    top_similarity: float | None
    confidence_level: ConfidenceLevel
    expanded_queries: list[str]


class AuditMetadata(BaseModel):
    query_id: str
    model_id: str
    embedding_model_id: str
    temperature: float
    timestamp: datetime


class QueryResponse(BaseModel):
    question: str
    answer: str
    confidence: ConfidenceLevel
    sources: list[SourceReference]
    retrieval: RetrievalMetadata
    audit: AuditMetadata


class EvalResult(BaseModel):
    question: str
    expected_keywords: list[str]
    answer: str
    confidence: ConfidenceLevel
    top_similarity: float | None
    keywords_found: list[str]
    keywords_missing: list[str]
    keyword_recall: float


class EvalSummary(BaseModel):
    total_questions: int
    avg_keyword_recall: float
    avg_top_similarity: float
    confidence_distribution: dict[str, int]
    results: list[EvalResult]


class ChunkQualityReport(BaseModel):
    total_chunks: int
    passed: int
    failed: int
    failure_rate: float
    issues: list[ChunkIssue]


class ChunkIssue(BaseModel):
    chunk_index: int
    page_number: int | None
    issue_type: str
    detail: str


# Rebuild models that have forward references
ChunkQualityReport.model_rebuild()
