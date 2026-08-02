"""Backend regression and unit tests for TalentLens.

This module provides focused unit and integration tests for the
production backend components and the requested regression queries.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("VECTOR_STORE_PROVIDER", "memory")

from src.resume_parser.normalizer import MetadataNormalizer
from src.embeddings.embedding_service import EmbeddingService
from src.retrieval.sparse.bm25_index import BM25Index
from src.chunks.schema import Chunk, ChunkMetadata
from src.models import ResumeDocument
from src.models.resume_metadata import ResumeMetadata
from src.search.search_service import SearchService
from src.search.schema import SearchFilters
from src.retrieval.hybrid.schema import HybridSearchResult, MatchedChunk, RetrievalSource
from src.retrieval.sparse.schema import BM25Document
from src.bootstrap.bootstrap_status import BootstrapStatus


@pytest.fixture
def sample_resume_doc():
    """Build a minimal but realistic ResumeDocument for ranking tests."""
    return ResumeMetadata(
        resume_id="123",
        candidate_name="Alice",
        role="Backend Engineer",
        skills=["python", "django", "aws", "postgres"],
        location="Remote / United States",
        experience_years=3.5,
        education=["B.S. Computer Science"],
        projects=["payment api", "internal tools"],
        certifications=["AWS Certified Developer"],
    )


def test_metadata_normalizer():
    """Unit: skill / location / experience normalization."""
    norm = MetadataNormalizer()
    assert norm.normalize_skill("java") == "Java"
    assert norm.normalize_skill("JAVA") == "Java"
    assert norm.normalize_location("US") == "United States"
    assert norm.normalize_experience_years("5 years") == 5.0
    assert norm.normalize_experience_years("60 months") == 5.0


def test_embedding_service_vector():
    """Unit: EmbeddingService can vectorize a chunk."""
    md = ResumeMetadata(resume_id="t1", candidate_name="Test")
    chunk = Chunk(
        chunk_id="c1",
        resume_id="t1",
        text="python django backend engineer",
        section="skills",
        candidate_name="Test",
        metadata=ChunkMetadata(),
        resume_metadata=md,
        chunk_order=0,
    )
    service = EmbeddingService(expected_dimension=384)
    records = service.embed_chunks([chunk])
    assert len(records) == 1
    assert len(records[0].vector) == 384


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer for test data."""
    return [t.lower().strip(",./;:!?") for t in text.split() if t.strip()]


def _bm25_doc(text: str, idx: int) -> BM25Document:
    tokens = _tokenize(text)
    md = ResumeMetadata(resume_id=f"d{idx}", candidate_name=f"Candidate {idx}")
    return BM25Document(
        document_id=f"d{idx}",
        chunk_id=f"c{idx}",
        section="summary",
        text=text,
        tokens=tokens,
        document_length=len(tokens),
        resume_metadata=md,
    )


def test_bm25_index_search():
    """Unit: small BM25 index can be searched."""
    docs = [
        "python django backend engineer",
        "java spring finance analyst",
        "react frontend developer remote",
    ]
    index = BM25Index()
    index.add_documents([_bm25_doc(t, i) for i, t in enumerate(docs)])
    results = index.search(["python"], top_k=2)
    assert len(results) > 0
    assert all(r["score"] >= 0 for r in results)


def test_search_filters_validation():
    """Unit: SearchFilters accepts valid data."""
    filters = SearchFilters(skills=["python", "aws"], experience_min=2.0)
    assert filters.skills == ["python", "aws"]
    assert filters.experience_min == 2.0


def _resume_from_meta(md: ResumeMetadata) -> ResumeDocument:
    from src.models import ResumeDocument

    return ResumeDocument(
        candidate_id=md.resume_id,
        resume_id=md.resume_id,
        resume_text=" ".join(md.skills + [md.role or "", md.location or ""]),
        resume_metadata=md,
        source_dataset="test",
    )


def test_search_service_score_resume(sample_resume_doc):
    """Unit: SearchService._score_resume produces a SearchResult."""
    resume = _resume_from_meta(sample_resume_doc)
    service = SearchService(hybrid_service=None)
    result = service._score_resume(
        resume,
        hybrid=None,
        filters=SearchFilters(skills=["python"]),
        query="python backend remote",
    )
    assert result is not None
    assert "python" in result.matched_skills
    assert result.metadata_score > 0


def test_hybrid_search_result_schema():
    """Unit: HybridSearchResult schema and score breakdown."""
    md = ResumeMetadata(resume_id="r1", candidate_name="Bob")
    matched = MatchedChunk(
        chunk_id="c1",
        section="skills",
        matched_text="python developer",
        score=0.9,
        retrieval_source=RetrievalSource.DENSE,
    )
    result = HybridSearchResult(
        query="python",
        chunk_id="c1",
        section="skills",
        rrf_score=0.02,
        resume_metadata=md,
        matched_chunks=[matched],
    )
    assert result.rrf_score == pytest.approx(0.02)
    assert result.matched_chunks[0].retrieval_source == RetrievalSource.DENSE


def test_bootstrap_status_enum():
    """Unit: BootstrapStatus enum covers the required states."""
    assert BootstrapStatus.LOADED_EXISTING_INDEXES.value == "LOADED_EXISTING_INDEXES"
    assert BootstrapStatus.BUILT_NEW_INDEXES.value == "BUILT_NEW_INDEXES"
    assert BootstrapStatus.FAILED.value == "FAILED"


@pytest.mark.parametrize(
    "query",
    [
        "python",
        "java",
        "finance",
        "backend engineer",
        "react developer",
        "aws",
        "machine learning",
        "2 years python remote",
    ],
)
def test_regression_queries_dont_crash(query, sample_resume_doc):
    """Regression: common queries should not crash hybrid search."""
    resume = _resume_from_meta(sample_resume_doc)
    # Mock a tiny hybrid service returning the candidate
    mock_hybrid = MagicMock()
    matched = MatchedChunk(
        chunk_id="c1",
        section="skills",
        matched_text=query,
        score=0.8,
        retrieval_source=RetrievalSource.DENSE,
    )
    mock_hybrid.search.return_value = [
        HybridSearchResult(
            query=query,
            chunk_id="c1",
            section="skills",
            rrf_score=0.1,
            resume_metadata=resume.resume_metadata,
            matched_chunks=[matched],
        )
    ]
    # Resume must exist in the loaded cache for SearchService to include it
    service = SearchService(hybrid_service=mock_hybrid)
    service._resume_cache[resume.resume_metadata.resume_id] = resume
    results = service.search(query, top_k=5)
    assert isinstance(results, list)
