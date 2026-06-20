"""
Dense Retrieval Package.

This package provides production-grade dense semantic retrieval functionality
for resume search applications. It implements a complete retrieval pipeline
with query embedding, vector search, score normalization, candidate aggregation,
and comprehensive caching.

Architecture Notes:
- Facade Pattern for simple API
- SOLID principles throughout
- Comprehensive error handling
- Performance metrics tracking
- Configurable components

Components:
    - DenseRetrievalService: Main service orchestrating the retrieval pipeline
    - QueryEmbedder: Generates embeddings for recruiter queries
    - ScoreNormalizer: Normalizes similarity scores to [0, 1]
    - CandidateAggregator: Aggregates chunks by candidate with weighted scoring
    - RetrievalValidator: Validates retrieval inputs and outputs
    - QueryCache: LRU cache for query results with TTL
    - Schema: Pydantic schemas for data validation

Usage:
    from src.retrieval.dense import DenseRetrievalService, AggregationStrategy
    
    # Create service with weighted aggregation
    service = DenseRetrievalService()
    results = service.search("Python developer with AWS experience", top_k=10)
    
    # Create service with max aggregation
    from src.retrieval.dense import CandidateAggregator, AggregationStrategy
    aggregator = CandidateAggregator(strategy=AggregationStrategy.MAX)
"""

from .schema import DenseSearchResult, AggregatedCandidateResult, RetrievalMetrics
from .validator import RetrievalValidator, ValidationError
from .cache import QueryCache, cached_query
from .score_normalizer import ScoreNormalizer, NormalizationStrategy
from .candidate_aggregator import CandidateAggregator, AggregationStrategy
from .query_embedder import QueryEmbedder
from .dense_retrieval_service import DenseRetrievalService

__all__ = [
    # Main service
    'DenseRetrievalService',
    
    # Schemas
    'DenseSearchResult',
    'AggregatedCandidateResult',
    'RetrievalMetrics',
    
    # Components
    'QueryEmbedder',
    'ScoreNormalizer',
    'CandidateAggregator',
    'RetrievalValidator',
    'QueryCache',
    
    # Enums and strategies
    'AggregationStrategy',
    'NormalizationStrategy',
    
    # Utilities
    'cached_query',
    'ValidationError',
]

__version__ = '1.0.0'
