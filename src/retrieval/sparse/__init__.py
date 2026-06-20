"""
Sparse Retrieval Package.

This package provides production-grade BM25 sparse retrieval functionality
for resume search applications. It implements a complete retrieval pipeline
with query tokenization, BM25 scoring, ranking, metadata filtering, and
comprehensive caching.

Architecture Notes:
- Facade Pattern for simple API
- SOLID principles throughout
- Comprehensive error handling
- Performance metrics tracking
- Configurable components

Components:
    - SparseRetrievalService: Main service orchestrating the retrieval pipeline
    - Tokenizer: Text tokenization with normalization and stop word removal
    - BM25Scorer: BM25 scoring algorithm with configurable parameters
    - BM25Index: Inverted index structure for efficient retrieval
    - IndexBuilder: Builder for creating BM25 indexes from Chunk objects
    - SparseRetrievalValidator: Validates retrieval inputs and outputs
    - QueryCache: LRU cache for query results with TTL
    - TokenCache: LRU cache for tokenized queries with TTL
    - Schema: Pydantic schemas for data validation

Usage:
    from src.retrieval.sparse import SparseRetrievalService, IndexBuilder, Tokenizer
    
    # Build index from chunks
    builder = IndexBuilder()
    index = builder.build_index(chunks)
    
    # Create service
    service = SparseRetrievalService(index)
    results = service.search("Python developer with AWS experience", top_k=10)
"""

from .schema import SparseSearchResult, BM25Document, BM25IndexStats, RetrievalMetrics
from .validator import SparseRetrievalValidator, ValidationError
from .cache import QueryCache, TokenCache
from .scorer import BM25Scorer
from .bm25_index import BM25Index
from .tokenizer import Tokenizer
from .index_builder import IndexBuilder
from .sparse_retrieval_service import SparseRetrievalService

__all__ = [
    # Main service
    'SparseRetrievalService',
    
    # Schemas
    'SparseSearchResult',
    'BM25Document',
    'BM25IndexStats',
    'RetrievalMetrics',
    
    # Components
    'Tokenizer',
    'BM25Scorer',
    'BM25Index',
    'IndexBuilder',
    'SparseRetrievalValidator',
    'QueryCache',
    'TokenCache',
    
    # Utilities
    'ValidationError',
]

__version__ = '1.0.0'
