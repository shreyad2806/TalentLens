"""
BM25 Retrieval Package - Sparse retrieval using BM25 algorithm.

This package provides a production-ready BM25 sparse retrieval system for
searching through resume chunks. It includes indexing, searching, validation,
and caching components.

Modules:
- schema: BM25Document schema
- bm25_index: BM25 inverted index with scoring
- index_builder: Build BM25 index from Chunk objects
- search_service: Search service with ranking and filtering
- validator: Validation for documents and index
- cache: In-memory cache for search results
"""

from .bm25_index import BM25Index
from .cache import BM25Cache, SearchResult
from .index_builder import IndexBuilder
from .schema import BM25Document
from .search_service import SearchService
from .validator import BM25Validator

__all__ = [
    'BM25Cache',
    'BM25Document',
    'BM25Index',
    'BM25Validator',
    'IndexBuilder',
    'SearchResult',
    'SearchService'
]
