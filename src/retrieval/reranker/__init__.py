"""
Reranker package for Cross-Encoder Reranking.

This package provides a production-grade cross-encoder reranking system
for improving search results from hybrid retrieval. It includes components
for model loading, caching, validation, batch processing, scoring, and
the main reranker service.

Architecture Overview:
- ModelLoader: Lazy singleton loading of cross-encoder models
- RerankCache: Thread-safe caching with TTL support
- RerankerValidator: Comprehensive input/output validation
- BatchProcessor: Efficient batch processing for inference
- CrossEncoderScorer: Cross-encoder scoring with normalization
- RerankerService: Main service orchestrating the reranking pipeline

Cross-Encoder Reranking:
Cross-encoders take a query-document pair as input and output a relevance score.
Unlike bi-encoders (which encode queries and documents separately), cross-encoders
jointly process the pair, allowing for more accurate relevance assessment at the
cost of higher computational overhead.

Usage Example:
```python
from src.retrieval.reranker import RerankerService, RerankerModel

# Initialize reranker service
reranker = RerankerService(
    model_name=RerankerModel.MINILM_V2.value,
    cache_enabled=True,
    batch_size=32
)

# Rerank candidates from hybrid retrieval
reranked_results = reranker.rerank(
    query="Python Backend Developer",
    candidates=hybrid_results,
    top_k=10
)

# Access reranked results
for result in reranked_results:
    print(f"{result.candidate_name}: {result.rerank_score:.4f}")
```

Supported Models:
- Default: cross-encoder/ms-marco-MiniLM-L-6-v2 (lightweight, fast)
- Production: BAAI/bge-reranker-v2-m3 (high accuracy, larger)
- Alternative: BAAI/bge-reranker-base, cross-encoder/ms-marco-bert-base-uncased

Package Structure:
- schema: Data models for reranked results and metrics
- model_loader: Lazy singleton model loading with offline support
- cache: Thread-safe caching with TTL and LRU eviction
- validator: Comprehensive validation for inputs and outputs
- batch_processor: Efficient batch processing for inference
- scorer: Cross-encoder scoring with normalization options
- reranker_service: Main service orchestrating the reranking pipeline
"""

from .schema import (
    RerankedResult,
    RerankMetrics,
    RerankEvidence
)
from .model_loader import (
    ModelLoader,
    RerankerModel
)
from .cache import (
    RerankCache,
    CacheEntry
)
from .validator import (
    RerankerValidator,
    ValidationError
)
from .batch_processor import (
    BatchProcessor,
    BatchResult
)
from .scorer import (
    CrossEncoderScorer
)
from .reranker_service import (
    RerankerService
)

__all__ = [
    # Schema
    'RerankedResult',
    'RerankMetrics',
    'RerankEvidence',
    
    # Model Loader
    'ModelLoader',
    'RerankerModel',
    
    # Cache
    'RerankCache',
    'CacheEntry',
    
    # Validator
    'RerankerValidator',
    'ValidationError',
    
    # Batch Processor
    'BatchProcessor',
    'BatchResult',
    
    # Scorer
    'CrossEncoderScorer',
    
    # Service
    'RerankerService',
]

__version__ = '1.0.0'
