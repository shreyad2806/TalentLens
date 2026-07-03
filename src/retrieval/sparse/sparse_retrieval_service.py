"""
Sparse Retrieval Service for Production BM25 Search.

This module provides the main SparseRetrievalService that orchestrates
the complete BM25 retrieval pipeline: query tokenization, BM25 scoring,
ranking, metadata filtering, and result formatting.

Architecture Notes:
- Facade Pattern for retrieval pipeline
- Orchestrates multiple components
- Comprehensive logging
- Performance metrics tracking
- Application code only calls this service

SOLID Principles Applied:
- Single Responsibility: Orchestrates retrieval pipeline
- Open/Closed: Open for extension with new components
- Dependency Inversion: Depends on component abstractions
- Interface Segregation: Focused service interface
"""

import logging
import time
from typing import List, Dict, Any, Optional

from .schema import SparseSearchResult, RetrievalMetrics
from .validator import SparseRetrievalValidator, ValidationError
from .cache import QueryCache, TokenCache
from .scorer import BM25Scorer
from .bm25_index import BM25Index
from .tokenizer import Tokenizer


logger = logging.getLogger(__name__)


class SparseRetrievalService:
    """
    Production Sparse Retrieval Service.
    
    This service provides a complete BM25 sparse retrieval pipeline for
    keyword-based search over resume data. It orchestrates query tokenization,
    BM25 scoring, ranking, metadata filtering, and result formatting.
    
    Architecture Pattern: Facade Pattern
    - Simplifies complex retrieval pipeline
    - Orchestrates multiple components
    - Provides single entry point for applications
    - Handles all retrieval complexity internally
    
    Pipeline:
        1. Query validation
        2. Query tokenization (with cache)
        3. BM25 scoring
        4. Ranking
        5. Metadata filtering
        6. Result formatting
        7. Metrics logging
    
    Features:
        - Query tokenization with caching
        - BM25 scoring with configurable parameters
        - Metadata filtering
        - Comprehensive logging
        - Performance metrics
        - Result caching
    """
    
    def __init__(
        self,
        index: BM25Index,
        tokenizer: Optional[Tokenizer] = None,
        scorer: Optional[BM25Scorer] = None,
        cache_enabled: bool = True,
        cache_max_size: int = 1000,
        cache_ttl_seconds: int = 3600
    ):
        """
        Initialize the sparse retrieval service.
        
        Args:
            index: BM25Index to search
            tokenizer: Optional Tokenizer instance (default: new Tokenizer)
            scorer: Optional BM25Scorer instance (default: new BM25Scorer)
            cache_enabled: Whether to enable query caching (default: True)
            cache_max_size: Maximum cache size (default: 1000)
            cache_ttl_seconds: Cache TTL in seconds (default: 3600 = 1 hour)
        """
        # Initialize components
        self.index = index
        self.validator = SparseRetrievalValidator()
        # Dependency injection: never instantiate dependent services here when provided.
        self.tokenizer = tokenizer or Tokenizer()
        self.scorer = scorer or BM25Scorer()

        
        # Initialize caches
        self.cache_enabled = cache_enabled
        if cache_enabled:
            self.query_cache = QueryCache(max_size=cache_max_size, ttl_seconds=cache_ttl_seconds)
            self.token_cache = TokenCache(max_size=cache_max_size, ttl_seconds=cache_ttl_seconds)
        else:
            self.query_cache = None
            self.token_cache = None
        
        # Temporary identity logging for BM25Index
        if logger.isEnabledFor(logging.INFO):
            try:
                logger.info(f"[IDENTITY] SparseRetrievalService bm25_id={id(self.index)}")
            except Exception:
                pass

        logger.info(
            f"SparseRetrievalService initialized with cache_enabled={cache_enabled}, "
            f"index_documents={index.total_documents}"
        )

    
    def search(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[SparseSearchResult]:
        """
        Perform BM25 sparse search.
        
        This is the main entry point for the retrieval service. It performs
        the complete retrieval pipeline and returns formatted results.
        
        Args:
            query: Search query
            top_k: Number of results to return (default: 10)
            filters: Optional metadata filters
            
        Returns:
            List of SparseSearchResult objects
            
        Raises:
            ValidationError: If input validation fails
            RuntimeError: If retrieval pipeline fails
        """
        print(f"------------------------------------")
        print(f"STAGE: SparseRetrievalService.search()")
        print(f"Input: query='{query}', top_k={top_k}, filters={filters}")
        print(f"------------------------------------")
        
        # Validate inputs
        self.validator.validate_query(query)
        self.validator.validate_index(self.index)
        self.validator.validate_top_k(top_k)
        self.validator.validate_filters(filters)
        
        # Check cache
        if self.cache_enabled and self.query_cache:
            cached_results = self.query_cache.get(query, filters, top_k)
            if cached_results is not None:
                print(f"CACHE HIT: Returning {len(cached_results)} cached results")
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cached_results
        
        print(f"Cache miss - proceeding with sparse search")
        
        # Track metrics
        start_time = time.time()
        tokenization_latency = 0.0
        scoring_latency = 0.0
        filtering_latency = 0.0
        
        try:
            # Step 1: Tokenize query
            tokenization_start = time.time()
            if self.cache_enabled and self.token_cache:
                query_tokens = self.token_cache.get(query)
                if query_tokens is None:
                    query_tokens = self.tokenizer.tokenize_query(query)
                    self.token_cache.set(query, query_tokens)
            else:
                query_tokens = self.tokenizer.tokenize_query(query)
            tokenization_latency = time.time() - tokenization_start
            
            print(f"Query tokens: {query_tokens}")
            logger.debug(f"Query tokenization completed in {tokenization_latency:.3f}s")
            
            # Step 2: BM25 scoring
            scoring_start = time.time()
            search_results = self.index.search(query_tokens, top_k * 3)  # Get more for filtering
            scoring_latency = time.time() - scoring_start
            
            print(f"BM25 scoring: {len(search_results)} results")
            if search_results:
                # Handle both dict and object formats
                if isinstance(search_results[0], dict):
                    doc_id = search_results[0].get('document', {}).get('resume_id', 'N/A') if isinstance(search_results[0].get('document'), dict) else getattr(search_results[0].get('document'), 'resume_id', 'N/A')
                    score = search_results[0].get('score', 0)
                else:
                    doc_id = getattr(search_results[0], 'doc_id', 'N/A')
                    score = getattr(search_results[0], 'score', 0)
                print(f"Example BM25 result: doc_id='{doc_id}', score={score}")
            logger.debug(f"BM25 scoring completed in {scoring_latency:.3f}s, returned {len(search_results)} results")
            
            # Step 3: Convert to SparseSearchResult
            filtering_start = time.time()
            search_results = self._convert_to_sparse_results(query, search_results, query_tokens)
            print(f"Converted to SparseSearchResult: {len(search_results)} results")
            
            # Step 4: Apply metadata filters
            if filters:
                before_filter = len(search_results)
                search_results = self._apply_filters(search_results, filters)
                after_filter = len(search_results)
                print(f"Metadata filtering: {before_filter} -> {after_filter} results (removed {before_filter - after_filter})")
                if before_filter > after_filter:
                    print(f"Filter applied: {filters}")
            else:
                print(f"No metadata filters applied")
            
            # Step 5: Sort by BM25 score
            search_results.sort(key=lambda x: x.bm25_score, reverse=True)
            print(f"After sorting: {len(search_results)} results")
            
            # Step 6: Assign ranks
            search_results = [
                SparseSearchResult(
                    query=result.query,
                    candidate_name=result.candidate_name,
                    resume_id=result.resume_id,
                    chunk_id=result.chunk_id,
                    section=result.section,
                    bm25_score=result.bm25_score,
                    metadata=result.metadata,
                    matched_terms=result.matched_terms,
                    matched_text=result.matched_text,
                    rank=i
                )
                for i, result in enumerate(search_results)
            ]
            
            # Step 7: Limit to top_k
            before_limit = len(search_results)
            search_results = search_results[:top_k]
            print(f"Limited to top_k={top_k}: {before_limit} -> {len(search_results)} results")
            
            filtering_latency = time.time() - filtering_start
            
            if search_results:
                print(f"Example: resume_id='{search_results[0].resume_id}', candidate_name='{search_results[0].candidate_name}', bm25_score={search_results[0].bm25_score}")
            
            # Cache results
            if self.cache_enabled and self.query_cache:
                self.query_cache.set(query, search_results, filters, top_k)
            
            # Calculate total latency
            total_latency = time.time() - start_time
            
            # Log metrics
            self._log_metrics(
                query_latency=total_latency,
                tokenization_latency=tokenization_latency,
                scoring_latency=scoring_latency,
                filtering_latency=filtering_latency,
                total_latency=total_latency,
                documents_searched=self.index.total_documents,
                vocabulary_size=len(self.index.vocabulary),
                cache_hit=False
            )
            
            logger.info(
                f"Search completed for query: {query[:50]}... "
                f"returned {len(search_results)} results in {total_latency:.3f}s"
            )
            
            print(f"Output: {len(search_results)} SparseSearchResult objects")
            print(f"Unique candidates: {len(set(r.resume_id for r in search_results))}")
            print(f"------------------------------------")
            
            return search_results
            
        except Exception as e:
            total_latency = time.time() - start_time
            logger.error(f"Search failed for query: {query[:50]}... after {total_latency:.3f}s: {e}")
            raise RuntimeError(f"Search failed: {e}") from e
    
    def _convert_to_sparse_results(
        self,
        query: str,
        search_results: List[Any],
        query_tokens: List[str]
    ) -> List[SparseSearchResult]:
        """
        Convert index search results to SparseSearchResult objects.
        
        Handles both dict format (from BM25Index) and object format (legacy).
        
        Args:
            query: Original search query
            search_results: Results from index search (dicts or objects)
            query_tokens: Tokenized query terms
            
        Returns:
            List of SparseSearchResult objects
        """
        sparse_results = []
        
        for result in search_results:
            # Handle both dict and object formats
            if isinstance(result, dict):
                # Dict format from BM25Index
                document = result.get('document')
                score = result.get('score', 0.0)
                
                # Extract document fields (handle both dict and object)
                if isinstance(document, dict):
                    candidate_name = document.get('candidate_name', 'Unknown')
                    resume_id = document.get('resume_id', '')
                    chunk_id = document.get('chunk_id', '')
                    section = document.get('section', '')
                    metadata = document.get('metadata', {})
                    text = document.get('text', '')
                    tokens = document.get('tokens', [])
                else:
                    # Object format
                    candidate_name = getattr(document, 'candidate_name', 'Unknown')
                    resume_id = getattr(document, 'resume_id', '')
                    chunk_id = getattr(document, 'chunk_id', '')
                    section = getattr(document, 'section', '')
                    metadata = getattr(document, 'metadata', {})
                    text = getattr(document, 'text', '')
                    tokens = getattr(document, 'tokens', [])
            else:
                # Object format (legacy BM25Result)
                document = getattr(result, 'document', None)
                score = getattr(result, 'score', 0.0)
                
                if document:
                    candidate_name = getattr(document, 'candidate_name', 'Unknown')
                    resume_id = getattr(document, 'resume_id', '')
                    chunk_id = getattr(document, 'chunk_id', '')
                    section = getattr(document, 'section', '')
                    metadata = getattr(document, 'metadata', {})
                    text = getattr(document, 'text', '')
                    tokens = getattr(document, 'tokens', [])
                else:
                    # Skip if no document
                    continue
            
            # Find matched terms
            matched_terms = []
            for term in query_tokens:
                if term in tokens:
                    matched_terms.append(term)
            
            # Find matched text (highlight matched terms)
            matched_text = self._find_matched_text(text, matched_terms)
            
            sparse_result = SparseSearchResult(
                query=query,
                candidate_name=candidate_name,
                resume_id=resume_id,
                chunk_id=chunk_id,
                section=section,
                bm25_score=score,
                metadata=metadata,
                matched_terms=matched_terms,
                matched_text=matched_text,
                rank=0  # Will be assigned later
            )
            
            sparse_results.append(sparse_result)
        
        return sparse_results
    
    def _find_matched_text(self, text: str, matched_terms: List[str]) -> str:
        """
        Find text segments that contain matched terms.
        
        Args:
            text: Document text
            matched_terms: List of matched terms
            
        Returns:
            Text segment with matched terms
        """
        if not matched_terms:
            return text[:200]  # Return first 200 chars if no matches
        
        # Find the first occurrence of any matched term
        text_lower = text.lower()
        for term in matched_terms:
            term_lower = term.lower()
            if term_lower in text_lower:
                # Find position and extract context
                pos = text_lower.find(term_lower)
                start = max(0, pos - 50)
                end = min(len(text), pos + len(term) + 50)
                return text[start:end]
        
        return text[:200]  # Return first 200 chars if no matches found
    
    def _apply_filters(self, results: List[SparseSearchResult], filters: Dict[str, Any]) -> List[SparseSearchResult]:
        """
        Apply metadata filters to search results.
        
        Args:
            results: List of search results
            filters: Dictionary of filters to apply
            
        Returns:
            Filtered list of search results
        """
        print(f"------------------------------------")
        print(f"STAGE: _apply_filters")
        print(f"Input: {len(results)} candidates, filters={filters}")
        print(f"------------------------------------")
        
        # Print example candidate before filtering
        if results:
            example = results[0]
            print(f"Example candidate BEFORE filtering:")
            print(f"  resume_id: {example.resume_id}")
            print(f"  candidate_name: {example.candidate_name}")
            print(f"  chunk_id: {example.chunk_id}")
            print(f"  section: {example.section}")
            print(f"  metadata keys: {list(example.metadata.keys())}")
            print(f"  metadata: {example.metadata}")
            print(f"  bm25_score: {example.bm25_score}")
            print(f"------------------------------------")
        
        filtered_results = []
        
        # Rejection counters
        rejection_reasons = {
            'resume_id': 0,
            'candidate_name': 0,
            'section': 0,
            'metadata_mismatch': 0,
            'metadata_missing': 0
        }
        
        for result in results:
            match = True
            rejection_reason = None
            
            for key, value in filters.items():
                if key == 'resume_id':
                    if result.resume_id != value:
                        match = False
                        rejection_reason = f"resume_id mismatch (expected '{value}', got '{result.resume_id}')"
                        rejection_reasons['resume_id'] += 1
                        break
                elif key == 'candidate_name':
                    if value.lower() not in result.candidate_name.lower():
                        match = False
                        rejection_reason = f"candidate_name mismatch (expected '{value}', got '{result.candidate_name}')"
                        rejection_reasons['candidate_name'] += 1
                        break
                elif key == 'section':
                    if value.lower() not in result.section.lower():
                        match = False
                        rejection_reason = f"section mismatch (expected '{value}', got '{result.section}')"
                        rejection_reasons['section'] += 1
                        break
                elif key in result.metadata:
                    if result.metadata[key] != value:
                        match = False
                        rejection_reason = f"metadata '{key}' mismatch (expected '{value}', got '{result.metadata[key]}')"
                        rejection_reasons['metadata_mismatch'] += 1
                        break
                else:
                    # Filter key not in metadata, skip
                    rejection_reason = f"metadata key '{key}' missing from candidate metadata"
                    rejection_reasons['metadata_missing'] += 1
                    # Don't break, continue checking other filters
                    continue
            
            if match:
                filtered_results.append(result)
            else:
                print(f"Rejected candidate:")
                print(f"  resume_id: {result.resume_id}")
                print(f"  candidate_name: {result.candidate_name}")
                print(f"  Reason: {rejection_reason}")
                print(f"  Available metadata: {result.metadata}")
        
        print(f"------------------------------------")
        print(f"Rejection Summary:")
        print(f"  Rejected by resume_id: {rejection_reasons['resume_id']}")
        print(f"  Rejected by candidate_name: {rejection_reasons['candidate_name']}")
        print(f"  Rejected by section: {rejection_reasons['section']}")
        print(f"  Rejected by metadata mismatch: {rejection_reasons['metadata_mismatch']}")
        print(f"  Rejected by metadata missing: {rejection_reasons['metadata_missing']}")
        print(f"  Total rejected: {len(results) - len(filtered_results)}")
        print(f"  Total passed: {len(filtered_results)}")
        print(f"------------------------------------")
        
        return filtered_results
    
    def _log_metrics(
        self,
        query_latency: float,
        tokenization_latency: float,
        scoring_latency: float,
        filtering_latency: float,
        total_latency: float,
        documents_searched: int,
        vocabulary_size: int,
        cache_hit: bool
    ) -> None:
        """
        Log retrieval performance metrics.
        
        Args:
            query_latency: Total query latency
            tokenization_latency: Tokenization latency
            scoring_latency: BM25 scoring latency
            filtering_latency: Filtering latency
            total_latency: Total end-to-end latency
            documents_searched: Number of documents searched
            vocabulary_size: Size of vocabulary
            cache_hit: Whether query was served from cache
        """
        logger.info(
            f"Retrieval metrics: "
            f"query_latency={query_latency:.3f}s, "
            f"tokenization_latency={tokenization_latency:.3f}s, "
            f"scoring_latency={scoring_latency:.3f}s, "
            f"filtering_latency={filtering_latency:.3f}s, "
            f"total_latency={total_latency:.3f}s, "
            f"documents_searched={documents_searched}, "
            f"vocabulary_size={vocabulary_size}, "
            f"cache_hit={cache_hit}"
        )
    
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics dictionary, or None if cache is disabled
        """
        if self.cache_enabled and self.query_cache:
            return {
                'query_cache': self.query_cache.get_stats(),
                'token_cache': self.token_cache.get_stats()
            }
        return None
    
    def clear_cache(self) -> None:
        """Clear the query and token caches."""
        if self.cache_enabled:
            self.query_cache.clear()
            self.token_cache.clear()
            logger.info("Query and token caches cleared")
