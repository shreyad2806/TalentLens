"""
Batch Processor module for Reranker.

This module provides efficient batch processing for cross-encoder inference,
avoiding the overhead of scoring one query-document pair at a time.

Architecture Notes:
- Batch Processing: Processes multiple pairs in a single inference
- Configurable Batch Size: Default 32, configurable for optimal performance
- Memory Efficient: Handles large candidate lists with batching
- Progress Tracking: Logs batch processing progress

Cross-Encoder Batch Processing:
Cross-encoders can process multiple query-document pairs in a single forward
pass, significantly improving throughput compared to processing pairs individually.
This is especially important for production systems where latency and throughput
are critical.

Batch Processing Benefits:
1. Improved Throughput: Multiple pairs processed in parallel
2. Reduced Latency: Fewer model forward passes
3. Better GPU Utilization: Maximizes GPU parallelism
4. Memory Efficiency: Controlled memory usage with configurable batch size

Batch Size Selection:
- Too small: Poor GPU utilization, high overhead
- Too large: Out of memory errors, diminishing returns
- Optimal: Depends on model size and GPU memory (default 32 is a good starting point)

SOLID Principles Applied:
- Single Responsibility: Only handles batch processing
- Open/Closed: Can be extended with different batching strategies
- Dependency Inversion: Depends on batch processor abstraction
- Interface Segregation: Focused batch processing interface
"""

import logging
import time
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """
    Data class for batch processing results.
    
    This class represents the results of processing a batch of
    query-document pairs through the cross-encoder.
    
    Attributes:
        scores: List of rerank scores for the batch
        batch_size: Number of pairs in the batch
        inference_time: Time taken for inference in seconds
        batch_index: Index of this batch in the overall processing
    """
    
    scores: List[float]
    batch_size: int
    inference_time: float
    batch_index: int


class BatchProcessor:
    """
    Batch processor for cross-encoder inference.
    
    This class handles batching of query-document pairs for efficient
    cross-encoder inference. It splits large candidate lists into
    manageable batches and processes them sequentially.
    
    Architecture Pattern: Batch Processing Pattern
    - Splits large tasks into manageable batches
    - Processes batches sequentially
    - Combines results from all batches
    - Tracks progress and metrics
    
    Batch Processing Strategy:
    1. Split candidates into batches of specified size
    2. For each batch, create query-document pairs
    3. Pass batch through cross-encoder
    4. Collect scores from all batches
    5. Return combined results
    
    Memory Management:
    The batch processor uses a configurable batch size to control memory
    usage. This is important for:
    - Preventing out-of-memory errors on GPUs
    - Handling large candidate lists
    - Optimizing for different hardware configurations
    
    Attributes:
        batch_size: Number of pairs to process per batch (default: 32)
        model: Cross-encoder model instance
        verbose: Whether to log detailed progress (default: False)
        _total_batches: Total number of batches processed
        _total_pairs: Total number of pairs processed
    """
    
    def __init__(self, batch_size: int = 32, verbose: bool = False):
        """
        Initialize the batch processor.
        
        Args:
            batch_size: Number of pairs to process per batch (default: 32)
            verbose: Whether to log detailed progress (default: False)
        """
        self.batch_size = batch_size
        self.verbose = verbose
        self._total_batches = 0
        self._total_pairs = 0
        
        logger.info(
            f"BatchProcessor initialized with batch_size={batch_size}, "
            f"verbose={verbose}"
        )
    
    def create_batches(
        self,
        query: str,
        candidates: List[Any]
    ) -> List[Tuple[str, List[Any]]]:
        """
        Create batches from candidates.
        
        This method splits the candidates into batches of the specified size.
        Each batch is a tuple of (query, candidate_subset).
        
        Args:
            query: The search query
            candidates: List of candidates to batch
            
        Returns:
            List of batches, where each batch is (query, candidate_subset)
        """
        batches = []
        num_candidates = len(candidates)
        
        for i in range(0, num_candidates, self.batch_size):
            batch_candidates = candidates[i:i + self.batch_size]
            batches.append((query, batch_candidates))
        
        logger.debug(
            f"Created {len(batches)} batches from {num_candidates} candidates "
            f"(batch_size={self.batch_size})"
        )
        
        return batches
    
    def create_query_document_pairs(
        self,
        query: str,
        candidates: List[Any]
    ) -> List[Tuple[str, str]]:
        """
        Create query-document pairs for cross-encoder input.
        
        This method creates pairs of (query, document_text) for each candidate.
        The document text is extracted from the candidate's matched chunks.
        
        Args:
            query: The search query
            candidates: List of candidates
            
        Returns:
            List of (query, document_text) pairs
        """
        pairs = []
        
        for candidate in candidates:
            # Extract document text from candidate
            if hasattr(candidate, 'matched_chunks') and candidate.matched_chunks:
                # Use matched chunks if available
                document_text = " ".join(
                    chunk.matched_text for chunk in candidate.matched_chunks
                )
            elif hasattr(candidate, 'matched_text'):
                # Use matched_text if available
                document_text = candidate.matched_text
            else:
                # Fallback to empty string
                document_text = ""
            
            pairs.append((query, document_text))
        
        logger.debug(f"Created {len(pairs)} query-document pairs")
        
        return pairs
    
    def process_batch(
        self,
        model: Any,
        query: str,
        candidates: List[Any],
        batch_index: int = 0
    ) -> BatchResult:
        """
        Process a single batch of candidates.
        
        This method processes a batch of candidates through the cross-encoder
        model and returns the rerank scores.
        
        Args:
            model: Cross-encoder model instance
            query: The search query
            candidates: List of candidates in this batch
            batch_index: Index of this batch (for logging)
            
        Returns:
            BatchResult with scores and metrics
        """
        start_time = time.time()
        
        # Create query-document pairs
        pairs = self.create_query_document_pairs(query, candidates)
        
        # Process through cross-encoder
        try:
            # Cross-encoder expects list of [query, document] pairs
            model_input = [[q, d] for q, d in pairs]
            scores = model.predict(model_input)
            
            # Ensure scores is a list
            if not isinstance(scores, list):
                scores = scores.tolist()
            
            inference_time = time.time() - start_time
            
            if self.verbose:
                logger.info(
                    f"Processed batch {batch_index}: {len(candidates)} candidates "
                    f"in {inference_time:.4f}s"
                )
            
            self._total_batches += 1
            self._total_pairs += len(candidates)
            
            return BatchResult(
                scores=scores,
                batch_size=len(candidates),
                inference_time=inference_time,
                batch_index=batch_index
            )
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_index}: {e}")
            raise
    
    def process_all_batches(
        self,
        model: Any,
        query: str,
        candidates: List[Any]
    ) -> List[float]:
        """
        Process all batches of candidates.
        
        This method splits candidates into batches, processes each batch
        through the cross-encoder, and combines the results.
        
        Args:
            model: Cross-encoder model instance
            query: The search query
            candidates: List of candidates to process
            
        Returns:
            Combined list of rerank scores for all candidates
        """
        if not candidates:
            return []
        
        # Create batches
        batches = self.create_batches(query, candidates)
        
        # Process each batch
        all_scores = []
        total_inference_time = 0.0
        
        for i, (batch_query, batch_candidates) in enumerate(batches):
            batch_result = self.process_batch(
                model,
                batch_query,
                batch_candidates,
                batch_index=i
            )
            all_scores.extend(batch_result.scores)
            total_inference_time += batch_result.inference_time
        
        logger.info(
            f"Processed {len(candidates)} candidates in {len(batches)} batches "
            f"in {total_inference_time:.4f}s "
            f"(avg: {total_inference_time/len(batches):.4f}s per batch)"
        )
        
        return all_scores
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get batch processing statistics.
        
        Returns:
            Dictionary with batch processing statistics
        """
        avg_pairs_per_batch = (
            self._total_pairs / self._total_batches
            if self._total_batches > 0
            else 0
        )
        
        return {
            "total_batches": self._total_batches,
            "total_pairs": self._total_pairs,
            "batch_size": self.batch_size,
            "avg_pairs_per_batch": avg_pairs_per_batch
        }
    
    def reset_stats(self):
        """Reset batch processing statistics."""
        self._total_batches = 0
        self._total_pairs = 0
        logger.debug("Batch processor statistics reset")
    
    def estimate_inference_time(
        self,
        num_candidates: int,
        avg_time_per_batch: float = 0.1
    ) -> float:
        """
        Estimate total inference time for a given number of candidates.
        
        This method provides a rough estimate of inference time based on
        the number of candidates and average time per batch.
        
        Args:
            num_candidates: Number of candidates to process
            avg_time_per_batch: Average time per batch in seconds (default: 0.1)
            
        Returns:
            Estimated total inference time in seconds
        """
        num_batches = (num_candidates + self.batch_size - 1) // self.batch_size
        estimated_time = num_batches * avg_time_per_batch
        
        logger.debug(
            f"Estimated inference time for {num_candidates} candidates: "
            f"{estimated_time:.4f}s ({num_batches} batches)"
        )
        
        return estimated_time
