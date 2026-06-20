"""
BM25 Scorer for Sparse Retrieval Service.

This module implements the BM25 scoring algorithm for ranking documents based on
query term relevance.

Architecture Notes:
- Standard BM25 implementation
- Configurable parameters (k1, b)
- Score explanation logging
- Efficient term frequency calculations

SOLID Principles Applied:
- Single Responsibility: Handles only BM25 scoring
- Open/Closed: Open for new scoring algorithms
- Dependency Inversion: Depends on scoring interface
"""

import logging
import math
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class BM25Scorer:
    """
    BM25 scorer for ranking documents.
    
    This class implements the BM25 (Best Matching 25) ranking function, which is
    widely used in information retrieval to estimate the relevance of documents
    to a given search query.
    
    BM25 Formula:
        score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))
    
    Where:
        - D: Document
        - Q: Query
        - qi: Query term i
        - f(qi, D): Term frequency of qi in document D
        - |D|: Length of document D
        - avgdl: Average document length in the collection
        - k1: Term saturation parameter (default: 1.2)
        - b: Length normalization parameter (default: 0.75)
        - IDF(qi): Inverse document frequency of qi
    
    IDF Calculation:
        IDF(qi) = log((N - df(qi) + 0.5) / (df(qi) + 0.5))
    
    Where:
        - N: Total number of documents
        - df(qi): Document frequency of qi (number of documents containing qi)
    
    Parameters:
        - k1: Controls term frequency saturation. Higher values give more weight to
              term frequency. Typical range: 1.2-2.0
        - b: Controls document length normalization. Higher values give more weight
            to document length. Typical range: 0.5-0.75
    """
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """
        Initialize the BM25 scorer.
        
        Args:
            k1: Term saturation parameter (default: 1.2)
                 - Controls how much term frequency affects the score
                 - Higher values (e.g., 2.0) give more weight to term frequency
                 - Lower values (e.g., 1.0) reduce the impact of term frequency
            b: Length normalization parameter (default: 0.75)
               - Controls how much document length affects the score
               - Higher values (e.g., 1.0) normalize more strongly by length
               - Lower values (e.g., 0.5) reduce length normalization
        """
        self.k1 = k1
        self.b = b
        
        logger.info(f"BM25Scorer initialized with k1={k1}, b={b}")
    
    def calculate_score(
        self,
        query_terms: List[str],
        document_terms: List[str],
        document_length: int,
        average_document_length: float,
        document_frequency: Dict[str, int],
        total_documents: int,
        explain: bool = False
    ) -> float:
        """
        Calculate BM25 score for a document.
        
        This method calculates the BM25 score for a document given a query.
        The score is the sum of individual term scores for each query term.
        
        Args:
            query_terms: List of query terms
            document_terms: List of document terms
            document_length: Length of the document in tokens
            average_document_length: Average document length in the collection
            document_frequency: Dictionary mapping terms to their document frequency
            total_documents: Total number of documents in the collection
            explain: Whether to log score explanation (default: False)
            
        Returns:
            BM25 score for the document
        """
        if not query_terms:
            return 0.0
        
        if not document_terms:
            return 0.0
        
        # Calculate term frequency for document
        term_frequency = self._calculate_term_frequency(document_terms)
        
        # Calculate BM25 score
        score = 0.0
        term_scores = {}
        
        for term in query_terms:
            # Calculate IDF for the term
            idf = self._calculate_idf(term, document_frequency, total_documents)
            
            # Get term frequency in document
            tf = term_frequency.get(term, 0)
            
            if tf == 0:
                continue
            
            # Calculate BM25 component for this term
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * document_length / average_document_length)
            term_score = idf * (numerator / denominator)
            
            # Ensure term score is non-negative
            if term_score < 0:
                term_score = 0.0
            
            score += term_score
            term_scores[term] = term_score
        
        # Ensure final score is non-negative
        if score < 0:
            score = 0.0
        
        if explain:
            self._log_score_explanation(
                query_terms,
                term_frequency,
                document_length,
                average_document_length,
                term_scores,
                score
            )
        
        return score
    
    def _calculate_term_frequency(self, terms: List[str]) -> Dict[str, int]:
        """
        Calculate term frequency for a list of terms.
        
        Args:
            terms: List of terms
            
        Returns:
            Dictionary mapping terms to their frequency
        """
        term_frequency = {}
        for term in terms:
            term_frequency[term] = term_frequency.get(term, 0) + 1
        return term_frequency
    
    def _calculate_idf(
        self,
        term: str,
        document_frequency: Dict[str, int],
        total_documents: int
    ) -> float:
        """
        Calculate inverse document frequency (IDF) for a term.
        
        IDF measures how rare a term is across all documents. Terms that appear
        in fewer documents have higher IDF values, making them more discriminative.
        
        IDF Formula:
            IDF(qi) = log((N - df(qi) + 0.5) / (df(qi) + 0.5))
        
        Args:
            term: Term to calculate IDF for
            document_frequency: Dictionary mapping terms to their document frequency
            total_documents: Total number of documents
            
        Returns:
            IDF value for the term
        """
        df = document_frequency.get(term, 0)
        
        if df == 0:
            return 0.0
        
        # Calculate IDF using the standard formula
        # Add validation to prevent math domain error
        numerator = total_documents - df + 0.5
        denominator = df + 0.5
        
        if numerator <= 0 or denominator <= 0:
            # Edge case: term appears in all documents or more
            # Return 0 to indicate no discriminative power
            return 0.0
        
        idf = math.log(numerator / denominator)
        
        return idf
    
    def _log_score_explanation(
        self,
        query_terms: List[str],
        term_frequency: Dict[str, int],
        document_length: int,
        average_document_length: float,
        term_scores: Dict[str, float],
        total_score: float
    ) -> None:
        """
        Log detailed score explanation for debugging.
        
        Args:
            query_terms: List of query terms
            term_frequency: Term frequency dictionary
            document_length: Document length
            average_document_length: Average document length
            term_scores: Individual term scores
            total_score: Total BM25 score
        """
        logger.debug("BM25 Score Explanation:")
        logger.debug(f"  Query terms: {query_terms}")
        logger.debug(f"  Document length: {document_length}")
        logger.debug(f"  Average document length: {average_document_length:.2f}")
        logger.debug(f"  k1: {self.k1}, b: {self.b}")
        logger.debug("  Term scores:")
        
        for term in query_terms:
            tf = term_frequency.get(term, 0)
            score = term_scores.get(term, 0.0)
            logger.debug(f"    {term}: TF={tf}, Score={score:.4f}")
        
        logger.debug(f"  Total score: {total_score:.4f}")
    
    def calculate_scores(
        self,
        query_terms: List[str],
        documents: List[Dict[str, Any]],
        document_frequency: Dict[str, int],
        total_documents: int,
        average_document_length: float,
        explain: bool = False
    ) -> List[float]:
        """
        Calculate BM25 scores for multiple documents.
        
        Args:
            query_terms: List of query terms
            documents: List of document dictionaries with 'terms' and 'length' keys
            document_frequency: Dictionary mapping terms to their document frequency
            total_documents: Total number of documents
            average_document_length: Average document length
            explain: Whether to log score explanation (default: False)
            
        Returns:
            List of BM25 scores for each document
        """
        scores = []
        
        for doc in documents:
            score = self.calculate_score(
                query_terms=query_terms,
                document_terms=doc['terms'],
                document_length=doc['length'],
                average_document_length=average_document_length,
                document_frequency=document_frequency,
                total_documents=total_documents,
                explain=explain
            )
            scores.append(score)
        
        return scores
    
    def set_parameters(self, k1: Optional[float] = None, b: Optional[float] = None) -> None:
        """
        Update BM25 parameters.
        
        Args:
            k1: New k1 parameter (if provided)
            b: New b parameter (if provided)
        """
        if k1 is not None:
            self.k1 = k1
            logger.info(f"Updated k1 parameter to: {k1}")
        
        if b is not None:
            self.b = b
            logger.info(f"Updated b parameter to: {b}")
    
    def get_parameters(self) -> Dict[str, float]:
        """
        Get current BM25 parameters.
        
        Returns:
            Dictionary with current parameters
        """
        return {
            'k1': self.k1,
            'b': self.b
        }
