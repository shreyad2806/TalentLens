"""
BM25 Index module - Inverted index with BM25 scoring algorithm.

This module implements the BM25 (Best Matching 25) ranking function for sparse
retrieval. It maintains an inverted index and computes relevance scores for
document-query pairs.
"""

import math
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class BM25Index:
    """
    BM25 index for sparse retrieval.
    
    This class implements an inverted index with BM25 scoring for ranking
    documents based on their relevance to a query.
    
    The BM25 algorithm considers:
    - Term frequency (TF): How often a term appears in a document
    - Inverse document frequency (IDF): How rare a term is across the corpus
    - Document length normalization: Accounts for varying document lengths
    - Free parameters (k1, b): Control the impact of TF and length normalization
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize the BM25 index.
        
        Args:
            k1: Term frequency saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        
        # Inverted index: term -> {document_id: term_frequency}
        self.inverted_index: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Document store: document_id -> BM25Document
        self.documents: Dict[str, 'BM25Document'] = {}
        
        # Document lengths: document_id -> token_count
        self.document_lengths: Dict[str, int] = {}
        
        # Vocabulary: set of all unique terms
        self.vocabulary: Set[str] = set()
        
        # Statistics
        self.num_documents: int = 0
        self.avg_doc_length: float = 0.0
        self.total_tokens: int = 0
        
        logger.info(f"BM25Index initialized with k1={k1}, b={b}")
    
    def add_document(self, document_id: str, tokens: List[str], document: 'BM25Document') -> None:
        """
        Add a document to the index.
        
        Args:
            document_id: Unique identifier for the document
            tokens: List of tokens in the document
            document: BM25Document object containing document metadata
        """
        # Store document
        self.documents[document_id] = document
        
        # Store document length
        token_count = len(tokens)
        self.document_lengths[document_id] = token_count
        self.total_tokens += token_count
        
        # Update inverted index
        for token in tokens:
            self.inverted_index[token][document_id] += 1
            self.vocabulary.add(token)
        
        self.num_documents += 1
        self._update_avg_doc_length()
        
        logger.debug(f"Added document {document_id} with {token_count} tokens")
    
    def _update_avg_doc_length(self) -> None:
        """Update the average document length."""
        if self.num_documents > 0:
            self.avg_doc_length = self.total_tokens / self.num_documents
    
    def get_idf(self, term: str) -> float:
        """
        Calculate the inverse document frequency (IDF) for a term.
        
        IDF measures how rare a term is across the corpus.
        
        Args:
            term: The term to calculate IDF for
            
        Returns:
            IDF score for the term
        """
        if term not in self.inverted_index:
            return 0.0
        
        df = len(self.inverted_index[term])  # Document frequency
        if df == 0:
            return 0.0
        
        # Standard IDF formula: log((N - df + 0.5) / (df + 0.5))
        idf = math.log((self.num_documents - df + 0.5) / (df + 0.5) + 1.0)
        return idf
    
    def get_term_frequency(self, term: str, document_id: str) -> int:
        """
        Get the term frequency for a term in a document.
        
        Args:
            term: The term to look up
            document_id: The document ID
            
        Returns:
            Term frequency (number of occurrences)
        """
        if term not in self.inverted_index:
            return 0
        if document_id not in self.inverted_index[term]:
            return 0
        return self.inverted_index[term][document_id]
    
    def score(self, query_tokens: List[str], document_id: str) -> float:
        """
        Calculate the BM25 score for a document-query pair.
        
        The BM25 score is computed as:
        score = sum(IDF(q) * (TF(q,d) * (k1 + 1)) / (TF(q,d) + k1 * (1 - b + b * |d|/avgdl)))
        
        where:
        - q is a query term
        - d is the document
        - TF is term frequency
        - IDF is inverse document frequency
        - |d| is document length
        - avgdl is average document length
        
        Args:
            query_tokens: List of tokens in the query
            document_id: The document ID to score
            
        Returns:
            BM25 relevance score
        """
        if document_id not in self.documents:
            return 0.0
        
        doc_length = self.document_lengths[document_id]
        score = 0.0
        
        for term in query_tokens:
            tf = self.get_term_frequency(term, document_id)
            if tf == 0:
                continue
            
            idf = self.get_idf(term)
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            term_score = idf * (numerator / denominator)
            
            score += term_score
        
        return score
    
    def search(self, query_tokens: List[str], k: int = 10) -> List[Tuple[str, float]]:
        """
        Search the index for documents matching the query.
        
        Args:
            query_tokens: List of tokens in the query
            k: Number of top results to return
            
        Returns:
            List of (document_id, score) tuples, sorted by score descending
        """
        if not query_tokens:
            return []
        
        # Score all documents
        scores = []
        for document_id in self.documents:
            score = self.score(query_tokens, document_id)
            if score > 0:
                scores.append((document_id, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return scores[:k]
    
    def get_document(self, document_id: str) -> 'BM25Document':
        """
        Retrieve a document by its ID.
        
        Args:
            document_id: The document ID
            
        Returns:
            BM25Document object
        """
        return self.documents.get(document_id)
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get index statistics.
        
        Returns:
            Dictionary with index statistics
        """
        return {
            'num_documents': self.num_documents,
            'vocabulary_size': len(self.vocabulary),
            'avg_doc_length': self.avg_doc_length,
            'total_tokens': self.total_tokens,
            'k1': self.k1,
            'b': self.b
        }
    
    def clear(self) -> None:
        """Clear the index."""
        self.inverted_index.clear()
        self.documents.clear()
        self.document_lengths.clear()
        self.vocabulary.clear()
        self.num_documents = 0
        self.avg_doc_length = 0.0
        self.total_tokens = 0
        logger.info("BM25Index cleared")
    
    def save_to_disk(self, index_path: Path) -> None:
        """
        Save the BM25 index to disk.
        
        This method serializes the index data structures to JSON format
        for persistence across sessions.
        
        Args:
            index_path: Path to save the index (directory)
        """
        index_path = Path(index_path)
        index_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save inverted index
            inverted_index_path = index_path / "inverted_index.json"
            with open(inverted_index_path, 'w', encoding='utf-8') as f:
                json.dump(dict(self.inverted_index), f)
            
            # Save document lengths
            doc_lengths_path = index_path / "document_lengths.json"
            with open(doc_lengths_path, 'w', encoding='utf-8') as f:
                json.dump(self.document_lengths, f)
            
            # Save vocabulary
            vocab_path = index_path / "vocabulary.json"
            with open(vocab_path, 'w', encoding='utf-8') as f:
                json.dump(list(self.vocabulary), f)
            
            # Save documents (convert to dict)
            docs_path = index_path / "documents.json"
            docs_dict = {}
            for doc_id, doc in self.documents.items():
                if hasattr(doc, 'model_dump'):
                    docs_dict[doc_id] = doc.model_dump()
                elif hasattr(doc, 'dict'):
                    docs_dict[doc_id] = doc.dict()
                else:
                    docs_dict[doc_id] = doc
            with open(docs_path, 'w', encoding='utf-8') as f:
                json.dump(docs_dict, f)
            
            # Save metadata
            metadata = {
                'k1': self.k1,
                'b': self.b,
                'num_documents': self.num_documents,
                'avg_doc_length': self.avg_doc_length,
                'total_tokens': self.total_tokens
            }
            metadata_path = index_path / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f)
            
            logger.info(f"BM25Index saved to {index_path}")
            
        except Exception as e:
            logger.error(f"Failed to save BM25Index: {e}")
            raise
    
    def load_from_disk(self, index_path: Path) -> None:
        """
        Load the BM25 index from disk.
        
        This method deserializes the index data structures from JSON format.
        
        Args:
            index_path: Path to load the index from (directory)
        """
        index_path = Path(index_path)
        
        try:
            # Load inverted index
            inverted_index_path = index_path / "inverted_index.json"
            with open(inverted_index_path, 'r', encoding='utf-8') as f:
                inverted_index_data = json.load(f)
                self.inverted_index = defaultdict(lambda: defaultdict(int))
                for term, doc_dict in inverted_index_data.items():
                    for doc_id, count in doc_dict.items():
                        self.inverted_index[term][doc_id] = count
            
            # Load document lengths
            doc_lengths_path = index_path / "document_lengths.json"
            with open(doc_lengths_path, 'r', encoding='utf-8') as f:
                self.document_lengths = json.load(f)
            
            # Load vocabulary
            vocab_path = index_path / "vocabulary.json"
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.vocabulary = set(json.load(f))
            
            # Load documents
            docs_path = index_path / "documents.json"
            with open(docs_path, 'r', encoding='utf-8') as f:
                docs_dict = json.load(f)
                from .schema import BM25Document
                self.documents = {}
                for doc_id, doc_data in docs_dict.items():
                    self.documents[doc_id] = BM25Document(**doc_data)
            
            # Load metadata
            metadata_path = index_path / "metadata.json"
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                self.k1 = metadata['k1']
                self.b = metadata['b']
                self.num_documents = metadata['num_documents']
                self.avg_doc_length = metadata['avg_doc_length']
                self.total_tokens = metadata['total_tokens']
            
            logger.info(f"BM25Index loaded from {index_path}")
            
        except Exception as e:
            logger.error(f"Failed to load BM25Index: {e}")
            raise
