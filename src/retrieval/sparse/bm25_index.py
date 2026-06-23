"""
BM25 Index for Sparse Retrieval Service.

This module implements the BM25 index structure for efficient document retrieval.
It maintains vocabulary, document frequency, IDF values, posting lists, and
supports incremental updates and deletions.

Architecture Notes:
- Inverted index structure
- Efficient term lookups
- Support for incremental updates
- Document deletion support
- Index rebuild capability

SOLID Principles Applied:
- Single Responsibility: Handles only index management
- Open/Closed: Open for new index structures
- Dependency Inversion: Depends on index interface
"""

import logging
import time
import json
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict
from pathlib import Path
import math

from .schema import BM25Document, BM25IndexStats
from .scorer import BM25Scorer

logger = logging.getLogger(__name__)


class BM25Index:
    """
    BM25 inverted index for sparse retrieval.
    
    This class maintains the BM25 index structure including:
    - Vocabulary: Set of all unique terms in the collection
    - Document frequency: Number of documents containing each term
    - Inverse document frequency: IDF values for each term
    - Posting lists: For each term, list of document IDs containing it
    - Document store: Mapping from document ID to document data
    - Average document length: Average length across all documents
    
    Index Structure:
        vocabulary: Set[str] - All unique terms
        document_frequency: Dict[str, int] - Term -> document count
        posting_lists: Dict[str, List[str]] - Term -> list of document IDs
        document_store: Dict[str, BM25Document] - Document ID -> document data
        average_document_length: float - Average document length
        total_documents: int - Total number of documents
    """
    
    def __init__(self, scorer: Optional[BM25Scorer] = None):
        """
        Initialize the BM25 index.
        
        Args:
            scorer: Optional BM25Scorer instance for scoring
        """
        self.scorer = scorer or BM25Scorer()
        
        # Index structures
        self.vocabulary: Set[str] = set()
        self.document_frequency: Dict[str, int] = defaultdict(int)
        self.posting_lists: Dict[str, List[str]] = defaultdict(list)
        self.document_store: Dict[str, BM25Document] = {}
        
        # Statistics
        self.total_documents: int = 0
        self.average_document_length: float = 0.0
        self.total_tokens: int = 0
        
        logger.info("BM25Index initialized")
    
    def add_document(self, document: BM25Document) -> None:
        """
        Add a document to the index incrementally.
        
        This method adds a document to the index by:
        1. Checking for duplicate document IDs
        2. Storing the document in the document store
        3. Updating vocabulary with document tokens
        4. Updating document frequency for each term
        5. Adding document ID to posting lists
        6. Recalculating average document length
        7. Logging the update operation
        
        Consistency Guarantees:
        - Vocabulary: Updated with new terms from document
        - Posting lists: Document ID added to each term's posting list
        - Average document length: Recalculated with new document
        - Document count: Incremented by 1
        
        Args:
            document: BM25Document to add to index
        """
        doc_id = document.chunk_id
        
        # Check for duplicate document
        if doc_id in self.document_store:
            logger.warning(f"Document {doc_id} already exists in index, skipping add operation")
            return
        
        logger.info(f"Adding document {doc_id} to index incrementally")
        
        # Store document
        self.document_store[doc_id] = document
        
        # Update vocabulary and posting lists
        new_terms = []
        for term in document.tokens:
            # Update vocabulary
            if term not in self.vocabulary:
                self.vocabulary.add(term)
                new_terms.append(term)
            
            # Update document frequency
            self.document_frequency[term] += 1
            
            # Update posting list
            self.posting_lists[term].append(doc_id)
        
        # Update statistics
        self.total_documents += 1
        self.total_tokens += document.document_length
        self._recalculate_average_document_length()
        
        logger.info(
            f"Document {doc_id} added successfully: "
            f"{len(document.tokens)} tokens, {len(new_terms)} new terms, "
            f"total_documents={self.total_documents}, "
            f"vocabulary_size={len(self.vocabulary)}, "
            f"avg_doc_length={self.average_document_length:.2f}"
        )
    
    def add_documents(self, documents: List[BM25Document]) -> None:
        """
        Add multiple documents to the index.
        
        Args:
            documents: List of BM25Documents to add
        """
        for document in documents:
            self.add_document(document)
        
        logger.info(f"Added {len(documents)} documents to index")
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Remove a document from the index incrementally.
        
        This method removes a document from the index by:
        1. Checking if document exists
        2. Removing from document store
        3. Updating document frequency for each term
        4. Removing document ID from posting lists
        5. Removing terms from vocabulary if no documents contain them
        6. Recalculating average document length
        7. Logging the update operation
        
        Consistency Guarantees:
        - Vocabulary: Terms removed if no documents contain them
        - Posting lists: Document ID removed from each term's posting list
        - Average document length: Recalculated without document
        - Document count: Decremented by 1
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if document was deleted, False if not found
        """
        if doc_id not in self.document_store:
            logger.warning(f"Document {doc_id} not found in index, cannot delete")
            return False
        
        logger.info(f"Removing document {doc_id} from index incrementally")
        
        # Get document
        document = self.document_store[doc_id]
        removed_terms = []
        
        # Remove from posting lists and update document frequency
        for term in document.tokens:
            if doc_id in self.posting_lists[term]:
                self.posting_lists[term].remove(doc_id)
                self.document_frequency[term] -= 1
            
            # Remove term from vocabulary if no documents contain it
            if self.document_frequency[term] == 0:
                self.vocabulary.discard(term)
                del self.document_frequency[term]
                del self.posting_lists[term]
                removed_terms.append(term)
        
        # Remove from document store
        del self.document_store[doc_id]
        
        # Update statistics
        self.total_documents -= 1
        self.total_tokens -= document.document_length
        self._recalculate_average_document_length()
        
        logger.info(
            f"Document {doc_id} removed successfully: "
            f"{len(document.tokens)} tokens, {len(removed_terms)} terms removed from vocabulary, "
            f"total_documents={self.total_documents}, "
            f"vocabulary_size={len(self.vocabulary)}, "
            f"avg_doc_length={self.average_document_length:.2f}"
        )
        
        return True
    
    def update_document(self, document: BM25Document) -> bool:
        """
        Update an existing document in the index incrementally.
        
        This method updates a document by:
        1. Checking if document exists
        2. Removing the old document from index structures
        3. Adding the new document to index structures
        4. Recalculating all statistics
        5. Logging the update operation
        
        This is implemented as a delete-then-add operation to ensure consistency
        of all index structures (vocabulary, posting lists, statistics).
        
        Consistency Guarantees:
        - Vocabulary: Updated with new terms, old orphaned terms removed
        - Posting lists: Document ID updated in all affected posting lists
        - Average document length: Recalculated with updated document
        - Document count: Remains the same (document replaced)
        
        Args:
            document: BM25Document with updated content
            
        Returns:
            True if document was updated, False if not found
        """
        doc_id = document.chunk_id
        
        if doc_id not in self.document_store:
            logger.warning(f"Document {doc_id} not found in index, cannot update")
            return False
        
        logger.info(f"Updating document {doc_id} in index incrementally")
        
        # Get old document for comparison
        old_document = self.document_store[doc_id]
        old_tokens = set(old_document.tokens)
        new_tokens = set(document.tokens)
        
        # Calculate changes
        added_terms = new_tokens - old_tokens
        removed_terms = old_tokens - new_tokens
        common_terms = old_tokens & new_tokens
        
        # Remove old document from index structures
        for term in old_document.tokens:
            if doc_id in self.posting_lists[term]:
                self.posting_lists[term].remove(doc_id)
                self.document_frequency[term] -= 1
            
            # Remove term from vocabulary if no documents contain it
            if self.document_frequency[term] == 0:
                self.vocabulary.discard(term)
                del self.document_frequency[term]
                del self.posting_lists[term]
        
        # Remove from document store
        del self.document_store[doc_id]
        
        # Update statistics (remove old document)
        self.total_tokens -= old_document.document_length
        
        # Add new document to index structures
        for term in document.tokens:
            # Update vocabulary
            if term not in self.vocabulary:
                self.vocabulary.add(term)
            
            # Update document frequency
            self.document_frequency[term] += 1
            
            # Update posting list
            self.posting_lists[term].append(doc_id)
        
        # Store new document
        self.document_store[doc_id] = document
        
        # Update statistics (add new document)
        self.total_tokens += document.document_length
        self._recalculate_average_document_length()
        
        logger.info(
            f"Document {doc_id} updated successfully: "
            f"old_tokens={len(old_tokens)}, new_tokens={len(new_tokens)}, "
            f"added_terms={len(added_terms)}, removed_terms={len(removed_terms)}, "
            f"total_documents={self.total_documents}, "
            f"vocabulary_size={len(self.vocabulary)}, "
            f"avg_doc_length={self.average_document_length:.2f}"
        )
        
        return True
    
    def get_document(self, doc_id: str) -> Optional[BM25Document]:
        """
        Get a document from the index.
        
        Args:
            doc_id: Document ID to retrieve
            
        Returns:
            BM25Document if found, None otherwise
        """
        return self.document_store.get(doc_id)
    
    def get_posting_list(self, term: str) -> List[str]:
        """
        Get the posting list for a term.
        
        Args:
            term: Term to get posting list for
            
        Returns:
            List of document IDs containing the term
        """
        return self.posting_lists.get(term, [])
    
    def get_document_frequency(self, term: str) -> int:
        """
        Get the document frequency for a term.
        
        Args:
            term: Term to get document frequency for
            
        Returns:
            Number of documents containing the term
        """
        return self.document_frequency.get(term, 0)
    
    def get_vocabulary(self) -> Set[str]:
        """
        Get the vocabulary of the index.
        
        Returns:
            Set of all unique terms in the index
        """
        return self.vocabulary.copy()
    
    def get_statistics(self) -> BM25IndexStats:
        """
        Get index statistics.
        
        Returns:
            BM25IndexStats with index statistics
        """
        return BM25IndexStats(
            num_documents=self.total_documents,
            vocabulary_size=len(self.vocabulary),
            average_document_length=self.average_document_length,
            total_tokens=self.total_tokens,
            index_build_time=0.0  # Would be set by IndexBuilder
        )
    
    def _recalculate_average_document_length(self) -> None:
        """Recalculate average document length."""
        if self.total_documents > 0:
            self.average_document_length = self.total_tokens / self.total_documents
        else:
            self.average_document_length = 0.0
    
    def rebuild(self, documents: List[BM25Document]) -> None:
        """
        Rebuild the index from scratch.
        
        This method clears the current index and rebuilds it from the provided
        documents. This is useful for batch updates or index optimization.
        
        Rebuild Process:
        1. Clear all index structures (vocabulary, posting lists, document store)
        2. Reset all statistics (document count, total tokens, average length)
        3. Add all documents using incremental add_document method
        4. Log rebuild statistics
        
        Consistency Guarantees:
        - Vocabulary: Completely rebuilt from all documents
        - Posting lists: Completely rebuilt from all documents
        - Average document length: Recalculated from all documents
        - Document count: Set to number of documents provided
        
        Note: This is a destructive operation that clears the entire index.
        Use incremental operations (add_document, update_document, delete_document)
        for single document updates to avoid full rebuilds.
        
        Args:
            documents: List of BM25Documents to rebuild index with
        """
        logger.info(f"Rebuilding index from scratch with {len(documents)} documents")
        start_time = time.time()
        
        # Clear current index
        self.vocabulary.clear()
        self.document_frequency.clear()
        self.posting_lists.clear()
        self.document_store.clear()
        self.total_documents = 0
        self.total_tokens = 0
        self.average_document_length = 0.0
        
        # Add all documents using incremental add_document
        for document in documents:
            self.add_document(document)
        
        rebuild_time = time.time() - start_time
        
        logger.info(
            f"Index rebuilt successfully in {rebuild_time:.2f}s: "
            f"{self.total_documents} documents, "
            f"{len(self.vocabulary)} vocabulary size, "
            f"{self.average_document_length:.2f} avg document length"
        )
    
    def search(
        self,
        query_terms: List[str],
        top_k: int = 10,
        explain: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search the index for documents matching the query terms.
        
        This method performs a BM25 search by:
        1. Finding documents containing query terms
        2. Calculating BM25 scores for each document
        3. Ranking documents by score
        4. Returning top-k results
        
        Args:
            query_terms: List of query terms
            top_k: Number of results to return
            explain: Whether to log score explanation
            
        Returns:
            List of dictionaries with document ID and score
        """
        if not query_terms:
            return []
        
        # Find documents containing query terms
        candidate_doc_ids = set()
        for term in query_terms:
            posting_list = self.get_posting_list(term)
            candidate_doc_ids.update(posting_list)
        
        if not candidate_doc_ids:
            return []
        
        # Calculate scores for candidate documents
        results = []
        for doc_id in candidate_doc_ids:
            document = self.get_document(doc_id)
            if not document:
                continue
            
            score = self.scorer.calculate_score(
                query_terms=query_terms,
                document_terms=document.tokens,
                document_length=document.document_length,
                average_document_length=self.average_document_length,
                document_frequency=self.document_frequency,
                total_documents=self.total_documents,
                explain=explain
            )
            
            results.append({
                'doc_id': doc_id,
                'score': score,
                'document': document
            })
        
        # Sort by score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top-k
        return results[:top_k]
    
    def is_empty(self) -> bool:
        """
        Check if the index is empty.
        
        Returns:
            True if index has no documents, False otherwise
        """
        return self.total_documents == 0
    
    def clear(self) -> None:
        """Clear all documents from the index."""
        self.vocabulary.clear()
        self.document_frequency.clear()
        self.posting_lists.clear()
        self.document_store.clear()
        self.total_documents = 0
        self.total_tokens = 0
        self.average_document_length = 0.0
        
        logger.info("Index cleared")
    
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
            # Save vocabulary
            vocab_path = index_path / "vocabulary.json"
            with open(vocab_path, 'w', encoding='utf-8') as f:
                json.dump(list(self.vocabulary), f)
            
            # Save document frequency
            doc_freq_path = index_path / "document_frequency.json"
            with open(doc_freq_path, 'w', encoding='utf-8') as f:
                json.dump(dict(self.document_frequency), f)
            
            # Save posting lists
            posting_lists_path = index_path / "posting_lists.json"
            with open(posting_lists_path, 'w', encoding='utf-8') as f:
                json.dump(dict(self.posting_lists), f)
            
            # Save documents (convert to dict)
            docs_path = index_path / "documents.json"
            docs_dict = {}
            for doc_id, doc in self.document_store.items():
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
                'total_documents': self.total_documents,
                'average_document_length': self.average_document_length,
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
            # Load vocabulary
            vocab_path = index_path / "vocabulary.json"
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.vocabulary = set(json.load(f))
            
            # Load document frequency
            doc_freq_path = index_path / "document_frequency.json"
            with open(doc_freq_path, 'r', encoding='utf-8') as f:
                self.document_frequency = defaultdict(int, json.load(f))
            
            # Load posting lists
            posting_lists_path = index_path / "posting_lists.json"
            with open(posting_lists_path, 'r', encoding='utf-8') as f:
                self.posting_lists = defaultdict(list, json.load(f))
            
            # Load documents
            docs_path = index_path / "documents.json"
            with open(docs_path, 'r', encoding='utf-8') as f:
                docs_dict = json.load(f)
                self.document_store = {}
                for doc_id, doc_data in docs_dict.items():
                    self.document_store[doc_id] = BM25Document(**doc_data)
            
            # Load metadata
            metadata_path = index_path / "metadata.json"
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                self.total_documents = metadata['total_documents']
                self.average_document_length = metadata['average_document_length']
                self.total_tokens = metadata['total_tokens']
            
            logger.info(f"BM25Index loaded from {index_path}")
            
        except Exception as e:
            logger.error(f"Failed to load BM25Index: {e}")
            raise
