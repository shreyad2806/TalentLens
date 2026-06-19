"""
Validator module - Validation for BM25 documents and index.

This module provides the BM25Validator class that validates BM25Document objects
and BM25Index objects to ensure data integrity and correctness.
"""

from typing import List, Dict, Any
import logging

from .schema import BM25Document
from .bm25_index import BM25Index

logger = logging.getLogger(__name__)


class BM25Validator:
    """
    Validator for BM25 documents and index.
    
    This class provides validation methods for BM25Document objects and
    BM25Index objects to ensure data integrity and correctness.
    """
    
    def validate_document(self, document: BM25Document) -> Dict[str, Any]:
        """
        Validate a single BM25Document.
        
        Args:
            document: BM25Document to validate
            
        Returns:
            Dictionary with 'valid' boolean and 'errors' list
        """
        errors = []
        
        # Validate document_id
        if not document.document_id or not document.document_id.strip():
            errors.append("Document ID is empty")
        
        # Validate chunk_id
        if not document.chunk_id or not document.chunk_id.strip():
            errors.append("Chunk ID is empty")
        
        # Validate resume_id
        if not document.resume_id or not document.resume_id.strip():
            errors.append("Resume ID is empty")
        
        # Validate candidate_name
        if not document.candidate_name or not document.candidate_name.strip():
            errors.append("Candidate name is empty")
        
        # Validate section
        if not document.section or not document.section.strip():
            errors.append("Section is empty")
        
        # Validate text
        if not document.text or not document.text.strip():
            errors.append("Text is empty")
        
        # Validate metadata
        if document.metadata is None:
            errors.append("Metadata is None")
        
        # Validate token_count
        if document.token_count < 0:
            errors.append("Token count is negative")
        
        # Validate created_at
        if not document.created_at or not document.created_at.strip():
            errors.append("Created at timestamp is empty")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def validate_documents(self, documents: List[BM25Document]) -> Dict[str, Any]:
        """
        Validate a list of BM25Document objects.
        
        Args:
            documents: List of BM25Document objects to validate
            
        Returns:
            Dictionary with 'valid' boolean, 'errors' list, and 'stats' dict
        """
        all_errors = []
        duplicate_document_ids = set()
        document_ids = set()
        empty_text_count = 0
        missing_metadata_count = 0
        
        for document in documents:
            # Check for duplicate document_ids
            if document.document_id in document_ids:
                duplicate_document_ids.add(document.document_id)
            document_ids.add(document.document_id)
            
            # Validate individual document
            result = self.validate_document(document)
            if not result['valid']:
                all_errors.extend([f"{document.document_id}: {error}" for error in result['errors']])
            
            # Check for empty text
            if not document.text or not document.text.strip():
                empty_text_count += 1
            
            # Check for missing metadata
            if document.metadata is None:
                missing_metadata_count += 1
        
        # Add duplicate errors
        if duplicate_document_ids:
            all_errors.extend([f"Duplicate document ID: {doc_id}" for doc_id in duplicate_document_ids])
        
        stats = {
            'total_documents': len(documents),
            'valid_documents': len(documents) - len(all_errors),
            'duplicate_document_ids': len(duplicate_document_ids),
            'empty_text_count': empty_text_count,
            'missing_metadata_count': missing_metadata_count
        }
        
        return {
            'valid': len(all_errors) == 0,
            'errors': all_errors,
            'stats': stats
        }
    
    def validate_chunks(self, chunks: List) -> Dict[str, Any]:
        """
        Validate a list of Chunk objects before indexing.
        
        Args:
            chunks: List of Chunk objects to validate
            
        Returns:
            Dictionary with 'valid' boolean and 'errors' list
        """
        errors = []
        
        for chunk in chunks:
            # Validate chunk_id
            if not chunk.chunk_id or not str(chunk.chunk_id).strip():
                errors.append(f"Chunk {chunk}: Chunk ID is empty")
            
            # Validate resume_id
            if not chunk.resume_id or not str(chunk.resume_id).strip():
                errors.append(f"Chunk {chunk.chunk_id}: Resume ID is empty")
            
            # Validate candidate_name
            if not chunk.candidate_name or not chunk.candidate_name.strip():
                errors.append(f"Chunk {chunk.chunk_id}: Candidate name is empty")
            
            # Validate section
            if not chunk.section or not chunk.section.strip():
                errors.append(f"Chunk {chunk.chunk_id}: Section is empty")
            
            # Validate text
            if not chunk.text or not chunk.text.strip():
                errors.append(f"Chunk {chunk.chunk_id}: Text is empty")
            
            # Validate metadata
            if chunk.metadata is None:
                errors.append(f"Chunk {chunk.chunk_id}: Metadata is None")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def validate_index(self, index: BM25Index) -> Dict[str, Any]:
        """
        Validate a BM25Index.
        
        Args:
            index: BM25Index to validate
            
        Returns:
            Dictionary with 'valid' boolean, 'errors' list, and 'stats' dict
        """
        errors = []
        
        # Check if index is empty
        if index.num_documents == 0:
            errors.append("Index is empty")
        
        # Check vocabulary size
        if index.num_documents > 0 and len(index.vocabulary) == 0:
            errors.append("Vocabulary is empty despite having documents")
        
        # Check average document length
        if index.avg_doc_length <= 0:
            errors.append("Average document length is invalid")
        
        # Check document lengths
        invalid_lengths = []
        for doc_id, length in index.document_lengths.items():
            if length <= 0:
                invalid_lengths.append(doc_id)
        
        if invalid_lengths:
            errors.extend([f"Invalid document length for: {doc_id}" for doc_id in invalid_lengths])
        
        # Check inverted index consistency
        total_indexed_docs = set()
        for term, doc_freq in index.inverted_index.items():
            for doc_id in doc_freq.keys():
                total_indexed_docs.add(doc_id)
        
        if len(total_indexed_docs) != index.num_documents:
            errors.append(f"Inverted index inconsistency: {len(total_indexed_docs)} docs in index vs {index.num_documents} expected")
        
        stats = index.get_statistics()
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'stats': stats
        }
    
    def validate_tokenization(self, text: str, tokens: List[str]) -> Dict[str, Any]:
        """
        Validate tokenization results.
        
        Args:
            text: Original text
            tokens: Tokenized result
            
        Returns:
            Dictionary with 'valid' boolean and 'errors' list
        """
        errors = []
        
        # Check if tokens is empty
        if not tokens:
            errors.append("Tokenization resulted in empty token list")
        
        # Check if all tokens are strings
        non_string_tokens = [t for t in tokens if not isinstance(t, str)]
        if non_string_tokens:
            errors.append(f"Non-string tokens found: {non_string_tokens}")
        
        # Check for empty tokens
        empty_tokens = [t for t in tokens if not t or not t.strip()]
        if empty_tokens:
            errors.append(f"Empty tokens found: {len(empty_tokens)}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
