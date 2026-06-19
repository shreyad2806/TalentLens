"""
Index Builder module - Build BM25 index from Chunk objects.

This module provides the IndexBuilder class that converts Chunk objects into
BM25Document objects and builds a BM25 index with proper tokenization,
normalization, and stop word removal.
"""

import re
import uuid
from typing import List, Optional
import logging

from ...chunks.schema import Chunk
from .schema import BM25Document
from .bm25_index import BM25Index
from .validator import BM25Validator

logger = logging.getLogger(__name__)


class IndexBuilder:
    """
    Builder for creating BM25 index from Chunk objects.
    
    This class handles the conversion of Chunk objects to BM25Document objects,
    tokenization, normalization, stop word removal, and index construction.
    """
    
    # Common English stop words
    STOP_WORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
        'have', 'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now',
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
        'you', 'your', 'yours', 'yourself', 'yourselves', 'him', 'his',
        'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
        'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
        'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
        'do', 'does', 'did', 'doing', 'would', 'should', 'could', 'ought',
        'im', 'youre', 'hes', 'shes', 'its', 'were', 'theyre', 'ive', 'youve',
        'weve', 'theyve', 'id', 'youd', 'hed', 'shed', 'wed', 'theyd',
        'ill', 'youll', 'hell', 'shell', 'well', 'theyll', 'isnt', 'arent',
        'wasnt', 'werent', 'hasnt', 'havent', 'hadnt', 'doesnt', 'dont',
        'didnt', 'wont', 'wouldnt', 'shant', 'shouldnt', 'cant', 'cannot',
        'couldnt', 'mustnt', 'lets', 'thats', 'whos', 'whats', 'heres',
        'theres', 'whens', 'wheres', 'whys', 'hows', 'a', 'an', 'the', 'and',
        'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
        'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
        'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
        'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
        'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very'
    }
    
    def __init__(self, k1: float = 1.5, b: float = 0.75, remove_stop_words: bool = True):
        """
        Initialize the IndexBuilder.
        
        Args:
            k1: BM25 k1 parameter (default: 1.5)
            b: BM25 b parameter (default: 0.75)
            remove_stop_words: Whether to remove stop words (default: True)
        """
        self.k1 = k1
        self.b = b
        self.remove_stop_words = remove_stop_words
        self.validator = BM25Validator()
        
        logger.info(f"IndexBuilder initialized with k1={k1}, b={b}, remove_stop_words={remove_stop_words}")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into individual tokens.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and keep only alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Split on whitespace
        tokens = text.split()
        
        return tokens
    
    def normalize(self, tokens: List[str]) -> List[str]:
        """
        Normalize tokens by removing stop words and applying stemming.
        
        Args:
            tokens: List of tokens to normalize
            
        Returns:
            List of normalized tokens
        """
        normalized = []
        
        for token in tokens:
            # Remove stop words if enabled
            if self.remove_stop_words and token in self.STOP_WORDS:
                continue
            
            # Simple stemming: remove common suffixes
            token = self._stem(token)
            
            # Keep only tokens with length >= 2
            if len(token) >= 2:
                normalized.append(token)
        
        return normalized
    
    def _stem(self, token: str) -> str:
        """
        Apply simple stemming to a token.
        
        This is a basic stemming implementation. For production use,
        consider using a proper stemming library like NLTK or spaCy.
        
        Args:
            token: Token to stem
            
        Returns:
            Stemmed token
        """
        # Remove common suffixes
        suffixes = ['ing', 'ly', 'ed', 'ies', 'es', 's', 'ment', 'ness', 'tion']
        for suffix in suffixes:
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                token = token[:-len(suffix)]
                break
        
        return token
    
    def chunk_to_document(self, chunk: Chunk) -> BM25Document:
        """
        Convert a Chunk object to a BM25Document.
        
        Args:
            chunk: Chunk object to convert
            
        Returns:
            Tuple of (BM25Document, normalized_tokens)
        """
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Tokenize and normalize the text
        tokens = self.tokenize(chunk.text)
        normalized_tokens = self.normalize(tokens)
        
        # Convert metadata to dictionary if it's a Pydantic model
        if hasattr(chunk.metadata, 'model_dump'):
            metadata_dict = chunk.metadata.model_dump()
        elif hasattr(chunk.metadata, 'dict'):
            metadata_dict = chunk.metadata.dict()
        else:
            metadata_dict = chunk.metadata if isinstance(chunk.metadata, dict) else {}
        
        # Create BM25Document
        document = BM25Document(
            document_id=document_id,
            chunk_id=str(chunk.chunk_id),
            resume_id=str(chunk.resume_id),
            candidate_name=chunk.candidate_name,
            section=chunk.section,
            text=chunk.text,
            metadata=metadata_dict,
            token_count=len(normalized_tokens)
        )
        
        return document, normalized_tokens
    
    def build_index(self, chunks: List[Chunk]) -> BM25Index:
        """
        Build a BM25 index from a list of Chunk objects.
        
        Args:
            chunks: List of Chunk objects to index
            
        Returns:
            BM25Index object
        """
        logger.info(f"Building BM25 index from {len(chunks)} chunks...")
        
        # Validate chunks
        validation_result = self.validator.validate_chunks(chunks)
        if not validation_result['valid']:
            logger.warning(f"Chunk validation found issues: {validation_result['errors']}")
        
        # Create BM25 index
        index = BM25Index(k1=self.k1, b=self.b)
        
        # Convert chunks to documents and add to index
        documents_added = 0
        for chunk in chunks:
            try:
                document, normalized_tokens = self.chunk_to_document(chunk)
                index.add_document(
                    document_id=document.document_id,
                    tokens=normalized_tokens,
                    document=document
                )
                documents_added += 1
            except Exception as e:
                logger.error(f"Failed to index chunk {chunk.chunk_id}: {str(e)}")
        
        # Get index statistics
        stats = index.get_statistics()
        
        logger.info(f"BM25 index built successfully:")
        logger.info(f"  Documents indexed: {documents_added}")
        logger.info(f"  Vocabulary size: {stats['vocabulary_size']}")
        logger.info(f"  Average document length: {stats['avg_doc_length']:.2f}")
        logger.info(f"  Total tokens: {stats['total_tokens']}")
        
        return index
    
    def build_index_from_documents(self, documents: List[BM25Document]) -> BM25Index:
        """
        Build a BM25 index from a list of BM25Document objects.
        
        Args:
            documents: List of BM25Document objects to index
            
        Returns:
            BM25Index object
        """
        logger.info(f"Building BM25 index from {len(documents)} documents...")
        
        # Validate documents
        validation_result = self.validator.validate_documents(documents)
        if not validation_result['valid']:
            logger.warning(f"Document validation found issues: {validation_result['errors']}")
        
        # Create BM25 index
        index = BM25Index(k1=self.k1, b=self.b)
        
        # Add documents to index
        documents_added = 0
        for document in documents:
            try:
                # Tokenize and normalize
                tokens = self.tokenize(document.text)
                normalized_tokens = self.normalize(tokens)
                
                index.add_document(
                    document_id=document.document_id,
                    tokens=normalized_tokens,
                    document=document
                )
                documents_added += 1
            except Exception as e:
                logger.error(f"Failed to index document {document.document_id}: {str(e)}")
        
        # Get index statistics
        stats = index.get_statistics()
        
        logger.info(f"BM25 index built successfully:")
        logger.info(f"  Documents indexed: {documents_added}")
        logger.info(f"  Vocabulary size: {stats['vocabulary_size']}")
        logger.info(f"  Average document length: {stats['avg_doc_length']:.2f}")
        logger.info(f"  Total tokens: {stats['total_tokens']}")
        
        return index
