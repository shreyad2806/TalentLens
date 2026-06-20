"""
Index Builder for Sparse Retrieval Service.

This module provides functionality to build BM25 indexes from Chunk objects.
It handles tokenization, document creation, and index construction.

Architecture Notes:
- Converts Chunk objects to BM25Document objects
- Tokenizes document text
- Builds BM25 index with statistics
- Preserves metadata
- Tracks build time

SOLID Principles Applied:
- Single Responsibility: Handles only index building
- Open/Closed: Open for new index building strategies
- Dependency Inversion: Depends on index interface
"""

import logging
import time
from typing import List, Dict, Any, Optional
from src.chunks.schema import Chunk

from .schema import BM25Document, BM25IndexStats
from .bm25_index import BM25Index
from .tokenizer import Tokenizer
from .scorer import BM25Scorer

logger = logging.getLogger(__name__)


class IndexBuilder:
    """
    Builder for BM25 indexes from Chunk objects.
    
    This class provides the functionality to build a BM25 index from a list of
    Chunk objects. Each chunk becomes one BM25 document with its tokens,
    metadata, and statistics preserved.
    
    Building Process:
        1. Validate input chunks
        2. Tokenize each chunk's text
        3. Create BM25Document objects
        4. Add documents to BM25 index
        5. Calculate index statistics
        6. Return built index
    
    Architecture Pattern: Builder Pattern
    - Separates index construction from index usage
    - Provides clean interface for index building
    - Tracks build time and statistics
    """
    
    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        scorer: Optional[BM25Scorer] = None
    ):
        """
        Initialize the index builder.
        
        Args:
            tokenizer: Optional Tokenizer instance (default: new Tokenizer)
            scorer: Optional BM25Scorer instance (default: new BM25Scorer)
        """
        self.tokenizer = tokenizer or Tokenizer()
        self.scorer = scorer or BM25Scorer()
        
        logger.info("IndexBuilder initialized")
    
    def build_index(self, chunks: List[Chunk]) -> BM25Index:
        """
        Build a BM25 index from a list of Chunk objects.
        
        This method converts each Chunk object into a BM25Document and builds
        the BM25 index. The process includes:
        1. Tokenizing chunk text
        2. Creating BM25Document objects with metadata
        3. Adding documents to the index
        4. Calculating index statistics
        
        Args:
            chunks: List of Chunk objects to index
            
        Returns:
            BM25Index with all documents indexed
        """
        if not chunks:
            logger.warning("No chunks provided, returning empty index")
            return BM25Index(scorer=self.scorer)
        
        logger.info(f"Building BM25 index from {len(chunks)} chunks")
        start_time = time.time()
        
        # Create BM25 index
        index = BM25Index(scorer=self.scorer)
        
        # Convert chunks to BM25Document objects
        bm25_documents = []
        for chunk in chunks:
            document = self._chunk_to_document(chunk)
            if document:
                bm25_documents.append(document)
        
        # Add documents to index
        index.add_documents(bm25_documents)
        
        # Calculate build time
        build_time = time.time() - start_time
        
        # Log statistics
        stats = index.get_statistics()
        logger.info(
            f"Index built in {build_time:.2f}s: "
            f"{stats.num_documents} documents, "
            f"{stats.vocabulary_size} vocabulary size, "
            f"{stats.average_document_length:.2f} avg document length"
        )
        
        return index
    
    def _chunk_to_document(self, chunk: Chunk) -> Optional[BM25Document]:
        """
        Convert a Chunk object to a BM25Document.
        
        This method converts a Chunk object into a BM25Document by:
        1. Tokenizing the chunk text
        2. Preserving metadata
        3. Calculating document length
        
        Args:
            chunk: Chunk object to convert
            
        Returns:
            BM25Document if successful, None otherwise
        """
        try:
            # Tokenize the chunk text
            tokens = self.tokenizer.tokenize_document(chunk.text)
            
            # Create BM25Document
            document = BM25Document(
                chunk_id=str(chunk.chunk_id),
                resume_id=str(chunk.resume_id),
                section=chunk.section,
                candidate_name=chunk.candidate_name or "Unknown",
                text=chunk.text,
                tokens=tokens,
                document_length=len(tokens),
                metadata={
                    'text_length': len(chunk.text),
                    'chunk_order': chunk.chunk_order,
                    'embedding_status': chunk.embedding_status.value,
                    'source_document': chunk.source_document
                }
            )
            
            return document
            
        except Exception as e:
            logger.error(f"Failed to convert chunk {chunk.chunk_id} to document: {e}")
            return None
    
    def build_index_incremental(
        self,
        index: BM25Index,
        chunks: List[Chunk]
    ) -> BM25Index:
        """
        Incrementally add chunks to an existing index.
        
        This method adds new chunks to an existing BM25 index without rebuilding
        the entire index. This is useful for incremental updates.
        
        Args:
            index: Existing BM25Index to add to
            chunks: List of Chunk objects to add
            
        Returns:
            Updated BM25Index
        """
        if not chunks:
            logger.warning("No chunks provided, returning unchanged index")
            return index
        
        logger.info(f"Adding {len(chunks)} chunks to existing index")
        start_time = time.time()
        
        # Convert chunks to BM25Document objects
        bm25_documents = []
        for chunk in chunks:
            document = self._chunk_to_document(chunk)
            if document:
                bm25_documents.append(document)
        
        # Add documents to index
        index.add_documents(bm25_documents)
        
        # Calculate build time
        build_time = time.time() - start_time
        
        logger.info(
            f"Added {len(bm25_documents)} documents to index in {build_time:.2f}s"
        )
        
        return index
    
    def add_document(self, index: BM25Index, chunk: Chunk) -> BM25Index:
        """
        Add a single chunk to the index incrementally.
        
        This method converts a Chunk object to BM25Document and adds it to the
        existing index without rebuilding the entire index. This is useful for
        single document updates (e.g., when a new resume is uploaded).
        
        Incremental Add Process:
        1. Convert Chunk to BM25Document
        2. Add document to index using index.add_document()
        3. Index maintains consistency of vocabulary, posting lists, and statistics
        
        Consistency Guarantees:
        - Vocabulary: Updated with new terms from chunk
        - Posting lists: Chunk ID added to each term's posting list
        - Average document length: Recalculated with new document
        - Document count: Incremented by 1
        
        Args:
            index: Existing BM25Index to add to
            chunk: Chunk object to add
            
        Returns:
            Updated BM25Index
        """
        logger.info(f"Adding single chunk {chunk.chunk_id} to index")
        start_time = time.time()
        
        # Convert chunk to BM25Document
        document = self._chunk_to_document(chunk)
        if not document:
            logger.error(f"Failed to convert chunk {chunk.chunk_id} to document")
            return index
        
        # Add document to index
        index.add_document(document)
        
        add_time = time.time() - start_time
        logger.info(f"Chunk {chunk.chunk_id} added to index in {add_time:.3f}s")
        
        return index
    
    def remove_document(self, index: BM25Index, chunk_id: str) -> BM25Index:
        """
        Remove a single chunk from the index incrementally.
        
        This method removes a chunk from the existing index without rebuilding
        the entire index. This is useful for single document deletions (e.g.,
        when a resume is deleted).
        
        Incremental Remove Process:
        1. Remove document from index using index.delete_document()
        2. Index maintains consistency of vocabulary, posting lists, and statistics
        
        Consistency Guarantees:
        - Vocabulary: Terms removed if no documents contain them
        - Posting lists: Chunk ID removed from each term's posting list
        - Average document length: Recalculated without document
        - Document count: Decremented by 1
        
        Args:
            index: Existing BM25Index to remove from
            chunk_id: Chunk ID to remove
            
        Returns:
            Updated BM25Index
        """
        logger.info(f"Removing single chunk {chunk_id} from index")
        start_time = time.time()
        
        # Remove document from index
        success = index.delete_document(chunk_id)
        
        remove_time = time.time() - start_time
        
        if success:
            logger.info(f"Chunk {chunk_id} removed from index in {remove_time:.3f}s")
        else:
            logger.warning(f"Failed to remove chunk {chunk_id} from index")
        
        return index
    
    def update_document(self, index: BM25Index, chunk: Chunk) -> BM25Index:
        """
        Update a single chunk in the index incrementally.
        
        This method updates a chunk in the existing index without rebuilding
        the entire index. This is useful for single document updates (e.g.,
        when a resume is modified).
        
        Incremental Update Process:
        1. Convert Chunk to BM25Document
        2. Update document in index using index.update_document()
        3. Index maintains consistency of vocabulary, posting lists, and statistics
        
        Consistency Guarantees:
        - Vocabulary: Updated with new terms, old orphaned terms removed
        - Posting lists: Chunk ID updated in all affected posting lists
        - Average document length: Recalculated with updated document
        - Document count: Remains the same (document replaced)
        
        Args:
            index: Existing BM25Index to update
            chunk: Chunk object with updated content
            
        Returns:
            Updated BM25Index
        """
        logger.info(f"Updating single chunk {chunk.chunk_id} in index")
        start_time = time.time()
        
        # Convert chunk to BM25Document
        document = self._chunk_to_document(chunk)
        if not document:
            logger.error(f"Failed to convert chunk {chunk.chunk_id} to document")
            return index
        
        # Update document in index
        success = index.update_document(document)
        
        update_time = time.time() - start_time
        
        if success:
            logger.info(f"Chunk {chunk.chunk_id} updated in index in {update_time:.3f}s")
        else:
            logger.warning(f"Failed to update chunk {chunk.chunk_id} in index")
        
        return index
    
    def rebuild_index(self, index: BM25Index, chunks: List[Chunk]) -> BM25Index:
        """
        Rebuild the index from scratch with new chunks.
        
        This method clears the existing index and rebuilds it from the provided
        chunks. This is useful for batch updates or index optimization.
        
        Args:
            index: Existing BM25Index to rebuild
            chunks: List of Chunk objects to rebuild with
            
        Returns:
            Rebuilt BM25Index
        """
        logger.info(f"Rebuilding index with {len(chunks)} chunks")
        start_time = time.time()
        
        # Convert chunks to BM25Document objects
        bm25_documents = []
        for chunk in chunks:
            document = self._chunk_to_document(chunk)
            if document:
                bm25_documents.append(document)
        
        # Rebuild index
        index.rebuild(bm25_documents)
        
        # Calculate build time
        build_time = time.time() - start_time
        
        logger.info(f"Index rebuilt in {build_time:.2f}s")
        
        return index
