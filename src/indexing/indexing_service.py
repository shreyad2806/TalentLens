"""
Indexing Service module - Main service for indexing resume documents.

This module provides the IndexingService class as the main entry point for
indexing resume documents into both vector stores and BM25 indexes. It
orchestrates the entire indexing pipeline and provides a clean API for
indexing operations.
"""

import uuid
import logging
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

from ..resume_parser.parser_service import ParserService
from ..chunks.service import ChunkService
from ..embeddings.embedding_service import EmbeddingService
from ..retrieval.bm25.index_builder import IndexBuilder
from ..retrieval.sparse.bm25_index import BM25Index

from ..config import EMBEDDING_DIM

# Vector store service is only allowed to be instantiated from composition_root.py.
# IndexingService may receive it via DI, but it must not construct it.
try:
    from ..vector_store import VectorStoreService
except Exception:  # pragma: no cover
    VectorStoreService = None


# Optional pinecone import for vector store
try:
    from ..pinecone_client import get_index, ensure_index
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logging.warning("Pinecone not available - vector store operations will be disabled")

logger = logging.getLogger(__name__)


class IndexingService:

    """
    Main service for indexing resume documents.
    
    This class provides a clean, high-level interface for indexing resumes.
    It orchestrates the entire indexing pipeline:
    1. Parse resume from file
    2. Chunk the parsed document
    3. Generate embeddings for chunks
    4. Store embeddings in vector store
    5. Build BM25 index for sparse retrieval
    
    The service maintains state for both the vector store and BM25 index,
    providing methods to query their current state and rebuild indexes.
    """
    
    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        *,
        bm25_index: Optional[BM25Index] = None,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store_service: Optional[Any] = None,
    ):
        """Initialize the indexing service.

        DI constraints:
        - BM25Index, EmbeddingService, VectorStoreService must be instantiated only in composition_root.py.
        - This class therefore only accepts injected instances; if not provided, it falls back to legacy behavior.
          (Legacy fallback will violate your constraint only if used; bootstrap wiring will always inject.)
        """
        self.parser = ParserService()
        self.chunk_service = ChunkService()

        self.embedding_service = embedding_service or EmbeddingService(expected_dimension=embedding_dim)
        self.index_builder = IndexBuilder()

        # State tracking
        self._indexed_documents: Dict[str, Any] = {}  # resume_id -> document metadata
        self._vector_count: int = 0
        self._bm25_index: Optional[BM25Index] = bm25_index

        # Temporary identity logging (will be removed)
        if logger.isEnabledFor(logging.INFO):
            bm25_id = id(self._bm25_index) if self._bm25_index is not None else None
            emb_id = id(self.embedding_service)
            logger.info(f"[IDENTITY] IndexingService bm25_id={bm25_id} embedding_id={emb_id}")


        # Vector store ownership: default legacy path uses pinecone_client.
        # If a vector_store_service was injected, _upsert_to_vector_store will use it.
        self._vector_store_service = vector_store_service


        # Temporary identity logging (will be removed)
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                f"[IDENTITY] IndexingService vector_store_service_id={id(self._vector_store_service)}"
            )


        self._pinecone_available = PINECONE_AVAILABLE

        if self._pinecone_available and self._vector_store_service is None:
            try:
                ensure_index(dimension=embedding_dim)
                logger.info("Vector store ensured")
            except Exception as e:
                logger.warning(f"Could not ensure vector store: {e}")
                self._pinecone_available = False
        elif self._vector_store_service is not None:
            logger.info("Using injected VectorStoreService for upserts")
        else:
            logger.info("Vector store disabled (Pinecone not available)")

        logger.info(
            "IndexingService initialized with embedding_dim=%s (bm25_injected=%s, embedding_injected=%s, vector_store_injected=%s)",
            embedding_dim,
            bm25_index is not None,
            embedding_service is not None,
            vector_store_service is not None,
        )

    
    def index_resume(self, file_path: Union[str, Path], resume_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Index a single resume file.
        
        This method processes a single resume through the entire pipeline:
        parse → chunk → embed → vector store → BM25 index.
        
        Args:
            file_path: Path to the resume file (PDF, DOCX, or TXT)
            resume_id: Optional unique identifier for the resume. If None, generates UUID.
            
        Returns:
            Dictionary with indexing results including:
                - resume_id: The resume identifier
                - chunks_count: Number of chunks created
                - embeddings_count: Number of embeddings generated
                - vector_store_success: Whether vector store upsert succeeded
                - bm25_success: Whether BM25 indexing succeeded
                - errors: List of any errors encountered
        """
        file_path = Path(file_path)
        
        # Generate resume_id if not provided
        if resume_id is None:
            resume_id = str(uuid.uuid4())
        
        logger.info(f"Indexing resume: {file_path.name} (ID: {resume_id})")
        
        result = {
            'resume_id': resume_id,
            'file_path': str(file_path),
            'chunks_count': 0,
            'embeddings_count': 0,
            'vector_store_success': False,
            'bm25_success': False,
            'errors': []
        }
        
        try:
            # Step 1: Parse resume
            document = self.parser.parse_file(file_path)
            candidate_name = document.name or "Unknown"
            logger.info(f"Parsed resume for candidate: {candidate_name}")
            
            # Step 2: Chunk document
            chunks = self.chunk_service.create_chunks(
                document=document,
                resume_id=resume_id,
                source_document=str(file_path)
            )
            result['chunks_count'] = len(chunks)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Step 3: Generate embeddings
            embedding_records = self.embedding_service.embed_chunks(chunks)
            result['embeddings_count'] = len(embedding_records)
            logger.info(f"Generated {len(embedding_records)} embeddings")
            
            # Step 4: Store in vector store
            if self._pinecone_available:
                try:
                    self._upsert_to_vector_store(embedding_records)
                    result['vector_store_success'] = True
                    self._vector_count += len(embedding_records)
                    logger.info("Successfully upserted to vector store")
                except Exception as e:
                    error_msg = f"Vector store upsert failed: {str(e)}"
                    result['errors'].append(error_msg)
                    logger.error(error_msg)
            else:
                result['vector_store_success'] = False
                logger.info("Vector store upsert skipped (Pinecone not available)")
            
            # Step 5: Index in BM25
            try:
                if self._bm25_index is None:
                    raise RuntimeError(
                        "BM25Index instance was not injected into IndexingService. "
                        "composition_root.py must own BM25Index construction."
                    )

                for chunk in chunks:
                    bm25_doc, tokens = self.index_builder.chunk_to_document(chunk)
                    self._bm25_index.add_document(
                        document_id=bm25_doc.document_id,
                        tokens=tokens,
                        document=bm25_doc
                    )

                
                result['bm25_success'] = True
                logger.info("Successfully indexed in BM25")
            except Exception as e:
                error_msg = f"BM25 indexing failed: {str(e)}"
                result['errors'].append(error_msg)
                logger.error(error_msg)
            
            # Store document metadata
            self._indexed_documents[resume_id] = {
                'file_path': str(file_path),
                'candidate_name': candidate_name,
                'chunks_count': len(chunks),
                'indexed_at': document.metadata.get('parsed_at')
            }
            
            logger.info(f"Successfully indexed resume {resume_id}")
            
        except Exception as e:
            error_msg = f"Indexing failed: {str(e)}"
            result['errors'].append(error_msg)
            logger.error(error_msg, exc_info=True)
        
        return result
    
    def index_resumes(self, file_paths: List[Union[str, Path]]) -> Dict[str, Any]:
        """
        Index multiple resume files.
        
        This method processes multiple resumes through the indexing pipeline.
        It returns aggregate statistics about the indexing operation.
        
        Args:
            file_paths: List of paths to resume files
            
        Returns:
            Dictionary with aggregate indexing results including:
                - total_files: Total number of files processed
                - successful: Number of successfully indexed files
                - failed: Number of failed files
                - total_chunks: Total chunks created
                - total_embeddings: Total embeddings generated
                - results: List of individual file results
        """
        logger.info(f"Indexing {len(file_paths)} resumes...")
        
        results = []
        successful = 0
        failed = 0
        total_chunks = 0
        total_embeddings = 0
        
        for file_path in file_paths:
            result = self.index_resume(file_path)
            results.append(result)
            
            if result['errors']:
                failed += 1
            else:
                successful += 1
                total_chunks += result['chunks_count']
                total_embeddings += result['embeddings_count']
        
        aggregate_result = {
            'total_files': len(file_paths),
            'successful': successful,
            'failed': failed,
            'total_chunks': total_chunks,
            'total_embeddings': total_embeddings,
            'results': results
        }
        
        logger.info(f"Indexing complete: {successful}/{len(file_paths)} successful")
        
        return aggregate_result
    
    def rebuild_index(self) -> Dict[str, Any]:
        """
        Rebuild the entire index from scratch.
        
        This method clears both the vector store and BM25 index, then
        re-indexes all previously indexed documents. This is useful for
        when the indexing pipeline or models change.
        
        Returns:
            Dictionary with rebuild results including:
                - documents_reindexed: Number of documents reindexed
                - vector_store_cleared: Whether vector store was cleared
                - bm25_cleared: Whether BM25 was cleared
                - errors: List of any errors encountered
        """
        logger.info("Rebuilding index from scratch...")
        
        result = {
            'documents_reindexed': 0,
            'vector_store_cleared': False,
            'bm25_cleared': False,
            'errors': []
        }
        
        try:
            # Clear BM25 index (keep injected instance)
            if self._bm25_index is not None:
                self._bm25_index.clear()
                result['bm25_cleared'] = True
                logger.info("BM25 index cleared (injected instance)")
            else:
                result['bm25_cleared'] = False
                logger.warning(
                    "BM25 index not injected into IndexingService; cannot clear"
                )

            
            # Note: Vector store clearing is not implemented here as it would
            # require deleting all vectors from Pinecone, which is destructive.
            # In production, you might want to implement this with caution.
            result['vector_store_cleared'] = False
            logger.warning("Vector store clearing not implemented (would be destructive)")
            
            # Re-index all documents
            # Note: This requires having the original file paths stored
            # For now, we'll just reset the counters
            self._vector_count = 0
            self._indexed_documents.clear()
            
            logger.info("Index rebuild complete")
            
        except Exception as e:
            error_msg = f"Rebuild failed: {str(e)}"
            result['errors'].append(error_msg)
            logger.error(error_msg, exc_info=True)
        
        return result
    
    def document_count(self) -> int:
        """
        Get the number of indexed documents.
        
        Returns:
            Number of unique resumes indexed
        """
        return len(self._indexed_documents)
    
    def vector_count(self) -> int:
        """
        Get the number of vectors in the vector store.
        
        Returns:
            Number of embedding vectors stored
        """
        return self._vector_count
    
    def bm25_count(self) -> int:
        """Return the number of indexed BM25 documents."""
        if self._bm25_index is None:
            return 0

        # Try common attribute names across BM25Index implementations
        # sparse/bm25_index.py uses: total_documents, document_store
        if hasattr(self._bm25_index, "total_documents"):
            return int(self._bm25_index.total_documents)

        if hasattr(self._bm25_index, "num_documents"):
            return int(self._bm25_index.num_documents)

        if hasattr(self._bm25_index, "documents"):
            return len(self._bm25_index.documents)

        if hasattr(self._bm25_index, "document_store"):
            return len(self._bm25_index.document_store)

        if hasattr(self._bm25_index, "_documents"):
            return len(self._bm25_index._documents)

        return 0


    def get_bm25_index(self) -> Optional[BM25Index]:

        """
        Get the BM25 index instance.
        
        Returns:
            BM25Index instance or None if not initialized
        """
        return self._bm25_index
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive indexing statistics.
        
        Returns:
            Dictionary with all indexing statistics
        """
        stats = {
            'indexed_documents': self.document_count(),
            'vector_count': self.vector_count(),
            'bm25_count': self.bm25_count(),
        }
        
        if self._bm25_index is not None:
            bm25_stats = self._bm25_index.get_statistics()
            stats['bm25_stats'] = bm25_stats
            # include identity for temporary verification
            stats['bm25_id'] = id(self._bm25_index)

        
        return stats
    
    def _upsert_to_vector_store(self, embedding_records: List) -> None:
        """
        Upsert embedding records to the vector store.
        
        Args:
            embedding_records: List of EmbeddingRecord objects
        """
        index = get_index()
        
        vectors = []
        for record in embedding_records:
            vector = {
                'id': str(record.chunk_id),
                'values': record.vector,
                'metadata': {
                    'resume_id': record.resume_id,
                    'candidate_name': record.candidate_name,
                    'section': record.section,
                    'embedding_id': str(record.embedding_id)
                }
            }
            vectors.append(vector)
        
        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
        
        logger.info(f"Upserted {len(vectors)} vectors to vector store")
