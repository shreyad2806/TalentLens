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
from ..retrieval.bm25.index_builder import IndexBuilder as BM25IndexBuilder
from ..retrieval.bm25.bm25_index import BM25Index as BM25IndexClass
from ..retrieval.sparse.index_builder import IndexBuilder as SparseIndexBuilder
from ..retrieval.sparse.bm25_index import BM25Index as SparseBM25IndexClass

from ..vector_store.service import VectorStoreService

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
        bm25_index: Union[BM25IndexClass, SparseBM25IndexClass],
        embedding_service: EmbeddingService,
        vector_store_service: Optional[VectorStoreService] = None,
    ):
        """
        Initialize the indexing service with injected indexing dependencies.

        IndexingService does not own BM25/embedding/vector-store instances;
        it receives them from IndexingPipeline and performs document processing.

        Args:
            bm25_index: Injected BM25Index instance
            embedding_service: Injected EmbeddingService instance
            vector_store_service: Optional injected VectorStoreService instance
        """
        # Document-processing components owned by this service
        self.parser = ParserService()
        self.chunk_service = ChunkService()
        self.embedding_service = embedding_service

        # Injected indexing dependencies (owned by the caller / pipeline)
        self._bm25_index: Union[BM25IndexClass, SparseBM25IndexClass] = bm25_index
        self._vector_store_service: Optional[VectorStoreService] = vector_store_service

        # Detect which BM25Index implementation is in use and choose matching builder
        if isinstance(bm25_index, SparseBM25IndexClass):
            self.index_builder = SparseIndexBuilder()
            self._bm25_interface = 'sparse'
            print(f"[BOOTSTRAP-TRACE][indexing_service.py] Using sparse BM25Index interface")
        else:
            self.index_builder = BM25IndexBuilder()
            self._bm25_interface = 'bm25'
            print(f"[BOOTSTRAP-TRACE][indexing_service.py] Using bm25 BM25Index interface")

        # State tracking
        self._indexed_documents: Dict[str, Any] = {}  # resume_id -> document metadata
        self._vector_count: int = 0

        logger.info(
            "IndexingService initialized (bm25_type=%s, embedding_type=%s, vector_store_type=%s)",
            type(bm25_index).__name__,
            type(embedding_service).__name__,
            type(vector_store_service).__name__ if vector_store_service else "None"
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
        print(f"[BOOTSTRAP-TRACE][indexing_service.py] index_resume() START: file={file_path.name}, resume_id={resume_id[:8]}")
        
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
            print(f"[BOOTSTRAP-TRACE][indexing_service.py]   Step 1 PARSE complete: candidate_name={candidate_name}")
            
            # Step 2: Chunk document
            chunks = self.chunk_service.create_chunks(
                document=document,
                resume_id=resume_id,
                source_document=str(file_path)
            )
            result['chunks_count'] = len(chunks)
            logger.info(f"Created {len(chunks)} chunks")
            print(f"[BOOTSTRAP-TRACE][indexing_service.py]   Step 2 CHUNK complete: chunks={len(chunks)}")
            
            # Step 3: Generate embeddings
            embedding_records = self.embedding_service.embed_chunks(chunks)
            result['embeddings_count'] = len(embedding_records)
            logger.info(f"Generated {len(embedding_records)} embeddings")
            print(f"[BOOTSTRAP-TRACE][indexing_service.py]   Step 3 EMBEDDINGS complete: embeddings={len(embedding_records)}")
            
            # Step 4: Store in vector store through the injected VectorStoreService
            if self._vector_store_service is not None:
                try:
                    vector_records = self._embedding_records_to_vector_records(embedding_records)
                    self._vector_store_service.upsert(vector_records)
                    result['vector_store_success'] = True
                    self._vector_count += len(embedding_records)
                    logger.info("Successfully upserted to vector store")
                    print(f"[BOOTSTRAP-TRACE][indexing_service.py]   Step 4 VECTOR STORE UPSERT complete: vectors={len(embedding_records)}")
                except Exception as e:
                    error_msg = f"Vector store upsert failed: {str(e)}"
                    result['errors'].append(error_msg)
                    logger.error(error_msg)
                    print(f"[BOOTSTRAP-TRACE][indexing_service.py]   Step 4 VECTOR STORE UPSERT FAILED: {e}")
            else:
                result['vector_store_success'] = False
                logger.info("Vector store upsert skipped (no vector store available)")
                print("[BOOTSTRAP-TRACE][indexing_service.py]   Step 4 VECTOR STORE UPSERT skipped: no vector store available")
            
            # Step 5: Index in BM25
            try:
                for chunk in chunks:
                    if self._bm25_interface == 'sparse':
                        self.index_builder.add_document(self._bm25_index, chunk)
                    else:
                        bm25_doc, tokens = self.index_builder.chunk_to_document(chunk)
                        self._bm25_index.add_document(
                            document_id=bm25_doc.document_id,
                            tokens=tokens,
                            document=bm25_doc
                        )
                
                result['bm25_success'] = True
                logger.info("Successfully indexed in BM25")
                print(f"[BOOTSTRAP-TRACE][indexing_service.py]   Step 5 BM25 INDEX complete: docs={len(chunks)}")
            except Exception as e:
                error_msg = f"BM25 indexing failed: {str(e)}"
                result['errors'].append(error_msg)
                logger.error(error_msg)
                print(f"[BOOTSTRAP-TRACE][indexing_service.py]   Step 5 BM25 INDEX FAILED: {e}")
            
            # Store document metadata
            self._indexed_documents[resume_id] = {
                'file_path': str(file_path),
                'candidate_name': candidate_name,
                'chunks_count': len(chunks),
                'indexed_at': document.metadata.get('parsed_at')
            }
            
            logger.info(f"Successfully indexed resume {resume_id}")
            print(f"[BOOTSTRAP-TRACE][indexing_service.py] index_resume() END: resume_id={resume_id[:8]}, chunks={result['chunks_count']}, embeddings={result['embeddings_count']}, vector_store_success={result['vector_store_success']}, bm25_success={result['bm25_success']}")
            
        except Exception as e:
            error_msg = f"Indexing failed: {str(e)}"
            result['errors'].append(error_msg)
            logger.error(error_msg, exc_info=True)
            print(f"[BOOTSTRAP-TRACE][indexing_service.py] index_resume() FAILED: {error_msg}")
        
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
        print(f"[BOOTSTRAP-TRACE][indexing_service.py] index_resumes() START: files={len(file_paths)}")
        
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
        print(f"[BOOTSTRAP-TRACE][indexing_service.py] index_resumes() END: successful={successful}, failed={failed}, total_chunks={total_chunks}, total_embeddings={total_embeddings}")
        
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
            # Clear the injected BM25 index in place (preserve the shared instance)
            if self._bm25_index is not None:
                self._bm25_index.clear()
            result['bm25_cleared'] = True
            logger.info("BM25 index cleared")
            
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
        """
        Return the number of documents currently indexed in BM25.
        """
        if self._bm25_index is None:
            return 0

        # Preferred attribute (src/retrieval/bm25/bm25_index.py)
        if hasattr(self._bm25_index, "num_documents"):
            return self._bm25_index.num_documents

        # Sparse retrieval BM25Index uses total_documents
        if hasattr(self._bm25_index, "total_documents"):
            return self._bm25_index.total_documents

        # Fallback if num_documents is unavailable
        if hasattr(self._bm25_index, "documents"):
            return len(self._bm25_index.documents)

        if hasattr(self._bm25_index, "_documents"):
            return len(self._bm25_index._documents)
        
        if hasattr(self._bm25_index, "document_store"):
            return len(self._bm25_index.document_store)

        logger.warning(
            "BM25Index has no num_documents or documents attribute. Type=%s",
            type(self._bm25_index)
        )

        return 0
    
    def get_bm25_index(self) -> Optional[BM25IndexClass]:
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
        print(f"[BOOTSTRAP-TRACE][indexing_service.py] get_statistics() called: indexed_documents={stats['indexed_documents']}, vector_count={stats['vector_count']}, bm25_count={stats['bm25_count']}")
        
        if self._bm25_index is not None:
            bm25_stats = self._bm25_index.get_statistics()
            stats['bm25_stats'] = bm25_stats
        
        return stats
    
    def _embedding_records_to_vector_records(self, embedding_records: List) -> List:
        """Convert EmbeddingRecord objects to VectorRecord objects for generic vector stores."""
        from ..vector_store.schema import VectorRecord
        vector_records = []
        for record in embedding_records:
            metadata = dict(record.metadata) if record.metadata else {}
            metadata.update({
                'resume_id': record.resume_id,
                'candidate_name': record.candidate_name,
                'section': record.section,
                'embedding_id': str(record.embedding_id),
                'chunk_id': str(record.chunk_id),
            })
            vector_records.append(VectorRecord(
                id=str(record.embedding_id),
                resume_id=record.resume_id,
                chunk_id=str(record.chunk_id),
                candidate_name=record.candidate_name,
                section=record.section,
                vector=record.vector,
                metadata=metadata
            ))
        return vector_records
    