"""
CSV Resume Ingestion Module for Bootstrap System.

This module handles loading resume records from CSV files for the bootstrap system.
It parses Resume.csv files and converts each row into ResumeDocument schema,
mapping Resume_str to raw_text and candidate fields to metadata.
"""

import csv
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import uuid
from datetime import datetime
import json
import pickle
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CSVIngestionResult:
    """Result of CSV ingestion operation."""
    csv_rows_loaded: int
    chunks_generated: int
    vectors_indexed: int
    bm25_documents_indexed: int
    errors: List[str]
    load_time_seconds: float


class CSVIngestionService:
    """
    Service for ingesting resume records from CSV files.
    
    This class handles parsing Resume.csv files and converting each row
    into ResumeDocument schema for processing through the indexing pipeline.
    
    Mapping:
    - Resume_str → raw_text
    - Candidate fields → metadata
    """
    
    def __init__(self):
        """Initialize the CSV ingestion service."""
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("CSVIngestionService initialized")
    
    def _save_cache(self, chunks, embeddings, bm25_index, indexed_documents):
        """Save cache to disk."""
        try:
            # Save embeddings
            embeddings_array = np.array([e.vector for e in embeddings])
            np.save(self.cache_dir / "embeddings.npy", embeddings_array)
            
            # Save chunks metadata
            chunks_data = []
            for chunk in chunks:
                chunks_data.append({
                    "chunk_id": chunk.chunk_id,
                    "resume_id": chunk.resume_id,
                    "candidate_name": chunk.candidate_name,
                    "section": chunk.section,
                    "text": chunk.text,
                    "metadata": chunk.metadata.dict() if hasattr(chunk.metadata, 'dict') else chunk.metadata,
                    "chunk_order": chunk.chunk_order,
                    "created_at": chunk.created_at.isoformat() if hasattr(chunk.created_at, 'isoformat') else str(chunk.created_at),
                    "embedding_status": chunk.embedding_status.value if hasattr(chunk.embedding_status, 'value') else str(chunk.embedding_status),
                    "source_document": chunk.source_document
                })
            with open(self.cache_dir / "chunks.json", "w", encoding="utf-8") as f:
                json.dump(chunks_data, f, indent=2)
            
            # Save BM25 index to persistent storage
            bm25_index_path = Path("data/indexes/bm25")
            bm25_index.save_to_disk(bm25_index_path)
            logger.info(f"BM25 index saved to {bm25_index_path}")
            
            # Save indexed documents
            with open(self.cache_dir / "indexed_documents.json", "w", encoding="utf-8") as f:
                json.dump(indexed_documents, f, indent=2)
            
            logger.info(f"Cache saved: {len(chunks)} chunks, {len(embeddings)} embeddings")
            print(f"CACHE SAVED: {len(chunks)} chunks, {len(embeddings)} embeddings")
            
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            print(f"WARNING: Failed to save cache: {e}")
    
    def _load_cache(self):
        """Load cache from disk."""
        try:
            # Check if cache files exist
            embeddings_path = self.cache_dir / "embeddings.npy"
            chunks_path = self.cache_dir / "chunks.json"
            indexed_docs_path = self.cache_dir / "indexed_documents.json"
            bm25_index_path = Path("data/indexes/bm25/metadata.json")
            
            if not all([embeddings_path.exists(), chunks_path.exists(), indexed_docs_path.exists(), bm25_index_path.exists()]):
                return None
            
            # Load embeddings
            embeddings_array = np.load(embeddings_path)
            
            # Load chunks metadata
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
            
            # Load indexed documents
            with open(indexed_docs_path, "r", encoding="utf-8") as f:
                indexed_documents = json.load(f)
            
            logger.info(f"Cache loaded: {len(chunks_data)} chunks, {len(embeddings_array)} embeddings")
            print(f"CACHE HIT")
            print(f"Loaded {len(embeddings_array)} vectors")
            
            return {
                "chunks_data": chunks_data,
                "embeddings_array": embeddings_array,
                "indexed_documents": indexed_documents,
                "bm25_index_path": Path("data/indexes/bm25")
            }
            
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return None
    
    def chunk_raw_text(self, raw_text: str, resume_id: str, candidate_name: Optional[str],
                      source_document: str, chunk_size: int = 1000, overlap: int = 100) -> List:
        """
        Chunk raw text into smaller pieces for indexing.
        
        This method splits raw_text into chunks of specified size with overlap,
        creating Chunk objects directly without requiring structured semantic fields.
        
        Args:
            raw_text: The raw resume text to chunk
            resume_id: Unique identifier for the resume
            candidate_name: Candidate name
            source_document: Source document identifier
            chunk_size: Maximum chunk size in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            List of Chunk objects
        """
        from src.chunks.schema import Chunk, ChunkMetadata, EmbeddingStatus
        
        chunks = []
        if not raw_text or len(raw_text) == 0:
            return chunks
        
        # If text is very short, create a single chunk if it has content
        if len(raw_text.strip()) < chunk_size:
            stripped_text = raw_text.strip()
            if stripped_text:
                chunk_metadata = ChunkMetadata(
                    role=None,
                    experience=None,
                    location=None,
                    education=None,
                    source_section="raw_text"
                )
                
                chunk = Chunk(
                    chunk_id=str(uuid.uuid4()),
                    resume_id=resume_id,
                    candidate_name=candidate_name or "Unknown",
                    section="raw_text_chunk_1",
                    text=stripped_text,
                    metadata=chunk_metadata,
                    chunk_order=0,
                    created_at=datetime.now(),
                    embedding_status=EmbeddingStatus.PENDING,
                    source_document=source_document
                )
                
                chunks.append(chunk)
            return chunks
        
        # Split text into chunks
        start = 0
        chunk_order = 0
        
        while start < len(raw_text):
            end = start + chunk_size
            chunk_text = raw_text[start:end]
            
            # Skip empty or whitespace-only chunks
            if not chunk_text or not chunk_text.strip():
                start = end - overlap
                if overlap >= chunk_size:
                    start = end
                continue
            
            # Create chunk metadata
            chunk_metadata = ChunkMetadata(
                role=None,
                experience=None,
                location=None,
                education=None,
                source_section="raw_text"
            )
            
            # Create chunk
            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                resume_id=resume_id,
                candidate_name=candidate_name or "Unknown",
                section=f"raw_text_chunk_{chunk_order + 1}",
                text=chunk_text,
                metadata=chunk_metadata,
                chunk_order=chunk_order,
                created_at=datetime.now(),
                embedding_status=EmbeddingStatus.PENDING,
                source_document=source_document
            )
            
            chunks.append(chunk)
            chunk_order += 1
            
            # Move to next chunk with overlap
            start = end - overlap
            
            # Prevent infinite loop if overlap >= chunk_size
            if overlap >= chunk_size:
                start = end
        
        return chunks
    
    def detect_csv_file(self, directory: Path) -> Optional[Path]:
        """
        Detect Resume.csv in the specified directory.
        
        Args:
            directory: Directory to search for Resume.csv
            
        Returns:
            Path to Resume.csv if found, None otherwise
        """
        csv_path = directory / "Resume.csv"
        
        if csv_path.exists() and csv_path.is_file():
            logger.info(f"Detected Resume.csv at: {csv_path}")
            return csv_path
        
        logger.debug(f"No Resume.csv found in: {directory}")
        return None
    
    def load_csv_records(self, csv_path: Path) -> List[Dict[str, Any]]:
        """
        Load all resume records from CSV file.
        
        Args:
            csv_path: Path to Resume.csv file
            
        Returns:
            List of dictionaries containing resume records
        """
        records = []
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    records.append(row)
            
            logger.info(f"Loaded {len(records)} records from {csv_path}")
            return records
            
        except Exception as e:
            logger.error(f"Failed to load CSV file {csv_path}: {e}")
            raise
    
    def convert_to_resume_document(self, record: Dict[str, Any], record_id: str) -> Dict[str, Any]:
        """
        Convert a CSV record to ResumeDocument schema.
        
        Args:
            record: Dictionary containing CSV row data
            record_id: Unique identifier for the record
            
        Returns:
            Dictionary with ResumeDocument-compatible structure
        """
        # Extract raw text from Resume_str field
        raw_text = record.get('Resume_str', '')
        
        # Extract candidate fields for metadata
        metadata = {}
        candidate_fields = ['Candidate', 'Name', 'Email', 'Phone', 'Location', 
                           'Skills', 'Experience', 'Education', 'Role', 
                           'Salary', 'Notice_Period', 'Category']
        
        for field in candidate_fields:
            if field in record and record[field]:
                metadata[field] = record[field]
        
        # Build document structure compatible with ResumeDocument
        document = {
            'resume_id': record_id,
            'raw_text': raw_text,
            'name': record.get('Candidate') or record.get('Name'),
            'email': record.get('Email'),
            'phone': record.get('Phone'),
            'metadata': metadata
        }
        
        return document
    
    def process_csv_for_indexing(self, csv_path: Path, indexing_service, skip_embedding: bool = False, use_cache: bool = True) -> CSVIngestionResult:
        """
        Process CSV file through the indexing pipeline.
        
        This method:
        1. Checks for cache and loads if available
        2. If no cache, loads all records from CSV
        3. Converts each record to ResumeDocument
        4. Passes each through: Chunking → Embedding → Vector Store → BM25
        5. Saves cache after successful processing
        
        Args:
            csv_path: Path to Resume.csv file
            indexing_service: IndexingService instance for processing
            skip_embedding: If True, skip embedding generation (for testing)
            use_cache: If True, use cache if available
            
        Returns:
            CSVIngestionResult with processing statistics
        """
        import time
        import uuid
        from src.resume_parser.schema import ResumeDocument
        
        start_time = time.time()
        errors = []
        
        total_chunks = 0
        total_vectors = 0
        total_bm25_docs = 0
        
        # Check for cache
        if use_cache and not skip_embedding:
            cache_data = self._load_cache()
            if cache_data:
                # Cache hit - load data into indexing service
                chunks_data = cache_data["chunks_data"]
                embeddings_array = cache_data["embeddings_array"]
                indexed_documents = cache_data["indexed_documents"]
                bm25_index_path = cache_data["bm25_index_path"]
                
                # Restore indexed documents
                indexing_service._indexed_documents = indexed_documents
                
                # Load BM25 index from persistent storage into the injected instance
                # (BM25Index construction is owned by composition_root.py only)
                print("Loading BM25 index from persistent storage...")
                assert indexing_service.get_bm25_index() is not None, (
                    "IndexingService must receive an injected BM25Index instance"
                )
                indexing_service.get_bm25_index().load_from_disk(bm25_index_path)
                print("BM25 index loaded from persistent storage")

                
                # Upsert cached vectors to vector store
                print("Upserting cached vectors to vector store...")
                from src.vector_store.service import VectorStoreService
                from src.vector_store.schema import VectorRecord
                import uuid
                
                # Use a fresh adapter only if the caller didn't inject; CSV ingestion manages its own wiring.
                vector_store_service = VectorStoreService()

                records = []
                
                for chunk_data, embedding_vector in zip(chunks_data, embeddings_array):
                    record = VectorRecord(
                        id=str(uuid.uuid4()),
                        resume_id=chunk_data.get("resume_id", ""),
                        chunk_id=chunk_data.get("chunk_id", ""),
                        candidate_name=chunk_data.get("candidate_name") or "Unknown",
                        section=chunk_data.get("section", ""),
                        vector=embedding_vector.tolist(),
                        metadata={
                            "text": chunk_data.get("text", ""),
                            "chunk_order": chunk_data.get("chunk_order", 0)
                        }
                    )
                    records.append(record)
                
                # Batch upsert
                batch_size = 100
                for i in range(0, len(records), batch_size):
                    batch = records[i:i + batch_size]
                    try:
                        vector_store_service.upsert(batch)
                    except Exception as e:
                        print(f"  Warning: Batch upsert failed: {e}")
                
                # Update vector count in indexing service
                indexing_service._vector_count = len(records)
                print(f"Upserted {len(records)} vectors to vector store")
                
                # Update counts
                total_chunks = len(chunks_data)
                total_vectors = len(embeddings_array)
                total_bm25_docs = len(chunks_data)
                
                load_time = time.time() - start_time
                
                result = CSVIngestionResult(
                    csv_rows_loaded=len(indexed_documents),
                    chunks_generated=total_chunks,
                    vectors_indexed=total_vectors,
                    bm25_documents_indexed=total_bm25_docs,
                    errors=errors,
                    load_time_seconds=load_time
                )
                
                logger.info(
                    f"CSV ingestion complete (from cache): "
                    f"rows={len(indexed_documents)}, "
                    f"chunks={total_chunks}, "
                    f"vectors={total_vectors}, "
                    f"bm25_docs={total_bm25_docs}, "
                    f"time={load_time:.2f}s"
                )
                
                return result
            else:
                print("CACHE MISS")
                print("Building embeddings")
        else:
            print("CACHE DISABLED")
            if skip_embedding:
                print("Skipping embedding generation")
        
        # Timing metrics
        total_parse_time = 0
        total_chunk_time = 0
        total_embedding_time = 0
        total_vector_upsert_time = 0
        total_bm25_time = 0
        
        # Success flags
        first_vector_insertion_logged = False
        first_bm25_insertion_logged = False
        
        # Collect all chunks and embeddings for caching
        all_chunks = []
        all_embeddings = []
        
        # Log vector store provider
        if indexing_service._pinecone_available:
            print("VECTOR STORE PROVIDER = pinecone")
            print("USING PINECONE")
        else:
            print("VECTOR STORE PROVIDER = memory")
            print("USING MEMORY STORE")
        
        try:
            # Step 1: Load CSV records
            records = self.load_csv_records(csv_path)
            csv_rows_loaded = len(records)
            
            logger.info(f"Processing {csv_rows_loaded} CSV records for indexing")
            
            # Step 2: Process each record through the pipeline
            for idx, record in enumerate(records):
                record_id = f"csv-{uuid.uuid4()}"
                
                # Progress logging every 100 resumes
                if (idx + 1) % 100 == 0:
                    print(f"Processing resume {idx + 1}/{csv_rows_loaded}")
                
                try:
                    # Parse timing
                    parse_start = time.time()
                    # Convert to ResumeDocument
                    document_dict = self.convert_to_resume_document(record, record_id)
                    parse_time = time.time() - parse_start
                    total_parse_time += parse_time
                    
                    # Extract raw text and candidate name
                    raw_text = document_dict['raw_text']
                    candidate_name = document_dict.get('name')
                    
                    # Chunk timing
                    chunk_start = time.time()
                    # Step 3: Chunk the raw text directly
                    chunks = self.chunk_raw_text(
                        raw_text=raw_text,
                        resume_id=record_id,
                        candidate_name=candidate_name,
                        source_document=str(csv_path)
                    )
                    chunk_time = time.time() - chunk_start
                    total_chunk_time += chunk_time
                    
                    chunks_count = len(chunks)
                    total_chunks += chunks_count
                    
                    # Collect chunks for caching
                    all_chunks.extend(chunks)
                    
                    # Log if no chunks generated
                    if chunks_count == 0:
                        logger.warning(f"No chunks generated for record {idx + 1}: raw_text_length={len(raw_text)}")
                        if len(errors) == 0:  # Log first failure
                            error_msg = f"First chunking failure at record {idx + 1}: raw_text_length={len(raw_text)}, candidate_name={candidate_name}"
                            errors.append(error_msg)
                            logger.error(error_msg)
                    
                    # Step 4: Generate embeddings (skip if flag is set)
                    if not skip_embedding:
                        embedding_start = time.time()
                        embedding_records = indexing_service.embedding_service.embed_chunks(chunks)
                        embedding_time = time.time() - embedding_start
                        total_embedding_time += embedding_time
                        
                        embeddings_count = len(embedding_records)
                        total_vectors += embeddings_count
                        
                        # Collect embeddings for caching
                        all_embeddings.extend(embedding_records)
                        
                        # Step 5: Store in vector store
                        if indexing_service._pinecone_available:
                            try:
                                vector_upsert_start = time.time()
                                print(f"ENTERING _upsert_to_vector_store() for record {idx + 1}")
                                indexing_service._upsert_to_vector_store(embedding_records)
                                print(f"EXITING _upsert_to_vector_store() for record {idx + 1}")
                                vector_upsert_time = time.time() - vector_upsert_start
                                total_vector_upsert_time += vector_upsert_time
                                
                                indexing_service._vector_count += embeddings_count
                                
                                # Log first successful vector insertion
                                if not first_vector_insertion_logged:
                                    print(f"FIRST SUCCESSFUL VECTOR INSERTION at record {idx + 1}")
                                    first_vector_insertion_logged = True
                            except Exception as e:
                                error_msg = f"Vector store upsert failed for record {record_id}: {str(e)}"
                                errors.append(error_msg)
                                logger.error(error_msg)
                    else:
                        logger.debug(f"Skipping embedding for record {idx + 1}")
                    
                    # Step 6: Index in BM25
                    bm25_start = time.time()
                    try:
                        if indexing_service._bm25_index is None:
                            raise RuntimeError(
                                "IndexingService must receive an injected BM25Index instance"
                            )

                        for chunk in chunks:

                            bm25_doc, tokens = indexing_service.index_builder.chunk_to_document(chunk)
                            indexing_service._bm25_index.add_document(
                                document_id=bm25_doc.document_id,
                                tokens=tokens,
                                document=bm25_doc
                            )
                            total_bm25_docs += 1
                            
                            # Log first successful BM25 insertion
                            if not first_bm25_insertion_logged:
                                print(f"FIRST SUCCESSFUL BM25 INSERTION at record {idx + 1}")
                                first_bm25_insertion_logged = True
                    except Exception as e:
                        error_msg = f"BM25 indexing failed for record {record_id}: {str(e)}"
                        errors.append(error_msg)
                        logger.error(error_msg)
                    bm25_time = time.time() - bm25_start
                    total_bm25_time += bm25_time
                    
                    # Store document metadata
                    indexing_service._indexed_documents[record_id] = {
                        'source': 'csv',
                        'csv_path': str(csv_path),
                        'candidate_name': candidate_name or "Unknown",
                        'chunks_count': chunks_count,
                        'indexed_at': datetime.now().isoformat()
                    }
                    
                    logger.debug(f"Processed CSV record {idx + 1}/{csv_rows_loaded}: {record_id}")
                    
                except Exception as e:
                    error_msg = f"Failed to process CSV record {idx + 1}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            
            load_time = time.time() - start_time
            
            # Save cache if embeddings were generated
            if not skip_embedding and use_cache and all_chunks and all_embeddings:
                self._save_cache(all_chunks, all_embeddings, indexing_service._bm25_index, indexing_service._indexed_documents)
            
            # Print timing summary
            print("\n=== TIMING SUMMARY ===")
            print(f"Total Parse Time: {total_parse_time:.2f}s")
            print(f"Total Chunk Time: {total_chunk_time:.2f}s")
            print(f"Total Embedding Time: {total_embedding_time:.2f}s")
            print(f"Total Vector Upsert Time: {total_vector_upsert_time:.2f}s")
            print(f"Total BM25 Time: {total_bm25_time:.2f}s")
            print(f"Total Load Time: {load_time:.2f}s")
            print("======================\n")
            
            result = CSVIngestionResult(
                csv_rows_loaded=csv_rows_loaded,
                chunks_generated=total_chunks,
                vectors_indexed=total_vectors,
                bm25_documents_indexed=total_bm25_docs,
                errors=errors,
                load_time_seconds=load_time
            )
            
            logger.info(
                f"CSV ingestion complete: "
                f"rows={csv_rows_loaded}, "
                f"chunks={total_chunks}, "
                f"vectors={total_vectors}, "
                f"bm25_docs={total_bm25_docs}, "
                f"time={load_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            load_time = time.time() - start_time
            error_msg = f"CSV ingestion failed: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            
            return CSVIngestionResult(
                csv_rows_loaded=0,
                chunks_generated=0,
                vectors_indexed=0,
                bm25_documents_indexed=0,
                errors=errors,
                load_time_seconds=load_time
            )
    
    def print_ingestion_results(self, result: CSVIngestionResult) -> None:
        """
        Print CSV ingestion results.
        
        Args:
            result: CSVIngestionResult to print
        """
        print("\n" + "="*60)
        print("📊 CSV Resume Ingestion Results")
        print("="*60)
        print(f"CSV Rows Loaded: {result.csv_rows_loaded}")
        print(f"Chunks Generated: {result.chunks_generated}")
        print(f"Vectors Indexed: {result.vectors_indexed}")
        print(f"BM25 Documents Indexed: {result.bm25_documents_indexed}")
        print(f"Load Time: {result.load_time_seconds:.2f}s")
        
        if result.errors:
            print(f"\n⚠️  Errors: {len(result.errors)}")
            for error in result.errors[:5]:  # Show first 5 errors
                print(f"   - {error}")
            if len(result.errors) > 5:
                print(f"   ... and {len(result.errors) - 5} more errors")
        
        print("="*60)
        
        if result.csv_rows_loaded > 0:
            print("🚀 CSV Resume Ingestion Complete")
            print(f"Resumes Indexed: {result.csv_rows_loaded}")
            print(f"Vectors Indexed: {result.vectors_indexed}")
            print(f"BM25 Docs: {result.bm25_documents_indexed}")
        else:
            print("❌ CSV Resume Ingestion Failed")
        
        print("="*60 + "\n")
