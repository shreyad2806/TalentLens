"""
Build Index Script for TalentLens

This script builds the production index for TalentLens by:
1. Loading resumes from CSV
2. Chunking resume text
3. Generating embeddings
4. Upserting vectors to Qdrant
5. Building BM25 index
6. Saving BM25 index to disk

Usage:
    python scripts/build_index.py
"""

import sys
import time
import csv
import pickle
import uuid
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import EMBEDDING_DIM, EMBEDDING_MODEL
from src.embeddings.embedding_service import EmbeddingService
from src.chunks.schema import Chunk, ChunkMetadata, EmbeddingStatus
from src.chunks.service import ChunkService
from src.vector_store.service import VectorStoreService
from src.vector_store.config import VectorStoreConfig, VectorStoreProvider
from src.retrieval.bm25.bm25_index import BM25Index
from src.retrieval.bm25.index_builder import IndexBuilder as BM25IndexBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionIndexBuilder:
    """Build production index for TalentLens."""
    
    def __init__(self):
        """Initialize the index builder."""
        self.csv_path = Path("Resume/Resume.csv")
        self.bm25_cache_path = Path("data/indexes/bm25")
        self.bm25_cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize vector store service
        self.vector_store_config = VectorStoreConfig()
        self.vector_store_service = VectorStoreService()
        
        # Initialize services
        self.embedding_service = EmbeddingService()
        self.chunk_service = ChunkService()
        self.bm25_index = BM25Index()
        self.index_builder = BM25IndexBuilder()
        
        # Statistics
        self.stats = {
            "resumes_loaded": 0,
            "chunks_generated": 0,
            "vectors_upserted": 0,
            "bm25_documents": 0,
            "total_time": 0
        }
    
    def load_resumes(self) -> List[Dict[str, Any]]:
        """Load resumes from CSV."""
        logger.info(f"Loading resumes from {self.csv_path}")
        resumes = []
        
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get('Resume_str', '')
                if text.strip():
                    resumes.append({
                        'id': row.get('ID', ''),
                        'text': text.strip(),
                        'candidate': row.get('Candidate', ''),
                        'category': row.get('Category', '')
                    })
        
        logger.info(f"Loaded {len(resumes)} resumes")
        self.stats["resumes_loaded"] = len(resumes)
        return resumes
    
    def chunk_resume(self, resume: Dict[str, Any], resume_id: str) -> List[Chunk]:
        """Chunk a single resume."""
        raw_text = resume['text']
        candidate_name = resume.get('candidate', '')
        
        # Simple chunking (similar to CSV ingestion)
        chunk_size = 1000
        overlap = 100
        
        chunks = []
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
            
            chunk_metadata = ChunkMetadata(
                candidate_name=candidate_name,
                role=None,
                experience=None,
                location=None,
                education=None,
                skills=[],
                email=None,
                phone=None,
                summary=None,
                source_section="raw_text"
            )
            
            # [META-WRITE] Log ChunkMetadata creation
            _meta_dict = chunk_metadata.dict()
            _non_null = {k: v for k, v in _meta_dict.items() if v is not None and v != [] and v != ''}
            print(f"[META-WRITE][ChunkMetadata][build_index] resume_id={resume_id[:8]}  chunk_order={chunk_order}  keys={sorted(_meta_dict.keys())}  non_null={list(_non_null.keys())}")
            
            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                resume_id=resume_id,
                candidate_name=candidate_name,
                section="raw_text",
                text=chunk_text,
                metadata=chunk_metadata,
                chunk_order=chunk_order,
                created_at=datetime.now(),
                embedding_status=EmbeddingStatus.PENDING,
                source_document=str(self.csv_path)
            )
            
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start = end - overlap
            
            # Prevent infinite loop if overlap >= chunk_size
            if overlap >= chunk_size:
                start = end
            
            chunk_order += 1
        
        return chunks
    
    def build_index(self):
        """Build the complete index."""
        start_time = time.time()
        
        # Startup diagnostics
        print("="*60)
        print("INDEX BUILDER CONFIGURATION")
        print("="*60)
        provider = self.vector_store_config.provider
        print(f"Provider: {provider.value}")
        print(f"Embedding Model: {EMBEDDING_MODEL}")
        print(f"Dimension: {EMBEDDING_DIM}")
        print(f"Persistence Enabled: True")
        print("="*60)
        
        # Log which adapter is being used
        if provider == VectorStoreProvider.MEMORY:
            print("\nUsing MemoryAdapter")
        elif provider == VectorStoreProvider.QDRANT:
            print("\nUsing QdrantAdapter")
        elif provider == VectorStoreProvider.PINECONE:
            print("\nUsing PineconeAdapter")
        
        print("\n" + "="*60)
        print("TALENTLENS INDEX BUILDER")
        print("="*60)
        
        # Step 1: Check vector store connection (skip for memory)
        print("\n[1/6] Checking vector store connection...")
        if provider == VectorStoreProvider.MEMORY:
            print(f"✓ Memory adapter: No connection required")
        else:
            try:
                health_status = self.vector_store_service.health()
                print(f"✓ Vector store connected")
                print(f"  Status: {health_status.get('status', 'unknown')}")
            except Exception as e:
                print(f"ERROR: Vector store health check failed: {e}")
                return False
        
        # Step 2: Clear existing data if needed
        print("\n[2/6] Checking existing data...")
        try:
            vector_count = self.vector_store_service.count()
            print(f"  Current vector count: {vector_count}")
            if vector_count > 0:
                response = input("Delete existing vectors? (y/n): ")
                if response.lower() == 'y':
                    print("  Clearing existing vectors...")
                    # Note: Memory adapter doesn't support delete, so we skip this
                    if provider == VectorStoreProvider.MEMORY:
                        print("  Memory adapter: Will overwrite on upsert")
                    else:
                        # For Qdrant/Pinecone, we would need to implement delete
                        print("  Clearing not implemented for this adapter")
                        print("  Will overwrite on upsert")
                else:
                    print("  Using existing data (will append)")
        except Exception as e:
            print(f"  Could not check vector count: {e}")
            print("  Continuing anyway...")
        
        # Step 3: Load resumes
        print("\n[3/6] Loading resumes from CSV...")
        resumes = self.load_resumes()
        
        # Step 4: Chunk and embed
        print("\n[4/6] Chunking and embedding resumes...")
        all_chunks = []
        all_embeddings = []
        all_vectors = []
        all_payloads = []
        
        batch_size = 100
        for idx, resume in enumerate(resumes):
            resume_id = resume['id'] or f"resume_{idx}"
            
            # Chunk resume
            chunks = self.chunk_resume(resume, resume_id)
            all_chunks.extend(chunks)
            
            # Generate embeddings
            if len(chunks) > 0:
                embeddings = self.embedding_service.embed_chunks(chunks)
                all_embeddings.extend(embeddings)
                
                # Prepare vectors and payloads
                for chunk, embedding in zip(chunks, embeddings):
                    all_vectors.append(embedding.vector)
                    all_payloads.append({
                        "chunk_id": chunk.chunk_id,
                        "resume_id": chunk.resume_id,
                        "candidate_name": chunk.candidate_name,
                        "section": chunk.section,
                        "text": chunk.text,
                        "chunk_order": chunk.chunk_order
                    })
            
            # Batch upsert to vector store
            if len(all_vectors) >= batch_size:
                print(f"  Upserting {len(all_vectors)} vectors to {provider.value}...")
                self._batch_upsert_vectors(all_vectors, all_payloads)
                self.stats["vectors_upserted"] += len(all_vectors)
                all_vectors = []
                all_payloads = []
            
            # Progress
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(resumes)} resumes")
        
        # Upsert remaining vectors
        if all_vectors:
            print(f"  Upserting {len(all_vectors)} vectors to {provider.value}...")
            self._batch_upsert_vectors(all_vectors, all_payloads)
            self.stats["vectors_upserted"] += len(all_vectors)
        
        self.stats["chunks_generated"] = len(all_chunks)
        print(f"✓ Generated {len(all_chunks)} chunks")
        print(f"✓ Upserted {self.stats['vectors_upserted']} vectors to {provider.value}")
        
        # Step 5: Build BM25 index
        print("\n[5/6] Building BM25 index...")
        for chunk in all_chunks:
            bm25_doc, tokens = self.index_builder.chunk_to_document(chunk)
            self.bm25_index.add_document(
                document_id=bm25_doc.document_id,
                tokens=tokens,
                document=bm25_doc
            )
        
        self.stats["bm25_documents"] = len(all_chunks)
        print(f"✓ Built BM25 index with {len(all_chunks)} documents")
        
        # Step 6: Save BM25 index
        print("\n[6/6] Saving BM25 index...")
        self.bm25_index.save_to_disk(self.bm25_cache_path)
        print(f"✓ BM25 index saved to {self.bm25_cache_path}")
        
        # Final timing
        total_time = time.time() - start_time
        self.stats["total_time"] = total_time
        
        # Log persistence path
        print(f"\nPersistence Path: {self.bm25_cache_path}")
        
        print("\n" + "="*60)
        print("INDEX BUILD COMPLETE")
        print("="*60)
        print(f"Resumes loaded: {self.stats['resumes_loaded']}")
        print(f"Chunks generated: {self.stats['chunks_generated']}")
        print(f"Vectors upserted: {self.stats['vectors_upserted']}")
        print(f"BM25 documents: {self.stats['bm25_documents']}")
        print(f"Total time: {total_time:.2f}s")
        print("="*60)
        
        return True
    
    def _batch_upsert_vectors(self, vectors: List[List[float]], payloads: List[Dict[str, Any]]):
        """Batch upsert vectors to the vector store."""
        from src.vector_store.schema import VectorRecord
        import uuid
        
        records = []
        for vector, payload in zip(vectors, payloads):
            record = VectorRecord(
                id=str(uuid.uuid4()),
                resume_id=payload.get("resume_id", ""),
                chunk_id=payload.get("chunk_id", ""),
                candidate_name=payload.get("candidate_name") or "Unknown",
                section=payload.get("section", ""),
                vector=vector,
                metadata={
                    "text": payload.get("text", ""),
                    "chunk_order": payload.get("chunk_order", 0)
                }
            )
            records.append(record)
        
        try:
            self.vector_store_service.upsert(records)
        except Exception as e:
            print(f"  Warning: Batch upsert failed: {e}")
            # Fall back to individual upserts
            for record in records:
                try:
                    self.vector_store_service.upsert([record])
                except Exception as e2:
                    print(f"  Error upserting record {record.id}: {e2}")


def main():
    """Main entry point."""
    builder = ProductionIndexBuilder()
    success = builder.build_index()
    
    if success:
        print("\n✓ Index built successfully!")
        print("You can now start the application with: streamlit run app.py")
    else:
        print("\n✗ Index build failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
