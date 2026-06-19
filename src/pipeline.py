"""
Ingestion Pipeline module - Integrated pipeline for resume processing.

This module provides a unified pipeline that orchestrates the entire resume
ingestion process: parsing, chunking, and embedding. The pipeline maintains
backward compatibility while adding the embedding layer.

Pipeline Flow:
Resume.pdf → ParserService → ResumeDocument → ChunkService → List[Chunk] → EmbeddingService → List[EmbeddingRecord]
"""

import logging
import time
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from datetime import datetime

from .resume_parser.parser_service import ParserService
from .resume_parser.schema import ResumeDocument
from .chunks.service import ChunkService
from .chunks.schema import Chunk
from .embeddings.embedding_service import EmbeddingService
from .embeddings.schema import EmbeddingRecord


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Unified ingestion pipeline for resume processing.
    
    This class orchestrates the entire resume ingestion process, integrating
    the parser, chunking, and embedding layers. It provides timing information
    and logging for each stage while maintaining backward compatibility.
    
    The pipeline follows a clear flow:
    1. Parse resume file to ResumeDocument
    2. Chunk ResumeDocument to List[Chunk]
    3. Embed List[Chunk] to List[EmbeddingRecord]
    """
    
    def __init__(self):
        """
        Initialize the ingestion pipeline with component services.
        """
        self.parser = ParserService()
        self.chunk_service = ChunkService()
        self.embedding_service = EmbeddingService()
        
        logger.info("IngestionPipeline initialized")
    
    def process_resume(
        self,
        file_path: Union[str, Path],
        resume_id: Optional[str] = None,
        enable_embeddings: bool = True
    ) -> Dict[str, Any]:
        """
        Process a resume file through the complete ingestion pipeline.
        
        This method orchestrates the entire pipeline: parsing, chunking, and
        optionally embedding. It returns a dictionary with all intermediate
        results and timing information.
        
        Args:
            file_path: Path to the resume file (PDF, DOCX, or TXT)
            resume_id: Optional resume ID (auto-generated if not provided)
            enable_embeddings: Whether to generate embeddings (default: True)
            
        Returns:
            Dictionary containing:
                - 'resume_document': Parsed ResumeDocument
                - 'chunks': List of Chunk objects
                - 'embeddings': List of EmbeddingRecord objects (if enabled)
                - 'timing': Dictionary with timing information for each stage
                - 'stats': Dictionary with statistics
        """
        # Generate resume ID if not provided
        if resume_id is None:
            resume_id = f"resume-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        logger.info(f"Starting ingestion pipeline for: {file_path}")
        logger.info(f"Resume ID: {resume_id}")
        logger.info(f"Embeddings enabled: {enable_embeddings}")
        
        # Initialize timing
        timing = {}
        total_start_time = time.time()
        
        # Stage 1: Parsing
        logger.info("Stage 1: Parsing resume...")
        parser_start = time.time()
        try:
            resume_document = self.parser.parse_file(file_path)
            parser_time = time.time() - parser_start
            timing['parser_time'] = parser_time
            logger.info(f"Parser completed in {parser_time:.3f} seconds")
        except Exception as e:
            logger.error(f"Parser failed: {e}")
            raise
        
        # Stage 2: Chunking
        logger.info("Stage 2: Creating chunks...")
        chunking_start = time.time()
        try:
            chunks = self.chunk_service.create_chunks(resume_document, resume_id=resume_id)
            chunking_time = time.time() - chunking_start
            timing['chunking_time'] = chunking_time
            logger.info(f"Chunking completed in {chunking_time:.3f} seconds")
            logger.info(f"Generated {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Chunking failed: {e}")
            raise
        
        # Stage 3: Embedding (optional)
        embeddings = None
        if enable_embeddings:
            logger.info("Stage 3: Generating embeddings...")
            embedding_start = time.time()
            try:
                result = self.embedding_service.embed_chunks_with_stats(chunks)
                embeddings = result['embeddings']
                embedding_stats = result['stats']
                embedding_time = time.time() - embedding_start
                timing['embedding_time'] = embedding_time
                timing['embedding_stats'] = embedding_stats
                logger.info(f"Embedding completed in {embedding_time:.3f} seconds")
                logger.info(f"Generated {len(embeddings)} embeddings")
                logger.info(f"Cache hit rate: {embedding_stats['cache']['hit_rate']:.2%}")
            except Exception as e:
                logger.error(f"Embedding failed: {e}")
                raise
        
        # Calculate total time
        total_time = time.time() - total_start_time
        timing['total_time'] = total_time
        
        # Build statistics
        stats = {
            'resume_id': resume_id,
            'candidate_name': resume_document.name,
            'total_chunks': len(chunks),
            'total_embeddings': len(embeddings) if embeddings else 0,
            'sections_detected': resume_document.metadata.get('sections_detected', []),
            'skills_count': len(resume_document.skills),
            'experience_count': len(resume_document.experience),
            'education_count': len(resume_document.education),
        }
        
        logger.info(f"Pipeline completed in {total_time:.3f} seconds")
        logger.info(f"Candidate: {stats['candidate_name']}")
        logger.info(f"Chunks: {stats['total_chunks']}")
        logger.info(f"Embeddings: {stats['total_embeddings']}")
        
        # Return results
        result = {
            'resume_document': resume_document,
            'chunks': chunks,
            'embeddings': embeddings,
            'timing': timing,
            'stats': stats
        }
        
        return result
    
    def process_resume_legacy(
        self,
        file_path: Union[str, Path],
        resume_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a resume file without embeddings (legacy compatibility).
        
        This method provides backward compatibility for code that expects
        the pipeline to return only parsed and chunked data without embeddings.
        
        Args:
            file_path: Path to the resume file (PDF, DOCX, or TXT)
            resume_id: Optional resume ID (auto-generated if not provided)
            
        Returns:
            Dictionary containing:
                - 'resume_document': Parsed ResumeDocument
                - 'chunks': List of Chunk objects
                - 'timing': Dictionary with timing information
        """
        result = self.process_resume(file_path, resume_id=resume_id, enable_embeddings=False)
        
        # Return legacy format
        return {
            'resume_document': result['resume_document'],
            'chunks': result['chunks'],
            'timing': result['timing']
        }
    
    def print_timing_summary(self, timing: Dict[str, Any]) -> None:
        """
        Print a formatted timing summary for the pipeline.
        
        Args:
            timing: Dictionary with timing information
        """
        print("\n" + "=" * 80)
        print("PIPELINE TIMING SUMMARY")
        print("=" * 80)
        print(f"Parser Time:      {timing.get('parser_time', 0):.3f} seconds")
        print(f"Chunk Time:       {timing.get('chunking_time', 0):.3f} seconds")
        print(f"Embedding Time:   {timing.get('embedding_time', 0):.3f} seconds")
        print(f"Total Pipeline:   {timing.get('total_time', 0):.3f} seconds")
        print("=" * 80)
        
        # Print cache stats if available
        if 'embedding_stats' in timing:
            cache_stats = timing['embedding_stats']['cache']
            print(f"Cache Hits:       {cache_stats['hits']}")
            print(f"Cache Misses:     {cache_stats['misses']}")
            print(f"Cache Hit Rate:   {cache_stats['hit_rate']:.2%}")
            print("=" * 80)
    
    def print_stats_summary(self, stats: Dict[str, Any]) -> None:
        """
        Print a formatted statistics summary for the pipeline.
        
        Args:
            stats: Dictionary with statistics
        """
        print("\n" + "=" * 80)
        print("PIPELINE STATISTICS")
        print("=" * 80)
        print(f"Resume ID:        {stats['resume_id']}")
        print(f"Candidate Name:   {stats['candidate_name']}")
        print(f"Total Chunks:     {stats['total_chunks']}")
        print(f"Total Embeddings: {stats['total_embeddings']}")
        print(f"Skills Count:     {stats['skills_count']}")
        print(f"Experience Count: {stats['experience_count']}")
        print(f"Education Count:  {stats['education_count']}")
        print(f"Sections:         {', '.join(stats['sections_detected'])}")
        print("=" * 80)
