"""
Chunk Generator module - Generates Chunk objects from ResumeDocument.

This module takes the output from the SemanticChunker and converts it into
proper Chunk objects with unique IDs and proper metadata structure.
"""

import uuid
from typing import List

from ..resume_parser.schema import ResumeDocument
from .semantic_chunker import SemanticChunker, ChunkData
from .schema import Chunk, ChunkMetadata
import logging

logger = logging.getLogger(__name__)


class ChunkGenerator:
    """
    Generator that creates Chunk objects from ResumeDocument.
    
    This class takes a ResumeDocument, uses the SemanticChunker to break it
    into logical sections, and then converts those sections into proper Chunk
    objects with unique IDs and structured metadata.
    
    Every chunk preserves metadata about the candidate and knows which resume
    it belongs to, making it suitable for RAG ingestion and retrieval.
    """
    
    def __init__(self):
        """
        Initialize the chunk generator with a semantic chunker.
        """
        self.semantic_chunker = SemanticChunker()
    
    def generate_chunks(self, document: ResumeDocument, resume_id: str) -> List[Chunk]:
        """
        Generate Chunk objects from a ResumeDocument.
        
        This method orchestrates the chunk generation process:
        1. Use SemanticChunker to break document into logical sections
        2. Convert each section into a Chunk object with unique ID
        3. Ensure proper metadata structure
        
        Args:
            document: The ResumeDocument to chunk
            resume_id: Unique identifier for the resume
            
        Returns:
            List of Chunk objects representing semantic chunks
        """
        # Step 1: Use semantic chunker to get raw chunk data
        chunk_data_list = self.semantic_chunker.chunk_document(document, resume_id)
        
        # Step 2: Convert each ChunkData to a Chunk object
        chunks = []
        for chunk_data in chunk_data_list:
            chunk = self._create_chunk(
                chunk_data=chunk_data,
                resume_id=resume_id,
                candidate_name=document.name,
                chunk_order=chunk_data_list.index(chunk_data)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, chunk_data: ChunkData, resume_id: str, 
                     candidate_name: str, chunk_order: int) -> Chunk:
        """
        Create a Chunk object from ChunkData.
        
        Args:
            chunk_data: The raw chunk data from semantic chunker
            resume_id: Unique identifier for the resume
            candidate_name: Name of the candidate
            chunk_order: Order of this chunk within the resume
            
        Returns:
            Chunk object
        """
        # Generate unique chunk ID
        chunk_id = str(uuid.uuid4())
        
        # Create structured metadata — propagate all ResumeDocument fields
        _skills = list(document.skills) if document.skills else []
        chunk_metadata = ChunkMetadata(
            candidate_name=candidate_name,
            experience=chunk_data.metadata.get('experience'),
            location=chunk_data.metadata.get('location'),
            role=chunk_data.metadata.get('role'),
            education=chunk_data.metadata.get('education'),
            skills=_skills,
            email=document.email,
            phone=document.phone,
            summary=document.summary,
            source_section=chunk_data.metadata.get('source_section')
        )
        
        # [META-WRITE] Log ChunkMetadata creation
        _meta_dict = chunk_metadata.dict()
        _non_null = {k: v for k, v in _meta_dict.items() if v is not None and v != [] and v != ''}
        print(f"[META-WRITE][ChunkMetadata][ChunkGenerator] resume_id={resume_id[:8]}  section={chunk_data.section}  keys={sorted(_meta_dict.keys())}  non_null={list(_non_null.keys())}")
        
        # Create and return Chunk object
        return Chunk(
            chunk_id=chunk_id,
            resume_id=resume_id,
            candidate_name=candidate_name,
            section=chunk_data.section,
            text=chunk_data.text,
            metadata=chunk_metadata,
            chunk_order=chunk_order
        )
