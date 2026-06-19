"""
Factory module - ChunkFactory for creating Chunk objects.

This module implements the ChunkFactory class that converts semantic sections
from ResumeDocument into structured Chunk objects with proper metadata and
unique identifiers.
"""

import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..resume_parser.schema import ResumeDocument
from .schema import Chunk, ChunkMetadata, EmbeddingStatus


class ChunkFactory:
    """
    Factory for creating Chunk objects from ResumeDocument.
    
    This class is responsible for converting semantic sections from a
    ResumeDocument into properly structured Chunk objects. It ensures that
    each chunk receives:
    - UUID chunk_id
    - resume_id
    - metadata
    - chunk_order
    
    The factory follows the Single Responsibility Principle by focusing only
    on chunk creation, not validation or business logic.
    """
    
    def __init__(self):
        """
        Initialize the ChunkFactory.
        """
        self.chunk_order_counter = 0
    
    def create_chunks(self, document: ResumeDocument, resume_id: str, 
                     source_document: Optional[str] = None) -> List[Chunk]:
        """
        Create Chunk objects from a ResumeDocument.
        
        This method converts semantic sections from the ResumeDocument into
        Chunk objects. Each semantic section becomes one Chunk.
        
        Args:
            document: The ResumeDocument to convert
            resume_id: Unique identifier for the resume
            source_document: Optional source document identifier or path
            
        Returns:
            List of Chunk objects
        """
        self.chunk_order_counter = 0
        chunks = []
        
        # Extract base metadata from document
        base_metadata = self._extract_base_metadata(document)
        
        # Create chunk for summary
        if document.summary:
            chunk = self._create_chunk(
                section="summary",
                text=document.summary,
                metadata={**base_metadata, "source_section": "summary"},
                resume_id=resume_id,
                candidate_name=document.name,
                source_document=source_document
            )
            chunks.append(chunk)
        
        # Create chunk for skills
        if document.skills:
            skills_text = ", ".join(document.skills)
            chunk = self._create_chunk(
                section="skills",
                text=skills_text,
                metadata={**base_metadata, "source_section": "skills"},
                resume_id=resume_id,
                candidate_name=document.name,
                source_document=source_document
            )
            chunks.append(chunk)
        
        # Create chunks for experience entries (one per entry)
        for i, exp in enumerate(document.experience):
            exp_text = self._format_experience_entry(exp)
            chunk = self._create_chunk(
                section=f"experience_{i+1}",
                text=exp_text,
                metadata={**base_metadata, "source_section": "experience"},
                resume_id=resume_id,
                candidate_name=document.name,
                source_document=source_document
            )
            chunks.append(chunk)
        
        # Create chunks for projects (one per project)
        for i, project in enumerate(document.projects):
            project_text = self._format_project_entry(project)
            chunk = self._create_chunk(
                section=f"project_{i+1}",
                text=project_text,
                metadata={**base_metadata, "source_section": "projects"},
                resume_id=resume_id,
                candidate_name=document.name,
                source_document=source_document
            )
            chunks.append(chunk)
        
        # Create chunks for education entries (one per entry)
        for i, edu in enumerate(document.education):
            edu_text = self._format_education_entry(edu)
            chunk = self._create_chunk(
                section=f"education_{i+1}",
                text=edu_text,
                metadata={**base_metadata, "source_section": "education"},
                resume_id=resume_id,
                candidate_name=document.name,
                source_document=source_document
            )
            chunks.append(chunk)
        
        # Create chunk for certifications
        if document.certifications:
            cert_text = self._format_certifications(document.certifications)
            chunk = self._create_chunk(
                section="certifications",
                text=cert_text,
                metadata={**base_metadata, "source_section": "certifications"},
                resume_id=resume_id,
                candidate_name=document.name,
                source_document=source_document
            )
            chunks.append(chunk)
        
        # Create chunk for languages
        if document.languages:
            lang_text = ", ".join(document.languages)
            chunk = self._create_chunk(
                section="languages",
                text=lang_text,
                metadata={**base_metadata, "source_section": "languages"},
                resume_id=resume_id,
                candidate_name=document.name,
                source_document=source_document
            )
            chunks.append(chunk)
        
        return chunks
    
    def _extract_base_metadata(self, document: ResumeDocument) -> Dict[str, Any]:
        """
        Extract base metadata from ResumeDocument.
        
        Args:
            document: The ResumeDocument
            
        Returns:
            Dictionary with base metadata
        """
        # Determine role from experience
        role = None
        if document.experience:
            role = document.experience[0].title
        
        # Get location from metadata
        location = document.metadata.get('location')
        
        # Get experience years from metadata
        experience_years = document.metadata.get('total_experience_years')
        
        # Get education from education entries
        education = None
        if document.education:
            education = document.education[0].institution
        
        return {
            "role": role,
            "experience": experience_years,
            "location": location,
            "education": education,
        }
    
    def _create_chunk(self, section: str, text: str, metadata: Dict[str, Any],
                     resume_id: str, candidate_name: Optional[str],
                     source_document: Optional[str] = None) -> Chunk:
        """
        Create a single Chunk object.
        
        Args:
            section: Section name
            text: Chunk text
            metadata: Chunk metadata
            resume_id: Resume identifier
            candidate_name: Candidate name
            source_document: Source document identifier
            
        Returns:
            Chunk object
        """
        chunk_id = str(uuid.uuid4())
        chunk_order = self.chunk_order_counter
        self.chunk_order_counter += 1
        
        # Create ChunkMetadata object
        chunk_metadata = ChunkMetadata(
            role=metadata.get('role'),
            experience=metadata.get('experience'),
            location=metadata.get('location'),
            education=metadata.get('education'),
            source_section=metadata.get('source_section')
        )
        
        return Chunk(
            chunk_id=chunk_id,
            resume_id=resume_id,
            candidate_name=candidate_name,
            section=section,
            text=text,
            metadata=chunk_metadata,
            chunk_order=chunk_order,
            created_at=datetime.now(),
            embedding_status=EmbeddingStatus.PENDING,
            source_document=source_document
        )
    
    def _format_experience_entry(self, exp) -> str:
        """
        Format an experience entry as text.
        
        Args:
            exp: Experience object
            
        Returns:
            Formatted text string
        """
        parts = []
        if exp.title:
            parts.append(f"Title: {exp.title}")
        if exp.company:
            parts.append(f"Company: {exp.company}")
        if exp.location:
            parts.append(f"Location: {exp.location}")
        if exp.start_date:
            end_date = exp.end_date or "Present"
            parts.append(f"Duration: {exp.start_date} - {end_date}")
        if exp.description:
            parts.append(f"Description: {exp.description}")
        
        return "\n".join(parts)
    
    def _format_project_entry(self, project) -> str:
        """
        Format a project entry as text.
        
        Args:
            project: Project object
            
        Returns:
            Formatted text string
        """
        parts = []
        if project.name:
            parts.append(f"Project: {project.name}")
        if project.technologies:
            parts.append(f"Technologies: {', '.join(project.technologies)}")
        if project.description:
            parts.append(f"Description: {project.description}")
        
        return "\n".join(parts)
    
    def _format_education_entry(self, edu) -> str:
        """
        Format an education entry as text.
        
        Args:
            edu: Education object
            
        Returns:
            Formatted text string
        """
        parts = []
        if edu.institution:
            parts.append(f"Institution: {edu.institution}")
        if edu.degree:
            parts.append(f"Degree: {edu.degree}")
        if edu.field_of_study:
            parts.append(f"Field of Study: {edu.field_of_study}")
        if edu.start_date:
            end_date = edu.end_date or "Present"
            parts.append(f"Duration: {edu.start_date} - {end_date}")
        if edu.gpa:
            parts.append(f"GPA: {edu.gpa}")
        
        return "\n".join(parts)
    
    def _format_certifications(self, certifications) -> str:
        """
        Format certifications as text.
        
        Args:
            certifications: List of Certification objects
            
        Returns:
            Formatted text string
        """
        cert_texts = []
        for cert in certifications:
            parts = []
            if cert.name:
                parts.append(cert.name)
            if cert.issuer:
                parts.append(f"(Issuer: {cert.issuer})")
            cert_texts.append(" ".join(parts))
        
        return "\n".join(cert_texts)
