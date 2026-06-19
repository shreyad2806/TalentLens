"""
Semantic Chunker module - Breaks ResumeDocument into semantic sections.

This module implements semantic chunking that follows logical resume boundaries
instead of using fixed-size token chunking. Each chunk corresponds to a meaningful
section or subsection of the resume.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.resume_parser.schema import ResumeDocument


@dataclass
class ChunkData:
    """
    Represents raw chunk data before conversion to Chunk objects.
    
    Attributes:
        section: Section name (e.g., "experience_1", "skills", "summary")
        text: The text content of the chunk
        metadata: Additional metadata about the chunk
    """
    section: str
    text: str
    metadata: Dict[str, Any]


class SemanticChunker:
    """
    Chunker that breaks ResumeDocument into semantic sections.
    
    This class implements semantic chunking that follows logical resume boundaries.
    Instead of using fixed-size token chunking, it creates chunks based on the
    natural structure of the resume (summary, skills, experience entries, etc.).
    
    Chunking strategy:
        - Summary: Single chunk
        - Skills: Single chunk
        - Experience: One chunk per experience entry
        - Projects: One chunk per project
        - Education: One chunk per education entry
        - Certifications: Single chunk
        - Languages: Single chunk
    """
    
    def __init__(self):
        """
        Initialize the semantic chunker.
        """
        self.chunk_order = 0
    
    def chunk_document(self, document: ResumeDocument, resume_id: str) -> List[ChunkData]:
        """
        Chunk a ResumeDocument into semantic sections.
        
        This method takes a ResumeDocument and breaks it down into logical chunks
        based on the resume's natural structure.
        
        Args:
            document: The ResumeDocument to chunk
            resume_id: Unique identifier for the resume
            
        Returns:
            List of ChunkData objects representing semantic chunks
        """
        self.chunk_order = 0
        chunks = []
        
        # Extract base metadata from document
        base_metadata = self._extract_base_metadata(document)
        
        # Chunk summary
        if document.summary:
            chunks.append(self._create_chunk_data(
                section="summary",
                text=document.summary,
                metadata={**base_metadata, "source_section": "summary"}
            ))
        
        # Chunk skills
        if document.skills:
            skills_text = ", ".join(document.skills)
            chunks.append(self._create_chunk_data(
                section="skills",
                text=skills_text,
                metadata={**base_metadata, "source_section": "skills"}
            ))
        
        # Chunk experience entries (one per entry)
        for i, exp in enumerate(document.experience):
            exp_text = self._format_experience_entry(exp)
            chunks.append(self._create_chunk_data(
                section=f"experience_{i+1}",
                text=exp_text,
                metadata={**base_metadata, "source_section": "experience"}
            ))
        
        # Chunk projects (one per project)
        for i, project in enumerate(document.projects):
            project_text = self._format_project_entry(project)
            chunks.append(self._create_chunk_data(
                section=f"project_{i+1}",
                text=project_text,
                metadata={**base_metadata, "source_section": "projects"}
            ))
        
        # Chunk education entries (one per entry)
        for i, edu in enumerate(document.education):
            edu_text = self._format_education_entry(edu)
            chunks.append(self._create_chunk_data(
                section=f"education_{i+1}",
                text=edu_text,
                metadata={**base_metadata, "source_section": "education"}
            ))
        
        # Chunk certifications
        if document.certifications:
            cert_text = self._format_certifications(document.certifications)
            chunks.append(self._create_chunk_data(
                section="certifications",
                text=cert_text,
                metadata={**base_metadata, "source_section": "certifications"}
            ))
        
        # Chunk languages
        if document.languages:
            lang_text = ", ".join(document.languages)
            chunks.append(self._create_chunk_data(
                section="languages",
                text=lang_text,
                metadata={**base_metadata, "source_section": "languages"}
            ))
        
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
            "experience": experience_years,
            "location": location,
            "role": role,
            "education": education,
        }
    
    def _create_chunk_data(self, section: str, text: str, metadata: Dict[str, Any]) -> ChunkData:
        """
        Create a ChunkData object with auto-incremented order.
        
        Args:
            section: Section name
            text: Chunk text
            metadata: Chunk metadata
            
        Returns:
            ChunkData object
        """
        chunk_data = ChunkData(
            section=section,
            text=text,
            metadata=metadata
        )
        self.chunk_order += 1
        return chunk_data
    
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
