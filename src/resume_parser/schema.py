"""
Schema module - Data models for resume documents.

This module defines Pydantic data models for structured resume data.
These models provide type safety, validation, and serialization capabilities.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class Experience(BaseModel):
    """
    Represents a work experience entry in a resume.
    
    Attributes:
        company: Name of the company/organization
        title: Job title/position
        location: Geographic location of the job
        start_date: Start date of employment
        end_date: End date of employment (None if current)
        description: Job description and responsibilities
        current: Whether this is the current position
    """
    company: Optional[str] = Field(None, description="Company name")
    title: Optional[str] = Field(None, description="Job title")
    location: Optional[str] = Field(None, description="Job location")
    start_date: Optional[str] = Field(None, description="Start date (flexible format)")
    end_date: Optional[str] = Field(None, description="End date (None if current)")
    description: Optional[str] = Field(None, description="Job description")
    current: bool = Field(False, description="Whether this is current position")
    
    @field_validator('end_date')
    @classmethod
    def validate_end_date(cls, v, info):
        """Ensure end_date is None if current is True."""
        if info.data.get('current') and v is not None:
            return None
        return v


class Education(BaseModel):
    """
    Represents an education entry in a resume.
    
    Attributes:
        institution: Name of the educational institution
        degree: Degree obtained (e.g., "Bachelor of Science")
        field_of_study: Major or field of study
        location: Geographic location of the institution
        start_date: Start date of education
        end_date: End date/graduation date
        gpa: Grade point average if applicable
        description: Additional details about education
    """
    institution: Optional[str] = Field(None, description="Institution name")
    degree: Optional[str] = Field(None, description="Degree obtained")
    field_of_study: Optional[str] = Field(None, description="Major/field of study")
    location: Optional[str] = Field(None, description="Institution location")
    start_date: Optional[str] = Field(None, description="Start date")
    end_date: Optional[str] = Field(None, description="Graduation date")
    gpa: Optional[str] = Field(None, description="Grade point average")
    description: Optional[str] = Field(None, description="Additional details")


class Project(BaseModel):
    """
    Represents a project entry in a resume.
    
    Attributes:
        name: Project name
        description: Project description
        technologies: List of technologies/tools used
        role: Role in the project
        start_date: Project start date
        end_date: Project end date
        url: Project URL/link if available
    """
    name: Optional[str] = Field(None, description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    technologies: List[str] = Field(default_factory=list, description="Technologies used")
    role: Optional[str] = Field(None, description="Role in project")
    start_date: Optional[str] = Field(None, description="Project start date")
    end_date: Optional[str] = Field(None, description="Project end date")
    url: Optional[str] = Field(None, description="Project URL")


class Certification(BaseModel):
    """
    Represents a certification entry in a resume.
    
    Attributes:
        name: Certification name
        issuer: Issuing organization
        issue_date: Date certification was issued
        expiration_date: Certification expiration date
        credential_id: Credential ID if applicable
        url: Certification verification URL
    """
    name: Optional[str] = Field(None, description="Certification name")
    issuer: Optional[str] = Field(None, description="Issuing organization")
    issue_date: Optional[str] = Field(None, description="Issue date")
    expiration_date: Optional[str] = Field(None, description="Expiration date")
    credential_id: Optional[str] = Field(None, description="Credential ID")
    url: Optional[str] = Field(None, description="Verification URL")


class ResumeDocument(BaseModel):
    """
    Unified resume document schema containing all extracted information.
    
    This is the main data model that the parser service returns. It contains
    all structured fields extracted from a resume document.
    
    Attributes:
        name: Full name of the candidate
        email: Email address
        phone: Phone number
        summary: Professional summary/objective
        skills: List of technical and soft skills
        experience: List of work experience entries
        projects: List of project entries
        education: List of education entries
        certifications: List of certification entries
        languages: List of languages spoken
        raw_text: Original raw text from the document
        metadata: Additional metadata (parsing timestamps, sources, etc.)
    """
    # Contact Information
    name: Optional[str] = Field(None, description="Full name of candidate")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    
    # Professional Summary
    summary: Optional[str] = Field(None, description="Professional summary")
    
    # Skills
    skills: List[str] = Field(default_factory=list, description="List of skills")
    
    # Experience
    experience: List[Experience] = Field(default_factory=list, description="Work experience")
    
    # Projects
    projects: List[Project] = Field(default_factory=list, description="Projects")
    
    # Education
    education: List[Education] = Field(default_factory=list, description="Education")
    
    # Certifications
    certifications: List[Certification] = Field(default_factory=list, description="Certifications")
    
    # Languages
    languages: List[str] = Field(default_factory=list, description="Languages spoken")
    
    # Raw Data
    raw_text: str = Field(..., description="Original raw text from document")
    
    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (parsing info, sources, etc.)"
    )
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
        }
    }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the ResumeDocument to a dictionary.
        
        Returns:
            Dictionary representation of the resume document
        """
        return self.dict()
    
    def to_json(self) -> str:
        """
        Convert the ResumeDocument to JSON string.
        
        Returns:
            JSON string representation
        """
        import json
        return json.dumps(self.dict(), indent=2)
