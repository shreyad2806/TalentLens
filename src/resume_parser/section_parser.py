"""
Section Parser module - Detects and parses semantic sections in resumes.

This module identifies different sections of a resume (Experience, Education, Skills, etc.)
based on heading patterns and extracts the content for each section.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Section:
    """
    Represents a detected section in a resume.
    
    Attributes:
        name: Name of the section (e.g., "Experience", "Education")
        content: Text content of the section
        start_line: Line number where section starts
        end_line: Line number where section ends
    """
    name: str
    content: str
    start_line: int
    end_line: int


class SectionParser:
    """
    Parser for detecting and extracting semantic sections from resume text.
    
    This class uses regex patterns to identify common resume section headings
    and extracts the content for each section. It handles various formatting
    styles and provides a structured representation of the resume.
    
    Common sections detected:
        - Experience / Work Experience
        - Education
        - Skills
        - Projects
        - Certifications
        - Summary / Objective
        - Languages
    """
    
    # Common section heading patterns (case-insensitive)
    SECTION_PATTERNS = {
        'experience': [
            r'^experience$',
            r'^work experience$',
            r'^professional experience$',
            r'^employment history$',
            r'^work history$',
        ],
        'education': [
            r'^education$',
            r'^academic background$',
            r'^educational qualification$',
        ],
        'skills': [
            r'^skills$',
            r'^technical skills$',
            r'^core competencies$',
            r'^technologies$',
            r'^key skills$',
        ],
        'projects': [
            r'^projects$',
            r'^personal projects$',
            r'^project experience$',
        ],
        'certifications': [
            r'^certifications$',
            r'^certificates$',
            r'^professional certifications$',
        ],
        'summary': [
            r'^summary$',
            r'^professional summary$',
            r'^objective$',
            r'^profile$',
            r'^about me$',
        ],
        'languages': [
            r'^languages$',
            r'^language proficiency$',
        ],
    }
    
    @staticmethod
    def detect_sections(text: str) -> Dict[str, Section]:
        """
        Detect all sections in the resume text.
        
        Args:
            text: Full resume text
            
        Returns:
            Dictionary mapping section names to Section objects
        """
        lines = text.split('\n')
        sections = {}
        
        # Find all section headings
        section_boundaries = SectionParser._find_section_boundaries(lines)
        
        # Extract content for each section
        for section_name, (start_idx, end_idx) in section_boundaries.items():
            content_lines = lines[start_idx + 1:end_idx]  # Skip heading line
            content = '\n'.join(content_lines).strip()
            
            sections[section_name] = Section(
                name=section_name,
                content=content,
                start_line=start_idx,
                end_line=end_idx
            )
        
        return sections
    
    @staticmethod
    def _find_section_boundaries(lines: List[str]) -> Dict[str, Tuple[int, int]]:
        """
        Find the start and end line indices for each section.
        
        Args:
            lines: List of text lines from the resume
            
        Returns:
            Dictionary mapping section names to (start, end) line indices
        """
        boundaries = {}
        current_section = None
        current_start = 0
        
        for i, line in enumerate(lines):
            line_clean = line.strip().lower()
            
            # Check if this line matches any section pattern
            matched_section = SectionParser._match_section_heading(line_clean)
            
            if matched_section:
                # Save previous section if exists
                if current_section:
                    boundaries[current_section] = (current_start, i)
                
                # Start new section
                current_section = matched_section
                current_start = i
        
        # Save last section
        if current_section:
            boundaries[current_section] = (current_start, len(lines))
        
        return boundaries
    
    @staticmethod
    def _match_section_heading(line: str) -> Optional[str]:
        """
        Check if a line matches any known section heading pattern.
        
        Args:
            line: Lowercase line text to check
            
        Returns:
            Section name if matched, None otherwise
        """
        for section_name, patterns in SectionParser.SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    return section_name
        return None
    
    @staticmethod
    def get_section_content(text: str, section_name: str) -> Optional[str]:
        """
        Get content for a specific section.
        
        Args:
            text: Full resume text
            section_name: Name of section to extract (e.g., 'experience', 'education')
            
        Returns:
            Section content if found, None otherwise
        """
        sections = SectionParser.detect_sections(text)
        section = sections.get(section_name)
        return section.content if section else None
    
    @staticmethod
    def extract_contact_info(text: str) -> Dict[str, Optional[str]]:
        """
        Extract contact information from the beginning of the resume.
        
        This method looks for email, phone, and name patterns in the
        first few lines of the resume (typically the header).
        
        Args:
            text: Full resume text
            
        Returns:
            Dictionary with 'name', 'email', and 'phone' keys
        """
        lines = text.split('\n')
        header_lines = lines[:10]  # Check first 10 lines for contact info
        
        contact_info = {
            'name': None,
            'email': None,
            'phone': None,
        }
        
        header_text = '\n'.join(header_lines)
        
        # Extract email
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        email_match = re.search(email_pattern, header_text)
        if email_match:
            contact_info['email'] = email_match.group()
        
        # Extract phone (various formats)
        phone_patterns = [
            r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # International
            r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',  # US format
            r'\d{10}',  # Simple 10-digit
        ]
        for pattern in phone_patterns:
            phone_match = re.search(pattern, header_text)
            if phone_match:
                contact_info['phone'] = phone_match.group()
                break
        
        # Extract name (heuristic: first line that's not email/phone)
        for line in header_lines:
            line_clean = line.strip()
            if line_clean and not re.search(email_pattern, line_clean):
                is_phone = False
                for pattern in phone_patterns:
                    if re.search(pattern, line_clean):
                        is_phone = True
                        break
                if not is_phone and len(line_clean.split()) <= 4:  # Name likely has 2-4 words
                    contact_info['name'] = line_clean
                    break
        
        return contact_info
    
    @staticmethod
    def extract_summary(text: str) -> Optional[str]:
        """
        Extract the professional summary section.
        
        Args:
            text: Full resume text
            
        Returns:
            Summary text if found, None otherwise
        """
        return SectionParser.get_section_content(text, 'summary')
    
    @staticmethod
    def extract_skills_section(text: str) -> Optional[str]:
        """
        Extract the skills section.
        
        Args:
            text: Full resume text
            
        Returns:
            Skills section text if found, None otherwise
        """
        return SectionParser.get_section_content(text, 'skills')
    
    @staticmethod
    def extract_experience_section(text: str) -> Optional[str]:
        """
        Extract the experience section.
        
        Args:
            text: Full resume text
            
        Returns:
            Experience section text if found, None otherwise
        """
        return SectionParser.get_section_content(text, 'experience')
    
    @staticmethod
    def extract_education_section(text: str) -> Optional[str]:
        """
        Extract the education section.
        
        Args:
            text: Full resume text
            
        Returns:
            Education section text if found, None otherwise
        """
        return SectionParser.get_section_content(text, 'education')
    
    @staticmethod
    def extract_projects_section(text: str) -> Optional[str]:
        """
        Extract the projects section.
        
        Args:
            text: Full resume text
            
        Returns:
            Projects section text if found, None otherwise
        """
        return SectionParser.get_section_content(text, 'projects')
    
    @staticmethod
    def extract_certifications_section(text: str) -> Optional[str]:
        """
        Extract the certifications section.
        
        Args:
            text: Full resume text
            
        Returns:
            Certifications section text if found, None otherwise
        """
        return SectionParser.get_section_content(text, 'certifications')
    
    @staticmethod
    def extract_languages_section(text: str) -> Optional[str]:
        """
        Extract the languages section.
        
        Args:
            text: Full resume text
            
        Returns:
            Languages section text if found, None otherwise
        """
        return SectionParser.get_section_content(text, 'languages')
