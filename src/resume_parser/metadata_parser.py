"""
Metadata Parser module - Extracts structured fields from resume sections.

This module parses section content to extract structured data such as
experience entries, education entries, skills, and other metadata.
"""

import re
from typing import List, Optional, Dict, Any

from .schema import Experience, Education, Project, Certification


class MetadataParser:
    """
    Parser for extracting structured metadata from resume sections.
    
    This class takes section content (e.g., the text from the "Experience" section)
    and parses it into structured data models (Experience, Education, etc.).
    It uses regex patterns and heuristics to extract information.
    """
    
    # Comprehensive skill database
    SKILL_DATABASE = [
        # Programming Languages
        "python", "java", "javascript", "typescript", "c++", "c#", "c", "go", "rust",
        "swift", "kotlin", "php", "ruby", "scala", "r", "matlab", "perl", "lua",
        
        # Web Frameworks
        "react", "angular", "vue", "svelte", "next.js", "nuxt.js", "django", "flask",
        "spring", "express", "fastapi", "rails", "laravel", "asp.net", "node.js",
        
        # Databases
        "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "sqlite",
        "oracle", "cassandra", "dynamodb", "firebase", "supabase",
        
        # Cloud & DevOps
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
        "jenkins", "gitlab", "circleci", "ci/cd", "linux", "ubuntu", "windows",
        
        # Data & ML
        "machine learning", "deep learning", "ai", "artificial intelligence",
        "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
        "spark", "hadoop", "tableau", "power bi", "excel", "airflow",
        
        # Tools & Others
        "git", "github", "gitlab", "jira", "confluence", "slack", "figma",
        "jira", "trello", "asana", "rest api", "graphql", "grpc", "microservices",
    ]
    
    @staticmethod
    def _skill_word_re(skill: str) -> re.Pattern:
        """Return a regex that matches a skill as a whole phrase, not a substring."""
        escaped = re.escape(skill)
        # Use lookarounds so punctuation/slashes/plus signs at skill boundaries
        # don't break the match (e.g. "c++" or "ci/cd").
        return re.compile(r'(?<!\w)' + escaped + r'(?!\w)', re.IGNORECASE)

    @staticmethod
    def extract_skills_keywords(text: Optional[str]) -> List[str]:
        """Extract only known skills from arbitrary text using keyword matching."""
        if not text:
            return []

        skills = []
        for skill in MetadataParser.SKILL_DATABASE:
            if MetadataParser._skill_word_re(skill).search(text):
                skills.append(skill)
        return list(set(skills))

    @staticmethod
    def parse_skills(skills_text: Optional[str]) -> List[str]:
        """
        Parse skills from skills section text.
        
        Args:
            skills_text: Text content of skills section
            
        Returns:
            List of extracted skills
        """
        if not skills_text:
            return []
        
        skills = []
        text_lower = skills_text.lower()
        
        # Match skills from database using whole-phrase matching to avoid
        # false positives (e.g. "ai" matching "detail", "r" matching "for").
        for skill in MetadataParser.SKILL_DATABASE:
            if MetadataParser._skill_word_re(skill).search(skills_text):
                skills.append(skill)
        
        # Also extract comma-separated or bullet-point skills.
        # Split by common delimiters and keep short, single-line phrases that
        # look like skills. This prevents entire sentences from being treated as
        # one skill when a resume is comma-heavy.
        delimiters = [',', ';', '\n', '•', '-', '*', '|']
        for delimiter in delimiters:
            if delimiter in skills_text:
                parts = skills_text.split(delimiter)
                for part in parts:
                    part_clean = part.strip().lower()
                    # Normalize internal whitespace/newlines
                    part_clean = re.sub(r'\s+', ' ', part_clean).strip()
                    if part_clean and len(part_clean) > 2 and part_clean not in skills:
                        # Skip obviously long sentence fragments
                        if len(part_clean) > 60:
                            continue
                        if len(part_clean.split()) > 6:
                            continue
                        # Check it looks like a skill (alphanumeric with spaces
                        # and common separators used in skill names).
                        if re.match(r'^[a-z0-9\s\+\#\.\-/\(\)&]+$', part_clean):
                            skills.append(part_clean)
        
        return list(set(skills))  # Remove duplicates
    
    @staticmethod
    def parse_experience(experience_text: Optional[str]) -> List[Experience]:
        """
        Parse experience entries from experience section text.
        
        Args:
            experience_text: Text content of experience section
            
        Returns:
            List of Experience objects
        """
        if not experience_text:
            return []
        
        experiences = []
        
        # Split by common delimiters (assumes each job entry is separated)
        # This is a simple heuristic - production would use more sophisticated parsing
        entries = re.split(r'\n\s*\n', experience_text)
        
        for entry in entries:
            entry = entry.strip()
            if len(entry) < 20:  # Skip very short entries
                continue
            
            # Try to extract company, title, dates
            exp = MetadataParser._parse_single_experience_entry(entry)
            if exp:
                experiences.append(exp)
        
        return experiences
    
    @staticmethod
    def _parse_single_experience_entry(entry_text: str) -> Optional[Experience]:
        """
        Parse a single experience entry.
        
        Args:
            entry_text: Text for a single job entry
            
        Returns:
            Experience object or None if parsing fails
        """
        # Extract dates (various formats)
        date_pattern = r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\s*[-–to]+\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|present|current|now)'
        date_match = re.search(date_pattern, entry_text, re.IGNORECASE)
        
        start_date = None
        end_date = None
        current = False
        
        if date_match:
            start_date = date_match.group(1)
            end_date_str = date_match.group(2).lower()
            if end_date_str in ['present', 'current', 'now']:
                current = True
            else:
                end_date = date_match.group(2)
        
        # Extract company (heuristic: first line or line with @ symbol)
        lines = entry_text.split('\n')
        company = None
        title = None
        
        for line in lines[:3]:  # Check first 3 lines
            line = line.strip()
            if '@' in line:
                # Format: "Title @ Company"
                parts = line.split('@')
                if len(parts) == 2:
                    title = parts[0].strip()
                    company = parts[1].strip()
                break
            elif not company and len(line.split()) <= 3:
                # Likely company name (short)
                company = line
            elif not title and any(word in line.lower() for word in ['engineer', 'developer', 'manager', 'analyst', 'director']):
                title = line
        
        # Extract location (heuristic: city names or state abbreviations)
        location_pattern = r'([A-Z][a-z]+,\s*[A-Z]{2}|[A-Z]{2}|Remote|remote)'
        location_match = re.search(location_pattern, entry_text)
        location = location_match.group(1) if location_match else None
        
        # Description is the remaining text
        description = entry_text
        
        return Experience(
            company=company,
            title=title,
            location=location,
            start_date=start_date,
            end_date=end_date,
            description=description,
            current=current
        )
    
    @staticmethod
    def parse_education(education_text: Optional[str]) -> List[Education]:
        """
        Parse education entries from education section text.
        
        Args:
            education_text: Text content of education section
            
        Returns:
            List of Education objects
        """
        if not education_text:
            return []
        
        educations = []
        
        # Split by common delimiters
        entries = re.split(r'\n\s*\n', education_text)
        
        for entry in entries:
            entry = entry.strip()
            if len(entry) < 20:
                continue
            
            edu = MetadataParser._parse_single_education_entry(entry)
            if edu:
                educations.append(edu)
        
        return educations
    
    @staticmethod
    def _parse_single_education_entry(entry_text: str) -> Optional[Education]:
        """
        Parse a single education entry.
        
        Args:
            entry_text: Text for a single education entry
            
        Returns:
            Education object or None if parsing fails
        """
        # Extract degree (common degree keywords)
        degree_keywords = ['bachelor', 'master', 'phd', 'doctorate', 'mba', 'associate']
        degree = None
        for keyword in degree_keywords:
            if keyword in entry_text.lower():
                degree_match = re.search(rf'{keyword}[\'s]?\s*(?:of\s+)?([\w\s]+)', entry_text, re.IGNORECASE)
                if degree_match:
                    degree = degree_match.group(0)
                    break
        
        # Extract institution (heuristic: first line or line with "University", "College", "Institute")
        lines = entry_text.split('\n')
        institution = None
        
        for line in lines[:3]:
            line = line.strip()
            if any(word in line.lower() for word in ['university', 'college', 'institute', 'school']):
                institution = line
                break
            elif not institution and len(line.split()) <= 4:
                institution = line
        
        # Extract field of study
        field_keywords = ['computer science', 'engineering', 'business', 'arts', 'science', 'mathematics']
        field_of_study = None
        for keyword in field_keywords:
            if keyword in entry_text.lower():
                field_of_study = keyword
                break
        
        # Extract dates
        date_pattern = r'(\d{4})\s*[-–to]+\s*(\d{4}|present|current)'
        date_match = re.search(date_pattern, entry_text)
        start_date = None
        end_date = None
        
        if date_match:
            start_date = date_match.group(1)
            end_date_str = date_match.group(2).lower()
            if end_date_str not in ['present', 'current']:
                end_date = date_match.group(2)
        
        # Extract GPA
        gpa_pattern = r'gpa[:\s]*([0-9]\.?\d*)'
        gpa_match = re.search(gpa_pattern, entry_text, re.IGNORECASE)
        gpa = gpa_match.group(1) if gpa_match else None
        
        return Education(
            institution=institution,
            degree=degree,
            field_of_study=field_of_study,
            start_date=start_date,
            end_date=end_date,
            gpa=gpa,
            description=entry_text
        )
    
    @staticmethod
    def parse_projects(projects_text: Optional[str]) -> List[Project]:
        """
        Parse project entries from projects section text.
        
        Args:
            projects_text: Text content of projects section
            
        Returns:
            List of Project objects
        """
        if not projects_text:
            return []
        
        projects = []
        
        # Split by common delimiters
        entries = re.split(r'\n\s*\n', projects_text)
        
        for entry in entries:
            entry = entry.strip()
            if len(entry) < 20:
                continue
            
            # Extract project name (first line)
            lines = entry.split('\n')
            name = lines[0].strip() if lines else None
            
            # Extract technologies (look for common tech keywords)
            technologies = []
            for skill in MetadataParser.SKILL_DATABASE:
                if skill in entry.lower():
                    technologies.append(skill)
            
            projects.append(Project(
                name=name,
                description=entry,
                technologies=technologies
            ))
        
        return projects
    
    @staticmethod
    def parse_certifications(certifications_text: Optional[str]) -> List[Certification]:
        """
        Parse certification entries from certifications section text.
        
        Args:
            certifications_text: Text content of certifications section
            
        Returns:
            List of Certification objects
        """
        if not certifications_text:
            return []
        
        certifications = []
        
        # Split by common delimiters
        entries = re.split(r'\n|,|•', certifications_text)
        
        for entry in entries:
            entry = entry.strip()
            if len(entry) < 5:
                continue
            
            # Extract issuer (common certification authorities)
            issuers = ['aws', 'google', 'microsoft', 'oracle', 'cisco', 'pmp', 'scrum']
            issuer = None
            for issuer_keyword in issuers:
                if issuer_keyword in entry.lower():
                    issuer = issuer_keyword
                    break
            
            certifications.append(Certification(
                name=entry,
                issuer=issuer
            ))
        
        return certifications
    
    @staticmethod
    def parse_languages(languages_text: Optional[str]) -> List[str]:
        """
        Parse languages from languages section text.
        
        Args:
            languages_text: Text content of languages section
            
        Returns:
            List of languages
        """
        if not languages_text:
            return []
        
        # Split by common delimiters
        languages = re.split(r'\n|,|•', languages_text)
        
        # Clean and filter
        cleaned_languages = []
        for lang in languages:
            lang = lang.strip()
            if len(lang) > 2:
                # Remove proficiency indicators
                lang = re.sub(r'\s*\(.*?\)', '', lang)
                lang = re.sub(r'\s*-\s*(fluent|native|intermediate|basic)', '', lang, flags=re.IGNORECASE)
                cleaned_languages.append(lang.strip())
        
        return cleaned_languages
    
    @staticmethod
    def extract_location(text: str) -> str:
        """
        Extract location from resume text.
        
        Args:
            text: Resume text or section text
            
        Returns:
            Extracted location or "Not specified"
        """
        locations = [
            "bangalore", "bengaluru", "mumbai", "delhi", "hyderabad",
            "pune", "chennai", "kolkata", "india", "usa", "uk", "canada",
            "new york", "california", "texas", "florida", "washington",
            "london", "toronto", "vancouver", "sydney", "melbourne",
            "berlin", "paris", "tokyo", "singapore", "dubai", "remote"
        ]
        
        text_lower = text.lower()
        
        for location in locations:
            if location in text_lower:
                return location.title()
        
        return "Not specified"
    
    @staticmethod
    def extract_experience_years(text: str) -> int:
        """
        Extract total years of experience from resume text.
        
        Args:
            text: Resume text or section text
            
        Returns:
            Number of years of experience (0 if not found)
        """
        # Look for patterns like "5 years", "10+ years", "3-5 years"
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*experience)?',
            r'(\d+)\s*-\s*(\d+)\s*years?',
            r'experience\s*:\s*(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if len(match.groups()) == 2 and match.group(2):
                    # Range pattern, take the higher number
                    try:
                        return int(match.group(2))
                    except ValueError:
                        continue
                else:
                    try:
                        return int(match.group(1))
                    except ValueError:
                        continue
        
        return 0
