from enum import Enum
from typing import Literal
import re

from .config import CATEGORIES

class CategoryEnum(str, Enum):
    HR = "HR"
    DESIGNER = "DESIGNER"
    INFORMATION_TECHNOLOGY = "INFORMATION-TECHNOLOGY"
    TEACHER = "TEACHER"
    ADVOCATE = "ADVOCATE"
    BUSINESS_DEVELOPMENT = "BUSINESS-DEVELOPMENT"
    HEALTHCARE = "HEALTHCARE"
    FITNESS = "FITNESS"
    AGRICULTURE = "AGRICULTURE"
    BPO = "BPO"
    SALES = "SALES"
    CONSULTANT = "CONSULTANT"
    DIGITAL_MEDIA = "DIGITAL-MEDIA"
    AUTOMOBILE = "AUTOMOBILE"
    CHEF = "CHEF"
    FINANCE = "FINANCE"
    APPAREL = "APPAREL"
    ENGINEERING = "ENGINEERING"
    ACCOUNTANT = "ACCOUNTANT"
    CONSTRUCTION = "CONSTRUCTION"
    PUBLIC_RELATIONS = "PUBLIC-RELATIONS"
    BANKING = "BANKING"
    ARTS = "ARTS"
    AVIATION = "AVIATION"


def extract_location(text):
    """Extract location from resume text"""
    text = text.lower()
    
    # Common location patterns and countries
    locations = [
        "india", "usa", "united states", "uk", "united kingdom", "canada", 
        "germany", "australia", "france", "japan", "singapore", "dubai",
        "bangalore", "bengaluru", "mumbai", "delhi", "hyderabad", "chennai",
        "new york", "california", "texas", "florida", "washington",
        "london", "manchester", "birmingham", "toronto", "vancouver",
        "sydney", "melbourne", "berlin", "munich", "paris", "tokyo"
    ]
    
    # Check for location mentions
    for loc in locations:
        if loc in text:
            return loc
    
    return "unknown"


def extract_experience(text):
    """Extract years of experience from resume text"""
    text = text.lower()
    
    # Look for patterns like "5 years", "10+ years", "3-5 years", etc.
    patterns = [
        r'(\d+)\+?\s*years?',
        r'(\d+)\s*-\s*(\d+)\s*years?',
        r'experience\s*:\s*(\d+)',
        r'(\d+)\s*years?\s*of\s*experience'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            if match.groups():
                # For range patterns, take the higher number
                if len(match.groups()) == 2:
                    return int(match.group(2))
                else:
                    return int(match.group(1))
    
    return 0


def classify_category(user_query: str) -> str:
    # Simple keyword-based classification as fallback
    query_lower = user_query.lower()
    
    # Simple keyword matching for categories
    category_keywords = {
        "INFORMATION-TECHNOLOGY": ["software", "developer", "programmer", "java", "python", "javascript", "web", "api", "database", "it", "tech", "engineering"],
        "HR": ["hr", "human resources", "recruiter", "staffing", "personnel"],
        "DESIGNER": ["design", "ui", "ux", "graphic", "creative", "artist"],
        "TEACHER": ["teacher", "education", "professor", "academic", "school"],
        "ADVOCATE": ["lawyer", "legal", "advocate", "attorney"],
        "BUSINESS-DEVELOPMENT": ["business", "sales", "marketing", "revenue"],
        "HEALTHCARE": ["doctor", "medical", "nurse", "health", "hospital"],
        "FINANCE": ["finance", "accounting", "bank", "financial"],
        "SALES": ["sales", "selling", "revenue", "customer"],
        "ENGINEERING": ["engineer", "mechanical", "civil", "electrical"],
    }
    
    # Count keyword matches for each category
    category_scores = {}
    for category, keywords in category_keywords.items():
        score = sum(1 for keyword in keywords if keyword in query_lower)
        category_scores[category] = score
    
    # Return category with highest score, default to first category
    if max(category_scores.values()) > 0:
        return max(category_scores, key=category_scores.get)
    else:
        return CATEGORIES[0]  # Default to HR if no matches


def compute_candidate_score(candidate, query):
    """Compute candidate score with proper location filtering"""
    text = candidate.get("text", "").lower()
    score = 0
    
    # Extract location from candidate data
    candidate_location = extract_location(text)
    candidate["location"] = candidate_location
    
    # Extract query components
    query_lower = query.lower()
    
    # Skills matching
    common_skills = ["java", "python", "javascript", "react", "node.js", "sql", 
                    "aws", "docker", "kubernetes", "machine learning", "ai",
                    "data science", "web development", "mobile development"]
    
    skill_matches = sum(1 for skill in common_skills if skill in query_lower and skill in text)
    score += skill_matches * 10
    
    # Experience matching
    exp = extract_experience(text)
    if exp > 0:
        # Give bonus for relevant experience
        if exp >= 2:
            score += 15
        if exp >= 5:
            score += 10
    
    # ✅ FIXED LOCATION LOGIC
    if candidate_location != "unknown":
        # Extract location from query
        query_location = extract_location(query)
        if query_location != "unknown":
            if query_location in candidate_location or candidate_location in query_location:
                score += 15  # Bonus for matching location
            else:
                score -= 10  # Penalize wrong location
    
    return score