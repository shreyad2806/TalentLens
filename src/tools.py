from enum import Enum
from typing import Literal

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