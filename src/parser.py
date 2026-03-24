import re

def extract_skills(text):
    """Extract skills from resume text"""
    skills_db = [
        "python", "java", "sql", "aws", "docker", "kubernetes",
        "react", "node", "spring", "ml", "ai", "tensorflow",
        "pytorch", "excel", "tableau", "power bi", "c++",
        "javascript", "postgresql", "mongodb", "git", "ci/cd",
        "html", "css", "angular", "vue", "django", "flask",
        "rest api", "graphql", "microservices", "linux", "ubuntu",
        "windows", "azure", "gcp", "salesforce", "jira"
    ]
    
    text = text.lower()
    return [skill for skill in skills_db if skill in text]


def extract_experience(text):
    """Extract years of experience from resume text"""
    matches = re.findall(r"(\d+)\+?\s+years", text.lower())
    if matches:
        return max(matches) + " years"
    return "Not specified"


def extract_location(text):
    """Extract location from resume text"""
    locations = [
        "bangalore", "mumbai", "delhi", "hyderabad",
        "pune", "chennai", "india", "usa", "uk", "remote",
        "new york", "california", "texas", "florida",
        "london", "toronto", "vancouver", "sydney"
    ]
    
    text = text.lower()
    
    for loc in locations:
        if loc in text:
            return loc.title()
    
    return "Not specified"


def extract_role(text):
    """Extract role from resume text"""
    lines = text.split("\n")[:5]
    
    for line in lines:
        if any(x in line.lower() for x in ["engineer", "developer", "scientist", "manager", "analyst"]):
            return line.strip()
    
    return "Software Developer"
