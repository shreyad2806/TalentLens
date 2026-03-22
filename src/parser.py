import re


def extract_skills(text: str):
    skills = [
        "python","java","c++","sql","machine learning","deep learning",
        "react","node","spring","aws","docker","kubernetes","nlp"
    ]
    if not text:
        return []
    text = text.lower()
    return list({skill for skill in skills if skill in text})


def extract_experience(text: str):
    if not text:
        return "Not specified"
    match = re.search(r"(\d{1,2})\+?\s+years", text.lower())
    if match:
        return f"{match.group(1)}+ years"
    return "Not specified"


def extract_location(text: str):
    if not text:
        return "Not specified"
    locations = ["bangalore","mumbai","delhi","hyderabad","pune","chennai"]
    text = text.lower()
    for loc in locations:
        if loc in text:
            return loc.title()
    return "Not specified"


def extract_role(text: str):
    if not text:
        return "Software Developer"
    lines = text.split("\n")
    for line in lines[:6]:
        if any(x in line.lower() for x in ["engineer","developer","manager","analyst"]):
            return line.strip()
    return "Software Developer"


def extract_candidate_info(text: str, query: str = "") -> dict:
    """Return a compact structured candidate info dict.

    Fields: role, experience, skills (max 6), matched_skills, location
    """
    if not text:
        return {
            "role": "Software Developer",
            "experience": "Not specified",
            "skills": [],
            "matched_skills": [],
            "location": "Not specified",
        }

    text_low = text.lower()

    # EXPERIENCE (first numeric years match)
    exp_match = re.search(r"(\d{1,2})\+?\s*(years|yrs)", text_low)
    experience = f"{exp_match.group(1)}+ years" if exp_match else "Not specified"

    # SKILLS (basic keyword matching)
    skill_keywords = [
        "python",
        "java",
        "sql",
        "aws",
        "docker",
        "kubernetes",
        "react",
        "node",
        "ml",
        "ai",
        "ci/cd",
        "django",
        "flask",
        "spring",
    ]
    skills = [s for s in skill_keywords if s in text_low]

    # ROLE DETECTION (simple)
    role = "Software Engineer"
    if "data scientist" in text_low or "data science" in text_low:
        role = "Data Scientist"
    elif "backend" in text_low or "backend developer" in text_low:
        role = "Backend Developer"
    elif "frontend" in text_low or "frontend developer" in text_low:
        role = "Frontend Developer"
    elif "manager" in text_low:
        role = "Engineering Manager"

    # LOCATION
    locations = ["pune", "delhi", "bangalore", "mumbai", "hyderabad", "chennai"]
    location = next((loc.title() for loc in locations if loc in text_low), "India")

    # MATCHED SKILLS (based on query)
    q_low = (query or "").lower()
    matched_skills = [s for s in skills if s in q_low]

    return {
        "role": role,
        "experience": experience,
        "skills": skills[:6],
        "matched_skills": matched_skills,
        "location": location,
    }
