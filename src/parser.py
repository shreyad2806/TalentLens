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
