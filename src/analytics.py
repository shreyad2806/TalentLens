import os
import re
from collections import Counter
import pandas as pd


def load_data(csv_path=None):
    if csv_path is None:
        base = os.path.dirname(os.path.dirname(__file__))
        csv_path = os.path.join(base, "Resume", "Resume.csv")

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        # return empty dataframe with expected columns
        df = pd.DataFrame()
    return df


def extract_skills(resumes):
    skills_list = []

    common_skills = [
        "python", "java", "c++", "sql", "machine learning", "deep learning",
        "react", "node", "spring", "aws", "docker", "kubernetes", "nlp"
    ]

    for text in resumes.fillna("") if hasattr(resumes, 'fillna') else resumes:
        text = str(text).lower()
        for skill in common_skills:
            if skill in text:
                skills_list.append(skill)

    return Counter(skills_list)


def extract_locations(resumes):
    locations = []
    pattern = r"(bangalore|mumbai|delhi|hyderabad|pune|chennai|blr|bengaluru)"

    for text in resumes.fillna("") if hasattr(resumes, 'fillna') else resumes:
        matches = re.findall(pattern, str(text).lower())
        locations.extend(matches)

    return Counter(locations)


def category_distribution(df):
    if df is None or df.empty:
        return pd.Series([], dtype=int)
    # Try common column names
    for col in ["Category", "category", "Role", "role"]:
        if col in df.columns:
            return df[col].value_counts()
    # fallback: no category column
    return pd.Series([], dtype=int)
