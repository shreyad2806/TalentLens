import os
import re
from collections import Counter
from typing import Iterable, Union
import pandas as pd


def load_data(csv_path: str = "Resume/Resume.csv") -> pd.DataFrame:
    """Load resume CSV safely. Tries provided path first, then resolves relative to project root.

    Returns an empty DataFrame on failure.
    """
    # Resolve path relative to repository root if needed
    if not os.path.isabs(csv_path) and not os.path.exists(csv_path):
        base = os.path.dirname(os.path.dirname(__file__))
        alt = os.path.join(base, csv_path)
        csv_path = alt if os.path.exists(alt) else csv_path

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame()

    # If there's no explicit Resume/text column, try to pick a sensible column
    if "Resume" not in df.columns:
        # pick first object (string) column if available
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        if obj_cols:
            df.rename(columns={obj_cols[0]: "Resume"}, inplace=True)

    # Ensure Resume column exists and drop rows without resume text
    if "Resume" in df.columns:
        df = df[df["Resume"].notna()]

    return df


def _to_iterable(resumes: Union[pd.Series, Iterable]) -> Iterable:
    if resumes is None:
        return []
    if hasattr(resumes, "fillna"):
        return resumes.fillna("")
    return resumes


def extract_skills(resumes: Union[pd.Series, Iterable]) -> Counter:
    """Extract common skills and return counts as a Counter."""
    skills = [
        "python", "java", "c++", "sql", "machine learning", "deep learning",
        "react", "node", "spring", "aws", "docker", "kubernetes", "nlp"
    ]
    counter = Counter()

    for text in _to_iterable(resumes):
        txt = str(text).lower()
        for skill in skills:
            if skill in txt:
                counter[skill] += 1

    return counter


def extract_locations(resumes: Union[pd.Series, Iterable]) -> Counter:
    """Find known city/location mentions in resumes and return counts."""
    pattern = r"\b(bangalore|bengaluru|blr|mumbai|delhi|hyderabad|pune|chennai|kolkata|bangalore)\b"
    counter = Counter()

    for text in _to_iterable(resumes):
        matches = re.findall(pattern, str(text).lower())
        for loc in matches:
            # normalize Bengaluru/blr to bangalore/bengaluru
            if loc == "blr":
                loc = "bengaluru"
            counter[loc] += 1

    return counter


def category_distribution(df: pd.DataFrame) -> Union[pd.Series, dict]:
    """Return value counts for candidate categories if present."""
    if df is None or df.empty:
        return pd.Series([], dtype=int)

    for col in ["Category", "category", "Role", "role"]:
        if col in df.columns:
            return df[col].value_counts()

    return pd.Series([], dtype=int)
