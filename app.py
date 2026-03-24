import os
import warnings
from dotenv import load_dotenv
import streamlit as st
import re
import html
import time
import pandas as pd
from PyPDF2 import PdfReader
import io
import docx

# Warm embedding model on startup to avoid loading during first query
try:
    from src.embed import load_embedding_model
    _ = load_embedding_model()
except Exception:
    pass

st.set_page_config(page_title="Talentlens", page_icon="🎯", layout="wide")

st.markdown("""
# Talentlens - Resume Intelligence Platform
AI-powered candidate discovery using Retrieval Augmented Generation.

This dashboard helps recruiters ask role-based questions and quickly surface matched candidates with explainability.
""")

# Tabs: Resume Search and Upload & Rank
tab_search, tab_upload = st.tabs(["Resume Search", "Upload & Rank"]) 

# Initialize session state properly
if "shortlist" not in st.session_state:
    st.session_state.shortlist = []
    st.session_state.shortlist_map = {}

if "selected_skills" not in st.session_state:
    st.session_state.selected_skills = []

if "upload_filter_skills" not in st.session_state:
    st.session_state.upload_filter_skills = []

if "search_results" not in st.session_state:
    st.session_state.search_results = []

if "upload_results" not in st.session_state:
    st.session_state.upload_results = []

if "selected_candidate" not in st.session_state:
    st.session_state.selected_candidate = None

# Reset any problematic session state values
if st.session_state.selected_skills and isinstance(st.session_state.selected_skills, list):
    valid_skills = ["SQL", "Python", "AWS", "RAG", "Docker", "Spark", "java", "javascript", "react", "node.js", "ml", "ai", "ci/cd", "postgresql", "mongodb", "spark"]
    st.session_state.selected_skills = [skill for skill in st.session_state.selected_skills if skill.lower() in [v.lower() for v in valid_skills]]


def add_to_shortlist(candidate: dict) -> None:
    ss = st.session_state.get("shortlist", [])
    smap = st.session_state.get("shortlist_map", {})
    cid = candidate.get("id")
    if cid not in ss:
        ss.append(cid)
        smap[cid] = {
            "name": candidate.get("name"),
            "role": candidate.get("role"),
            "score": candidate.get("score")
        }
        st.session_state.shortlist = ss
        st.session_state.shortlist_map = smap
        st.success("Added to shortlist!")


def skill_selector(title, suggested_skills):
    """Modern skill selector with clickable tags like LinkedIn/Figma UI"""
    st.markdown(f"### ⚡ {title}")
    
    # Session state for this component
    session_key = f"{title.lower().replace(' ', '_')}_skills"
    
    # Initialize session state if not exists
    if session_key not in st.session_state:
        st.session_state[session_key] = []
    
    # Get current skills safely
    current_skills = st.session_state.get(session_key, [])
    
    # Search input
    custom_skill = st.text_input("🔍 Add custom skill", placeholder="Type and press Enter...", key=f"custom_{session_key}")
    
    # Add custom skill on Enter (check if new skill was entered)
    if custom_skill and custom_skill.strip() and custom_skill.lower() not in [s.lower() for s in current_skills]:
        st.session_state[session_key] = current_skills + [custom_skill.strip().lower()]
        # Clear the input by setting it to empty
        st.rerun()
    
    # Suggested skills as clickable tags
    st.markdown("#### 🎯 Click to Add Skills")
    
    # Calculate grid layout
    cols_per_row = 6
    for i in range(0, len(suggested_skills), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            skill_idx = i + j
            if skill_idx < len(suggested_skills):
                skill = suggested_skills[skill_idx]
                with cols[j]:
                    # Check if skill is already selected
                    is_selected = skill.lower() in st.session_state[session_key]
                    if is_selected:
                        st.button(f"✅ {skill}", key=f"add_{title}_{skill_idx}_{skill}", disabled=True)
                    else:
                        if st.button(f"+ {skill}", key=f"add_{title}_{skill_idx}_{skill}"):
                            current_skills = st.session_state.get(session_key, [])
                            if skill.lower() not in current_skills:
                                st.session_state[session_key] = current_skills + [skill.lower()]
    
    # Display selected skills as removable chips
    current_skills = st.session_state.get(session_key, [])
    if current_skills:
        st.markdown("#### ✅ Selected Skills")
        
        # Calculate grid for selected skills
        selected_cols_per_row = 8
        for i in range(0, len(current_skills), selected_cols_per_row):
            cols = st.columns(selected_cols_per_row)
            for j in range(selected_cols_per_row):
                skill_idx = i + j
                if skill_idx < len(current_skills):
                    skill = current_skills[skill_idx]
                    with cols[j]:
                        if st.button(f"❌ {skill.title()}", key=f"remove_{title}_{skill_idx}_{skill}"):
                            current_skills = st.session_state.get(session_key, [])
                            if skill in current_skills:
                                current_skills.remove(skill)
                                st.session_state[session_key] = current_skills
    
    return st.session_state.get(session_key, [])


# Define smart suggested skills
SUGGESTED_SKILLS = [
    "Python", "SQL", "AWS", "Docker", "Kubernetes",
    "React", "Node", "Java", "Spring", "C++",
    "Machine Learning", "AI", "Deep Learning",
    "RAG", "LLMs", "Prompt Engineering",
    "Power BI", "Tableau", "Excel",
    "MongoDB", "PostgreSQL", "Git", "CI/CD",
    "HTML", "CSS", "Angular", "Vue", "Django",
    "Flask", "REST API", "GraphQL", "Microservices",
    "Linux", "Azure", "GCP", "Salesforce",
    "Figma", "Photoshop", "Canva", "WordPress",
    "TypeScript", "Next.js", "GraphQL", "Redis"
]


def extract_text(file):
    """Safe text extraction for PDF, DOCX, and TXT files"""
    try:
        file.seek(0)  # ✅ VERY IMPORTANT: Reset file pointer
        
        filename = file.name.lower()
        
        # 📄 PDF
        if filename.endswith(".pdf"):
            pdf_bytes = file.read()
            pdf_stream = io.BytesIO(pdf_bytes)
            
            reader = PdfReader(pdf_stream)
            
            text = ""
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    text += content
            
            return text.lower().strip()
        
        # 📝 DOCX
        elif filename.endswith(".docx"):
            file.seek(0)
            doc = docx.Document(file)
            
            text = "\n".join([p.text for p in doc.paragraphs])
            return text.lower().strip()
        
        # 📃 TXT
        elif filename.endswith(".txt"):
            file.seek(0)
            return file.read().decode("utf-8").lower().strip()
        
        return ""
    
    except Exception as e:
        return f"ERROR: {str(e)}"


def parse_resume(text):
    """Parse resume text into structured data - NEVER CRASHES"""
    try:
        # Use new parser functions
        from src.parser import extract_skills, extract_experience, extract_location, extract_role
        
        skills = extract_skills(text)
        experience = extract_experience(text)
        location = extract_location(text)
        role = extract_role(text)
        
        return {
            "skills": skills,
            "experience": experience,
            "location": location,
            "role": role,
            "text": text
        }
    except Exception as e:
        # Fallback - never crash the app
        return {
            "skills": [],
            "experience": "Not specified",
            "location": "Not specified",
            "role": "Software Developer",
            "text": text
        }


def parse_jd(jd_text):
    """Parse job description into structured data - NEVER CRASHES"""
    try:
        # Use new parser functions
        from src.parser import extract_skills, extract_experience, extract_location
        
        skills = extract_skills(jd_text)
        experience = extract_experience(jd_text)
        location = extract_location(jd_text)
        
        return {
            "skills": skills,
            "experience": experience,
            "location": location,
            "text": jd_text.lower()
        }
    except Exception as e:
        # Fallback - never crash the app
        return {
            "skills": [],
            "experience": 0,
            "location": "Not specified",
            "text": jd_text.lower()
        }


def generate_reasons(candidate, jd_data):
    """Generate match reasons for candidate"""
    reasons = []
    
    # Skill matches
    for skill in jd_data["skills"]:
        if skill in candidate["skills"]:
            reasons.append(f"{skill} match")
    
    # Experience match
    if candidate["experience"] != "Not specified" and jd_data["experience"] != "Not specified":
        try:
            cand_exp = int(re.findall(r"(\d+)", candidate["experience"])[0]) if re.findall(r"(\d+)", candidate["experience"]) else 0
            jd_exp = int(re.findall(r"(\d+)", jd_data["experience"])[0]) if re.findall(r"(\d+)", jd_data["experience"]) else 0
            if cand_exp >= jd_exp:
                reasons.append("Experience match")
        except:
            pass
    
    # Location match
    if (candidate["location"] != "Not specified" and 
        jd_data["location"] != "Not specified" and
        candidate["location"].lower() == jd_data["location"].lower()):
        reasons.append("Location match")
    
    return reasons[:3]


def extract_experience(text):
    """Extract years of experience from resume text"""
    import re
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

def extract_skills_from_text(text):
    """Extract skills from text using comprehensive keyword matching"""
    SKILL_KEYWORDS = [
        "python","sql","aws","docker","kubernetes","react","node","java",
        "machine learning","ai","rag","llm","postgresql","mongodb","git",
        "ci/cd","flask","django","spark","hadoop","excel","power bi",
        "tableau","tensorflow","pytorch","c++","javascript","typescript",
        "angular","vue","spring","rest api","graphql","microservices",
        "linux","azure","gcp","salesforce","jira","html","css"
    ]

    text = text.lower()
    found_skills = [skill for skill in SKILL_KEYWORDS if skill in text]
    
    return list(set(found_skills))

def extract_jd_skills(jd_text):
    """Extract skills from job description"""
    return extract_skills_from_text(jd_text)

def normalize_skills(skills):
    """Normalize skills to lowercase and remove empties"""
    return [s.lower().strip() for s in skills if s and s.strip()]

def compute_match_score(candidate_skills, jd_skills):
    """Fixed scoring function - always returns 0-100%"""
    if not jd_skills:
        return 0, []

    # Normalize both skill lists
    candidate_skills = normalize_skills(candidate_skills)
    jd_skills = normalize_skills(jd_skills)

    matched = list(set(candidate_skills) & set(jd_skills))

    score = (len(matched) / len(jd_skills)) * 100
    score = min(score, 100)

    return round(score, 2), matched

# Smart skill pool for autocomplete
SKILL_POOL = [
    "Python","SQL","AWS","Docker","Kubernetes","React","Node.js","Java",
    "Machine Learning","AI","RAG","LLMs","PostgreSQL","MongoDB","Git",
    "CI/CD","Flask","Django","Spark","Figma","Tableau","Power BI",
    "TypeScript","Next.js","GraphQL","Redis","Azure","GCP","Salesforce",
    "Jira","Excel","JavaScript","Angular","Vue","Spring","C++","HTML",
    "CSS","REST API","Microservices","Linux","Hadoop","Canva","WordPress"
]


def render_candidate_card(candidate, idx):
    """Render clean candidate card"""
    name = candidate.get("name", f"Candidate {idx+1}")
    role = candidate.get("role", "Software Developer")
    location = candidate.get("location", "Not specified")
    experience = candidate.get("experience", "Not specified")
    skills = candidate.get("skills", [])[:6]
    score = candidate.get("score", 0)  # Already a percentage
    matched_skills = candidate.get("matched_skills", [])
    
    st.markdown("---")
    st.markdown(f"### 👤 {name}")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.caption(f"{role} • {location}")
    with col2:
        st.markdown(f"### {score}%")
    
    if skills:
        st.markdown("🧠 " + " ".join([f"`{s}`" for s in skills]))
    
    st.write(f"📅 {experience}")
    
    # 🔥 HIGHLIGHT: Show matched skills prominently
    if matched_skills:
        st.markdown("🎯 **Match Skills:** " + " ".join([f"`{s}`" for s in matched_skills]))
        st.success(f"✅ {len(matched_skills)} skill match{'es' if len(matched_skills) != 1 else ''}")
    else:
        st.caption("❌ No skill matches")
    
    # Match reasons
    reasons = candidate.get("reasons", [])
    if reasons:
        st.caption("💡 " + " | ".join(reasons[:3]))
    
    # Action buttons (fixed to not reset page)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 View", key=f"view_{candidate['id']}"):
            st.session_state.selected_candidate = candidate
            st.rerun()
    with col2:
        if st.button("⭐ Shortlist", key=f"short_{candidate['id']}"):
            if candidate not in st.session_state.shortlist:
                st.session_state.shortlist.append(candidate)
                st.success("Added to shortlist!")
            else:
                st.warning("Already in shortlist")
            st.rerun()


# Resume Search Tab
with tab_search:
    # Shortlist sidebar (interactive)
    with st.sidebar:
        st.subheader("⭐ Shortlist")
        ss = st.session_state.get("shortlist", [])
        smap = st.session_state.get("shortlist_map", {})
        if ss:
            for cid in ss:
                item = smap.get(cid, {})
                score_pct = int((item.get("score") or 0) * 100)
                st.markdown(f"**{item.get('name', 'Unknown')}**")
                st.caption(f"{item.get('role', 'Unknown')} • {score_pct}%")
        else:
            st.caption("No shortlisted candidates yet")

    # Main search form (only contains search inputs)
    with st.form("search_form"):
        user_query = st.text_input(
            "Your query",
            placeholder="E.g., Find senior Python developers with AWS experience in Bangalore",
            help="Describe the role, skills, experience, and location you're looking for."
        )

        st.markdown("### 🎯 Refine Your Search")
        col1, col2, col3 = st.columns(3)
        with col1:
            num_candidates = st.selectbox("Number of Candidates", [5, 10, 15, 20], index=1)
        with col2:
            exp_range = st.slider("Experience (Years)", 0, 20, (2, 6))
        with col3:
            location = st.text_input("Preferred Location", "India")

        submitted = st.form_submit_button("Search 🚀")

    if submitted and user_query.strip():
        # Import here to avoid heavy model loading at Streamlit startup.
        from src.query_pipeline import retrieve, answer

        with st.spinner("Searching candidates and generating answer..."):
            try:
                # ✅ FIXED: Extract skills from user query as fallback
                jd_skills = extract_jd_skills(user_query)
                
                # Ensure JD skills are never empty
                if not jd_skills:
                    st.warning("⚠️ No skills detected. Using default tech skills.")
                    jd_skills = ["python", "sql", "aws"]
                
                # Debug info (important for troubleshooting)
                with st.expander("🔍 Debug: Skill Matching"):
                    st.write("**JD Skills:**", jd_skills)
                    st.write("**User Query:**", user_query)
                
                refined_query = {
                    "text": user_query,
                    "skills": jd_skills,
                    "experience_min": exp_range[0],
                    "experience_max": exp_range[1],
                    "location": location,
                    "num_candidates": num_candidates
                }
                
                retrieved = retrieve(refined_query["text"], top_k=refined_query.get("num_candidates", 10))
                docs = retrieved.get("docs", [])
                
                # Process and score candidates
                scored_results = []
                for i, d in enumerate(docs):
                    meta = d.get("meta", {}) or {}
                    doc = d.get("text", "")
                    
                    # Extract candidate info
                    candidate = {
                        "id": str(i),
                        "name": meta.get("name", f"Candidate {i+1}"),
                        "role": meta.get("role", "Software Developer"),
                        "location": meta.get("location", "Not specified"),
                        "experience": meta.get("experience", "Not specified"),
                        "skills": meta.get("skills", []),
                        "text": doc
                    }
                    
                    # ✅ CRITICAL FIX: Extract skills from resume text if not present
                    if not candidate["skills"] or len(candidate["skills"]) == 0:
                        candidate["skills"] = extract_skills_from_text(doc)
                    
                    # Ensure candidate has at least some skills
                    if not candidate["skills"]:
                        candidate["skills"] = ["unknown"]
                    
                    # Score the candidate with fixed logic
                    score, matched_skills = compute_match_score(
                        candidate["skills"],
                        jd_skills
                    )
                    
                    candidate["score"] = score
                    candidate["matched_skills"] = matched_skills
                    
                    scored_results.append(candidate)
                
                # Sort by score
                scored_results = sorted(scored_results, key=lambda x: x["score"], reverse=True)
                
                # Store in session state
                st.session_state.search_results = scored_results
                
                # Debug: Show first candidate details
                if scored_results:
                    with st.expander("🔍 Debug: Matching Details"):
                        first = scored_results[0]
                        st.write("**JD Skills:**", jd_skills)
                        st.write("**Candidate Name:**", first["name"])
                        st.write("**Candidate Skills:**", first["skills"])
                        st.write("**Matched Skills:**", first["matched_skills"])
                        st.write("**Score:**", first["score"])
                        
                        # Show first 3 candidates for comparison
                        st.write("---")
                        st.write("**First 3 Candidates:**")
                        for i, c in enumerate(scored_results[:3]):
                            st.write(f"{i+1}. {c['name']}: {c['skills']} → {c['matched_skills']} ({c['score']}%)")
                
            except Exception as e:
                st.warning("Some components could not load properly. Please try again.")

    # Always read from session state
    results = st.session_state.search_results
    
    if results:
        st.markdown("### 🎯 Matched Candidates")
        
        # Export option
        df = pd.DataFrame([{
            "name": r["name"],
            "score": r["score"],
            "role": r["role"],
            "matched_skills": ", ".join(r["matched_skills"])
        } for r in results])
        st.download_button("📁 Export Results", df.to_csv(index=False), "candidates.csv")
        
        # Display candidate cards
        for i, candidate in enumerate(results):
            render_candidate_card(candidate, i)
    
    # Display selected candidate if any
    if st.session_state.selected_candidate:
        st.markdown("---")
        st.markdown("### 📄 Resume Details")
        st.json(st.session_state.selected_candidate)
        
        if st.button("Close Resume", key="close_resume"):
            st.session_state.selected_candidate = None
            st.rerun()


# Upload & Rank tab
with tab_upload:
    st.markdown("## 📁 Upload Resumes & Rank Candidates")
    
    # Job Description input (moved above)
    jd = st.text_area(
        "Enter Job Description",
        placeholder="Paste job description here...",
        height=150
    )
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload Resumes",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    
    # Run ranking button
    run_ranking = st.button("🚀 Rank Candidates")
    
    # Process files when button is clicked
    if run_ranking and uploaded_files and jd.strip():
        # 🧹 Remove error spam - validate inputs first
        if not uploaded_files:
            st.warning("Please upload resumes (PDF, DOCX, TXT)")
            st.stop()
        
        if not jd.strip():
            st.warning("Enter Job Description")
            st.stop()
        
        # 🛡️ Filter valid files
        valid_files = [f for f in uploaded_files if f.name.endswith(("pdf", "docx", "txt"))]
        
        if not valid_files:
            st.error("No valid resume files found. Please upload PDF, DOCX, or TXT files.")
            st.stop()
        
        st.success(f"{len(valid_files)} files uploaded successfully")
        
        # 🧪 Add loading state
        with st.spinner("Analyzing resumes..."):
            # Parse job description
            jd_data = parse_jd(jd)
            
            # ✅ FIXED: Extract skills from JD text as fallback
            jd_data["skills"] = extract_jd_skills(jd)
            
            # Ensure JD skills are never empty
            if not jd_data["skills"]:
                st.warning("⚠️ No skills detected in JD. Using default tech skills.")
                jd_data["skills"] = ["python", "sql", "aws"]
            
            # Debug info
            with st.expander("🔍 Debug: Upload Skill Matching"):
                st.write("**JD Skills:**", jd_data["skills"])
                st.write("**JD Text Length:**", len(jd))
            
            scored_results = []
            failed_files = []
            
            # Process each uploaded resume with full error protection
            for file in valid_files:
                try:
                    text = extract_text(file)
                    
                    # ❌ Skip broken files with proper error handling
                    if text.startswith("ERROR") or len(text.strip()) < 50:
                        failed_files.append(file.name)
                        continue
                    
                    # ✅ CRITICAL FIX: Parse resume with proper skill extraction
                    parsed = {
                        "skills": extract_skills_from_text(text),
                        "experience": extract_experience(text) if hasattr(extract_experience, '__call__') else "Not specified",
                        "location": extract_location(text) if hasattr(extract_location, '__call__') else "Not specified",
                        "role": extract_role(text) if hasattr(extract_role, '__call__') else "Software Developer"
                    }
                    
                    # Ensure candidate has at least some skills
                    if not parsed["skills"]:
                        parsed["skills"] = ["unknown"]
                    
                    # Score the candidate
                    score, matched_skills = compute_match_score(
                        parsed["skills"],
                        jd_data["skills"]
                    )
                    
                    candidate = {
                        "id": file.name,
                        "name": file.name,
                        "score": score,
                        "skills": parsed["skills"],
                        "matched_skills": matched_skills,
                        "experience": parsed["experience"],
                        "location": parsed["location"],
                        "role": parsed["role"],
                        "reasons": generate_reasons(parsed, jd_data)
                    }
                    
                    scored_results.append(candidate)
                
                except Exception:
                    # Never let individual file failures break whole pipeline
                    failed_files.append(file.name)
                    continue
            
            # ✅ Sort results by score (descending)
            scored_results = sorted(scored_results, key=lambda x: x["score"], reverse=True)
            
            # Store in session state
            st.session_state.upload_results = scored_results
            
            # Show failed files warning (no red stacktrace)
            if failed_files:
                st.warning(f"⚠️ {len(failed_files)} resumes could not be processed")
                with st.expander("View failed files"):
                    st.write(failed_files)
    
    elif run_ranking:
        if not uploaded_files:
            st.error("Please upload resume PDFs, DOCX, or TXT files first.")
        if not jd.strip():
            st.error("Please enter a job description first.")
    
    # Always read from session state
    results = st.session_state.upload_results
    
    if results:
        st.markdown("### 🎯 Ranked Candidates")
        
        # Export option
        df = pd.DataFrame([{
            "name": r["name"],
            "score": r["score"],
            "role": r["role"],
            "matched_skills": ", ".join(r["matched_skills"])
        } for r in results])
        st.download_button("📁 Export Results", df.to_csv(index=False), "ranked_candidates.csv")
        
        # Display clean candidate cards
        for i, candidate in enumerate(results):
            render_candidate_card(candidate, i)
    
    # Display selected candidate if any
    if st.session_state.selected_candidate:
        st.markdown("---")
        st.markdown("### 📄 Resume Details")
        st.json(st.session_state.selected_candidate)
        
        if st.button("Close Resume", key="close_upload_resume"):
            st.session_state.selected_candidate = None
            st.rerun()
