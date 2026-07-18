import os
import sys
import platform
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

from src.debug_logger import log_stage_start, log_stage_end, log_error

st.set_page_config(page_title="Talentlens", page_icon="🎯", layout="wide")

# ── Cached resource factories (lazy — only build on first call) ────────────

@st.cache_resource(show_spinner=False)
def _get_embedding_model():
    """Load embedding model exactly once."""
    from src.embed import load_embedding_model
    return load_embedding_model()

@st.cache_resource(show_spinner=False)
def _get_retrieval_bundle():
    """Build retrieval bundle exactly once (bootstrap, vector store, BM25, etc.)."""
    from src.bootstrap.composition_root import create_retrieval_bundle
    return create_retrieval_bundle()

# ── STAGE 1 — APPLICATION STARTUP (runs once per session) ──────────────────

_t_startup_total = time.perf_counter()
_t_config_end = None

if "_stage1_done" not in st.session_state:
    from src.config import EMBEDDING_MODEL, EMBEDDING_DIM

    _t_config = time.perf_counter()
    _t_config_end = _t_config

    log_stage_start(1, "APPLICATION STARTUP",
        Python_Version=sys.version.split()[0],
        OS=platform.system(),
        OS_Version=platform.version(),
        Embedding_Model=EMBEDDING_MODEL,
        Embedding_Dim=EMBEDDING_DIM,
        Vector_Store_Provider=os.getenv("VECTOR_STORE_PROVIDER", "qdrant"),
        Resume_Folder=os.getenv("RESUME_FOLDER", "Resume/"),
        Index_Folder=os.getenv("INDEX_FOLDER", "data/indexes"),
        Cache_Folder=os.getenv("CACHE_FOLDER", "data/cache"),
    )
    log_stage_end(1, "APPLICATION STARTUP",
        status="SUCCESS",
        time_ms=(_t_config - _t_startup_total) * 1000,
        sample={
            "Python": sys.version.split()[0],
            "Platform": platform.system(),
            "Model": EMBEDDING_MODEL,
            "Dim": EMBEDDING_DIM,
        },
        extra={"Config": "Loaded from .env + defaults"},
    )
    st.session_state._stage1_done = True

# ── Startup Timing Table ────────────────────────────────────────────────────
_t_ui_render = time.perf_counter()
_startup_total_ms = (_t_ui_render - _t_startup_total) * 1000

with st.expander("⏱ Startup Timing", expanded=False):
    _timing_rows = [
        {"Phase": "Configuration (.env + imports)", "Time (ms)": f"{(_t_config - _t_startup_total) * 1000:.1f}" if _t_config_end else "cached"},
        {"Phase": "Stage 1 banner", "Time (ms)": f"{(_t_config - _t_startup_total) * 1000:.1f}" if _t_config_end else "cached"},
        {"Phase": "Embedding model", "Time (ms)": "deferred (lazy, cached)"},
        {"Phase": "Retrieval bundle", "Time (ms)": "deferred (lazy, cached)"},
        {"Phase": "UI rendering", "Time (ms)": f"{(_t_ui_render - _t_config) * 1000:.1f}" if _t_config_end else f"{(_t_ui_render - _t_startup_total) * 1000:.1f}"},
        {"Phase": "TOTAL startup", "Time (ms)": f"{_startup_total_ms:.1f}"},
    ]
    st.table(pd.DataFrame(_timing_rows))

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
            "score": candidate.get("score"),
            "match_pct": candidate.get("match_pct", 0),
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
            doc = docx.Document (file)
            
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
    score = candidate.get("score", 0)          # rrf_score (retrieval ranking)
    match_pct = candidate.get("match_pct", 0)  # skill-match % (supplementary)
    matched_skills = candidate.get("matched_skills", [])
    email = candidate.get("email")
    phone = candidate.get("phone")
    
    st.markdown("---")
    st.markdown(f"### 👤 {name}")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.caption(f"{role} • {location}")
        # Show contact info if available
        contact_parts = []
        if email:
            contact_parts.append(f"📧 {email}")
        if phone:
            contact_parts.append(f"📞 {phone}")
        if contact_parts:
            st.caption(" • ".join(contact_parts))
    with col2:
        # Show retrieval score (rrf_score) as primary ranking signal
        st.markdown(f"### {score:.4f}")
        st.caption("RRF score")
    
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
                score_pct = item.get("match_pct", 0)
                st.markdown(f"**{item.get('name', 'Unknown')}**")
                st.caption(f"{item.get('role', 'Unknown')} • RRF {item.get('score', 0):.4f}")
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
        _ui_start = time.perf_counter()
        log_stage_start(12, "UI OUTPUT", Query=user_query[:80], Top_K=num_candidates)

        with st.spinner("Searching candidates and generating answer..."):
            try:
                # Use cached retrieval bundle (built once, reused on every search)
                bundle = _get_retrieval_bundle()

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
                
                # ── STAGE 4 — METADATA FILTER PARSING ────────────────────────────
                _s4_start = time.perf_counter()
                log_stage_start(4, "METADATA FILTER PARSING",
                    Extracted_Skills=jd_skills,
                    Experience_Range=f"{exp_range[0]}-{exp_range[1]} years",
                    Location=location,
                    Education="(not extracted)",
                    Company="(not extracted)",
                    Role="(not extracted)",
                )
                log_stage_end(4, "METADATA FILTER PARSING",
                    status="SUCCESS",
                    time_ms=(time.perf_counter() - _s4_start) * 1000,
                    output_count=len(jd_skills),
                    sample={"Skills": jd_skills, "Location": location, "Exp": f"{exp_range[0]}-{exp_range[1]}"},
                    extra={"Note": "Filters extracted in UI layer; not passed to retrieval pipeline"},
                )
                
                # Use cached bundle for hybrid search (no re-creation)
                _t_retrieve = time.perf_counter()
                hybrid_results = bundle.hybrid_service.search(
                    query=refined_query["text"],
                    top_k=refined_query.get("num_candidates", 10),
                )
                _t_retrieve_ms = (time.perf_counter() - _t_retrieve) * 1000

                # ── Score mapping: HybridSearchResult → docs[] ────────────────
                print()
                print("[SCORE TRACE] Mapping HybridSearchResult → docs[]")
                print("[META TRACE] Mapping HybridSearchResult → docs[]")
                if hybrid_results:
                    print(f"  BEFORE: {len(hybrid_results)} HybridSearchResult objects")
                    print(f"  BEFORE: top rrf_score = {hybrid_results[0].rrf_score:.6f}  "
                          f"({hybrid_results[0].candidate_name})")
                    _h0 = hybrid_results[0]
                    print(f"  BEFORE META: keys={list(_h0.metadata.keys()) if _h0.metadata else '[]'}")
                    print(f"  BEFORE META: candidate_name={_h0.candidate_name}, "
                          f"resume_id={_h0.resume_id}")
                else:
                    print("  BEFORE: no hybrid results")

                # Convert HybridSearchResult to legacy dict format
                # Enrich metadata with top-level fields so downstream code can find everything in meta
                docs = []
                for r in hybrid_results:
                    # Merge top-level fields into metadata for unified downstream access
                    enriched_meta = dict(r.metadata) if r.metadata else {}
                    enriched_meta.setdefault("candidate_name", r.candidate_name)
                    enriched_meta.setdefault("resume_id", r.resume_id)
                    enriched_meta.setdefault("section", r.section)

                    docs.append({
                        "id": r.resume_id,
                        "text": enriched_meta.get("text", ""),
                        "resume": enriched_meta.get("text", ""),
                        "score": r.rrf_score,
                        "section": r.section,
                        "candidate_name": r.candidate_name,
                        "chunk_id": r.chunk_id,
                        "metadata": enriched_meta,
                    })

                if docs:
                    print(f"  AFTER:  {len(docs)} docs, top score = {docs[0]['score']:.6f}")
                    print(f"  AFTER META: keys={list(docs[0]['metadata'].keys())}")
                
                # ── Score mapping: docs[] → scored_results ─────────────────────
                print()
                print("[SCORE TRACE] Mapping docs[] → scored_results")
                print("[META TRACE] Mapping docs[] → scored_results")
                scored_results = []
                for i, d in enumerate(docs):
                    meta = d.get("metadata", {}) or {}
                    doc_text = d.get("text", "")
                    rrf_score = d.get("score", 0.0)          # retrieval score from pipeline

                    if i < 3:  # log first 3 for meta trace
                        print(f"  [{i}] meta keys = {list(meta.keys())}")
                    
                    # Extract candidate info from metadata (all fields now propagated)
                    candidate = {
                        "id": str(i),
                        "resume_id": d.get("id", ""),
                        "name": meta.get("candidate_name") or d.get("candidate_name", f"Candidate {i+1}"),
                        "role": meta.get("role") or "Software Developer",
                        "location": meta.get("location") or "Not specified",
                        "experience": meta.get("experience") or "Not specified",
                        "skills": meta.get("skills") or [],
                        "email": meta.get("email"),
                        "phone": meta.get("phone"),
                        "summary": meta.get("summary"),
                        "text": doc_text,
                    }
                    
                    # Extract skills from resume text if metadata had none
                    if not candidate["skills"]:
                        candidate["skills"] = extract_skills_from_text(doc_text)
                    if not candidate["skills"]:
                        candidate["skills"] = ["unknown"]
                    
                    if i < 3:  # log first 3 for meta trace
                        print(f"  [{i}] name={candidate['name']}, role={candidate['role']}, "
                              f"location={candidate['location']}, skills={candidate['skills'][:3]}, "
                              f"email={candidate.get('email')}")
                    
                    # Skill overlap (supplementary — used for matched_skills display)
                    _skill_match_pct, matched_skills = compute_match_score(
                        candidate["skills"],
                        jd_skills,
                    )
                    
                    # PRESERVE rrf_score as the primary ranking score
                    candidate["score"] = rrf_score
                    candidate["match_pct"] = _skill_match_pct
                    candidate["matched_skills"] = matched_skills
                    
                    if i < 5:  # log first 5 for trace
                        print(f"  [{i}] {candidate['name']:<25}  rrf={rrf_score:.6f}  "
                              f"skill_match={_skill_match_pct:.1f}%  "
                              f"matched={matched_skills}")
                    
                    scored_results.append(candidate)
                
                # Sort by retrieval score (descending)
                scored_results = sorted(scored_results, key=lambda x: x["score"], reverse=True)
                
                if scored_results:
                    print(f"  AFTER:  top candidate = {scored_results[0]['name']}, "
                          f"score = {scored_results[0]['score']:.6f}")
                    _c0 = scored_results[0]
                    print(f"  AFTER META: name={_c0['name']}, role={_c0.get('role')}, "
                          f"location={_c0.get('location')}, skills={_c0.get('skills', [])[:3]}, "
                          f"email={_c0.get('email')}, phone={_c0.get('phone')}")
                
                # ── STAGE 11 — FINAL RANKING ──────────────────────────────────────
                _s11_start = time.perf_counter()
                log_stage_start(11, "FINAL RANKING", Total_Candidates=len(scored_results))
                
                # Print ranking table
                if scored_results:
                    print(f"  {'Rank':<6}{'Candidate':<25}{'RRF':<12}{'Skill%':<10}")
                    print(f"  {'-'*53}")
                    for rank_i, c in enumerate(scored_results[:10]):
                        print(f"  {rank_i+1:<6}{c['name']:<25}{c['score']:<12.6f}{c.get('match_pct',0):<10.1f}")
                
                _s11_sample = None
                if scored_results:
                    _s11_sample = {
                        "Rank_1": scored_results[0]["name"],
                        "RRF_Score": f"{scored_results[0]['score']:.6f}",
                        "Skill_Match": f"{scored_results[0].get('match_pct', 0):.1f}%",
                    }
                
                log_stage_end(11, "FINAL RANKING",
                    status="SUCCESS",
                    time_ms=(time.perf_counter() - _s11_start) * 1000,
                    output_count=len(scored_results),
                    sample=_s11_sample,
                    extra={"Scoring": "RRF retrieval score (primary) + skill-match % (supplementary)"},
                )
                
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
                log_error(12, "UI OUTPUT", e, reraise=False)
                st.warning("Some components could not load properly. Please try again.")

        # ── STAGE 12 — UI OUTPUT ─────────────────────────────────────────────
        _ui_elapsed = (time.perf_counter() - _ui_start) * 1000
        _results_count = len(st.session_state.get("search_results", []))
        if _results_count > 0:
            log_stage_end(12, "UI OUTPUT",
                status="SUCCESS",
                time_ms=_ui_elapsed,
                output_count=_results_count,
                sample={
                    "Top_Candidate": st.session_state.search_results[0].get("name", "N/A") if _results_count else "N/A",
                    "Top_Score": st.session_state.search_results[0].get("score", 0) if _results_count else 0,
                },
                extra={"Cards_Rendered": _results_count, "Total_Latency_ms": f"{_ui_elapsed:.1f}"},
            )

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
