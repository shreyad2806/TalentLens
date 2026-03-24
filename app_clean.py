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

st.set_page_config(page_title="Resume Intelligence Platform", page_icon="🎯", layout="wide")

st.markdown("""
# Resume Intelligence Platform
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


def get_experience_years(text: str) -> int:
    if not text:
        return 0
    m = re.search(r"(\d{1,2})", str(text))
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return 0
    return 0


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


def compute_match_score(resume, jd):
    """Compute match score between resume and job description"""
    score = 0
    
    # 🔹 Skill match (HIGH weight)
    matched_skills = list(set(resume["skills"]) & set(jd["skills"]))
    skill_score = len(matched_skills) * 15
    
    # 🔹 Experience match
    exp_score = 0
    try:
        resume_exp = int(re.findall(r"(\d+)", resume.get("experience", "0"))[0]) if re.findall(r"(\d+)", resume.get("experience", "0")) else 0
        jd_exp = int(re.findall(r"(\d+)", str(jd.get("experience", "0")))[0]) if re.findall(r"(\d+)", str(jd.get("experience", "0"))) else 0
        if resume_exp >= jd_exp:
            exp_score = 25
        else:
            exp_score = max(0, 15 - abs(resume_exp - jd_exp))
    except Exception:
        pass
    
    # 🔹 Text relevance (keyword overlap)
    keyword_score = sum(1 for word in jd["skills"] if word.lower() in resume.get("text", "").lower()) * 5
    
    total_score = skill_score + exp_score + keyword_score
    
    # Normalize to %
    total_score = min(total_score, 100)
    
    return total_score, matched_skills


def render_candidate_card(candidate, idx):
    """Render clean candidate card"""
    name = candidate.get("name", f"Candidate {idx+1}")
    role = candidate.get("role", "Software Developer")
    location = candidate.get("location", "Not specified")
    experience = candidate.get("experience", "Not specified")
    skills = candidate.get("skills", [])[:6]
    score = int(candidate.get("score", 0))
    
    st.markdown(f"### 👤 {name}")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.caption(f"{role} • {location}")
    with col2:
        st.markdown(f"### {score}%")
    
    if skills:
        st.markdown("🧠 " + " ".join([f"`{s}`" for s in skills]))
    
    st.write(f"📅 {experience}")
    
    # Match reasons
    reasons = candidate.get("reasons", [])
    if reasons:
        st.caption("✔ " + " | ✔ ".join(reasons[:3]))
    
    col1, col2 = st.columns(2)
    with col1:
        st.button("📄 View", key=f"view_{idx}")
    with col2:
        st.button("⭐ Shortlist", key=f"short_{idx}")
    
    st.markdown("---")


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

    # Main search form
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

        st.markdown("### ⚡ Suggested Skills")
        default_skills = ["SQL", "Python", "AWS", "RAG", "Docker", "Spark"]

        # multi-select for skills (works inside form)
        selected = st.multiselect("Select Skills (suggested)", default_skills, default=st.session_state.get("selected_skills", []))
        st.session_state.selected_skills = selected or []

        custom_skill = st.text_input("Add custom skill")

        st.write("Selected Skills:", ", ".join(st.session_state.selected_skills or []))

        submitted = st.form_submit_button("Search 🚀")

    if submitted and user_query.strip():
        # if custom skill provided inside form, add it to selected skills
        if custom_skill.strip():
            st.session_state.selected_skills.append(custom_skill.strip())

        # Import here to avoid heavy model loading at Streamlit startup.
        from src.query_pipeline import retrieve, answer

        with st.spinner("Searching candidates and generating answer..."):
            try:
                refined_query = {
                    "text": user_query,
                    "skills": st.session_state.selected_skills,
                    "experience_min": exp_range[0],
                    "experience_max": exp_range[1],
                    "location": location,
                    "num_candidates": num_candidates
                }
                
                retrieved = retrieve(refined_query["text"], top_k=refined_query.get("num_candidates", 10))
                docs = retrieved.get("docs", [])
                
                # Display results
                if docs:
                    st.markdown("### 🎯 Matched Candidates")
                    
                    # Export option
                    df = pd.DataFrame([{
                        "name": d.get("meta", {}).get("name", f"Candidate {i+1}"),
                        "score": d.get("score", 0),
                        "role": d.get("meta", {}).get("role", "Software Developer")
                    } for i, d in enumerate(docs)])
                    st.download_button("📁 Export Results", df.to_csv(index=False), "candidates.csv")
                    
                    # Display candidate cards
                    for i, d in enumerate(docs):
                        meta = d.get("meta", {}) or {}
                        doc = d.get("text", "")
                        
                        st.markdown("---")
                        col1, col2 = st.columns([4, 1])
                        
                        with col1:
                            st.subheader(f"👤 {meta.get('name', f'Candidate {i+1}')}")
                            st.caption(f"{meta.get('role', 'Software Developer')} • {meta.get('location', 'Not specified')}")
                        
                        with col2:
                            score_pct = int((d.get('score', 0) or 0) * 100)
                            st.metric("Match", f"{score_pct}%")
                        
                        st.write(f"📅 Experience: {meta.get('experience', 'Not specified')}")
                        
                        skills = meta.get("skills", [])[:6]
                        if skills:
                            st.write("🧠 Skills:")
                            st.write(", ".join(skills))
                        
                        matched = meta.get("matched_skills", [])
                        if matched:
                            st.write("✅ Matched Skills:")
                            st.success(", ".join(matched))
                        else:
                            st.caption("No strong match")
                        
                        # Action buttons
                        b1, b2 = st.columns(2)
                        with b1:
                            if st.button("📄 View Resume", key=f"view_{i}"):
                                st.session_state["view_resume_text"] = doc
                        with b2:
                            if st.button("⭐ Shortlist", key=f"short_{i}"):
                                add_to_shortlist({
                                    "id": str(i),
                                    "name": meta.get('name', f'Candidate {i+1}'),
                                    "role": meta.get('role', 'Software Developer'),
                                    "score": d.get('score', 0)
                                })
                else:
                    st.info("No matching candidates found. Try adjusting your search criteria.")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write("Please try again or check your configuration.")


# Upload & Rank tab
with tab_upload:
    st.markdown("## 📁 Upload Resumes & Rank Candidates")
    
    # Job Description input
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
            
            valid_results = []
            failed_files = []
            
            # Process each uploaded resume with full error protection
            for file in valid_files:
                try:
                    text = extract_text(file)
                    
                    # ❌ Skip broken files with proper error handling
                    if text.startswith("ERROR") or len(text.strip()) < 50:
                        failed_files.append(file.name)
                        continue
                    
                    parsed = parse_resume(text)
                    
                    score, matched_skills = compute_match_score(parsed, jd_data)
                    
                    candidate = {
                        "name": file.name,
                        "score": score,
                        "skills": parsed["skills"],
                        "matched_skills": matched_skills,
                        "experience": parsed["experience"],
                        "location": parsed["location"],
                        "role": parsed["role"]
                    }
                    
                    # Add match reasons
                    candidate["reasons"] = generate_reasons(candidate, jd_data)
                    
                    valid_results.append(candidate)
                
                except Exception:
                    # Never let individual file failures break whole pipeline
                    failed_files.append(file.name)
                    continue
            
            # ✅ Sort results by score (descending) - FIXES SAME SCORE ISSUE
            valid_results = sorted(valid_results, key=lambda x: x["score"], reverse=True)
            
            # 🧪 Add debug panel
            if valid_results:
                with st.expander("Debug Info"):
                    st.write("JD Skills:", jd_data["skills"])
                    st.write("JD Experience:", jd_data["experience"])
                    st.write("JD Location:", jd_data["location"])
                    if valid_results:
                        st.write("Sample Resume Skills:", valid_results[0]["skills"])
                        st.write("Sample Resume Experience:", valid_results[0]["experience"])
                        st.write("Sample Resume Location:", valid_results[0]["location"])
            
            # Show failed files warning (no red stacktrace)
            if failed_files:
                st.warning(f"⚠️ {len(failed_files)} resumes could not be processed")
                with st.expander("View failed files"):
                    st.write(failed_files)
            
            # Display results
            if valid_results:
                st.markdown("### 🎯 Ranked Candidates")
                
                # Export option
                df = pd.DataFrame(valid_results)
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "📁 Export Results",
                    csv_data,
                    "ranked_candidates.csv",
                    mime="text/csv"
                )
                
                # Display clean candidate cards
                for idx, candidate in enumerate(valid_results):
                    render_candidate_card(candidate, idx)
            else:
                st.warning("No valid resumes found. Please check your files.")
    
    elif run_ranking:
        if not uploaded_files:
            st.error("Please upload resume PDFs, DOCX, or TXT files first.")
        if not jd.strip():
            st.error("Please enter a job description first.")
    
    # Show upload shortlist if exists
    if "upload_shortlist" in st.session_state and st.session_state.upload_shortlist:
        st.markdown("### ⭐ Upload Shortlist")
        for name in st.session_state.upload_shortlist:
            st.write(f"📄 {name}")
