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

    # call to ensure model resource is initialized (cached_resource)
    _ = load_embedding_model()
except Exception:
    # don't block startup if model cannot be loaded here
    pass

# Precompute resume embeddings if not present (will be cached on disk)
try:
    from src.analytics import load_data
    from src.embed import get_or_create_resume_embeddings

    df_res = load_data()
    if df_res is not None and not df_res.empty:
        # this will load persisted embeddings or compute+persist if missing
        embs, emb_ids = get_or_create_resume_embeddings(df_res)
        # expose counts in sidebar for visibility
        st.sidebar.write(f"🗂 Resumes indexed: {len(embs)}")
except Exception:
    # non-fatal; if persistence unavailable, the app still runs
    pass


# Suppress warnings to avoid threading issues
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Fix tokenizer threading issues

load_dotenv()

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
    # Filter out any invalid values
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
            "experience": candidate.get("experience"),
            "score": float(candidate.get("score") or 0),
        }
        st.session_state["shortlist"] = ss
        st.session_state["shortlist_map"] = smap


def remove_from_shortlist(candidate_id: str) -> None:
    ss = st.session_state.get("shortlist", [])
    smap = st.session_state.get("shortlist_map", {})
    ss = [c for c in ss if c != candidate_id]
    smap.pop(candidate_id, None)
    st.session_state["shortlist"] = ss
    st.session_state["shortlist_map"] = smap


# Fallback extractor helpers (module-level, with type hints) to satisfy static analysis
def fallback_extract_skills(text: str) -> list:
    return []


def fallback_extract_experience(text: str) -> str:
    return "Not specified"


def fallback_extract_location(text: str) -> str:
    return "Not specified"


def fallback_extract_role(text: str) -> str:
    return "Software Developer"


def get_skill_suggestions(query: str):
    base_skills = [
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
        "postgresql",
        "mongodb",
        "spark",
        "rag",
    ]
    q = (query or "").lower()
    suggestions = [s for s in base_skills if s in q]
    # append a few defaults
    for s in base_skills[:5]:
        if s not in suggestions:
            suggestions.append(s)
    return suggestions


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
        # Use existing extract functions from tools
        from src.tools import extract_skills, extract_experience, extract_location
        
        skills = extract_skills(text)
        experience = extract_experience(text)
        location = extract_location(text)
        
        return {
            "skills": skills,
            "experience": experience,
            "location": location,
            "text": text
        }
    except Exception as e:
        # Fallback - never crash the app
        return {
            "skills": [],
            "experience": "Not found",
            "location": "unknown",
            "text": text
        }


def parse_jd(jd_text):
    """Parse job description into structured data - NEVER CRASHES"""
    try:
        jd_text = jd_text.lower()
        
        # Use existing extract functions from tools
        from src.tools import extract_skills, extract_experience
        
        skills = extract_skills(jd_text)
        experience = extract_experience(jd_text)
        
        return {
            "skills": skills,
            "experience": experience,
            "text": jd_text
        }
    except Exception as e:
        # Fallback - never crash the app
        return {
            "skills": [],
            "experience": 0,
            "text": jd_text.lower()
        }


def compute_match_score(resume, jd):
    """Compute match score between resume and job description"""
    score = 0
    
    # 🔹 Skill match (HIGH weight)
    matched_skills = list(set(resume["skills"]) & set(jd["skills"]))
    skill_score = len(matched_skills) * 15
    
    # 🔹 Experience match
    exp_score = 0
    try:
        resume_exp = get_experience_years(resume.get("experience", "0"))
        jd_exp = get_experience_years(str(jd.get("experience", "0")))
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
                st.write(f"{item.get('name','Unknown')} ({score_pct}%)")
                if st.button("Remove", key=f"remove_{cid}"):
                    remove_from_shortlist(cid)
        else:
            st.caption("No shortlisted candidates")

    with st.form("query_form"):
        user_query = st.text_area("Your question", height=120, placeholder="e.g., Find senior Java developers with Spring experience in Bangalore")

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

        # Get current selected skills from session state
        current_skills = st.session_state.get("selected_skills", [])
        
        # Ensure current skills are in the options (handle session state issues)
        available_skills = list(set(default_skills + current_skills))
        
        # multi-select for skills (works inside form)
        selected = st.multiselect("Select Skills (suggested)", available_skills, default=current_skills)
        st.session_state.selected_skills = selected or []

        custom_skill = st.text_input("Add custom skill")

        st.write("Selected Skills:", ", ".join(st.session_state.selected_skills or []))

        submitted = st.form_submit_button("Search 🚀")

    if submitted and user_query.strip():
        # if custom skill provided inside form, add it to selected skills
        try:
            if custom_skill:
                if custom_skill not in st.session_state.selected_skills:
                    st.session_state.selected_skills.append(custom_skill)
        except Exception:
            pass
        # Build structured query from pre-search controls
        structured_query = {
            "text": user_query,
            "skills": st.session_state.selected_skills or [],
            "experience_min": int(exp_range[0]) if isinstance(exp_range, (list, tuple)) else int(exp_range),
            "experience_max": int(exp_range[1]) if isinstance(exp_range, (list, tuple)) else int(exp_range),
            "location": location,
            "num_candidates": int(num_candidates),
        }

        # Compose a refined textual query for retrieval/reranking
        refined_query = f"{user_query} " + " ".join(structured_query["skills"]) + f" location:{structured_query['location']} exp:{structured_query['experience_min']}-{structured_query['experience_max']}"

        # Import here to avoid heavy model loading at Streamlit startup.
        from src.query_pipeline import retrieve, answer

        with st.spinner("Searching candidates and generating answer..."):
            try:
                # pass refined_query and set top_k=num_candidates to retrieval
                retrieved = retrieve(refined_query, top_k=structured_query.get("num_candidates", 10))
                category = retrieved.get("category")
                docs = retrieved.get("docs", [])

                # --- Candidate helpers ---
                def _safe_list(x):
                    if x is None:
                        return []
                    if isinstance(x, (list, tuple)):
                        return list(x)
                    # comma separated
                    return [s.strip() for s in str(x).split(",") if s.strip()]

                def _normalize_score(s):
                    try:
                        f = float(s)
                    except Exception:
                        return 0.0
                    if f > 1:
                        # assume percentage
                        f = max(0.0, min(100.0, f)) / 100.0
                    return max(0.0, min(1.0, f))

                def _extract_location_from_text(text):
                    if not text:
                        return None
                    patt = r"\b(bangalore|bengaluru|blr|mumbai|delhi|hyderabad|pune|chennai|kolkata)\b"
                    m = re.search(patt, str(text).lower())
                    if m:
                        loc = m.group(1)
                        if loc == "blr":
                            return "Bengaluru"
                        return loc.title()
                    return None

                def build_candidate(doc, score, idx=0):
                    # doc is a dict with text and metadata
                    meta = (doc.get("meta") or {})
                    text = doc.get("text") or ""

                    name = meta.get("name") or doc.get("name") or f"Candidate {idx+1}"

                    # Use parser for robust extraction
                    try:
                        from src.parser import extract_skills, extract_experience, extract_location, extract_role
                    except Exception:
                        # fallbacks: map to module-level helpers with explicit type signatures
                        extract_skills = fallback_extract_skills
                        extract_experience = fallback_extract_experience
                        extract_location = fallback_extract_location
                        extract_role = fallback_extract_role

                    role = meta.get("role") or meta.get("title") or extract_role(text)
                    
                    # ✅ FIXED: Use location from doc if available (extracted by query pipeline)
                    location = doc.get("location") or meta.get("location") or meta.get("city") or extract_location(text)
                    
                    skills = meta.get("skills") or meta.get("keywords") or []
                    if isinstance(skills, (str,)):
                        skills = [s.strip() for s in skills.split(",") if s.strip()]
                    if not skills:
                        skills = extract_skills(text)

                    experience = meta.get("experience") or meta.get("years") or extract_experience(text)

                    summary = (meta.get("summary") or meta.get("short") or str(text)[:250]).strip()

                    # score expected 0-1 float
                    try:
                        score_val = float(score)
                    except Exception:
                        score_val = 0.0

                    return {
                        "id": doc.get("id") or meta.get("id") or f"doc_{idx}",
                        "name": name,
                        "role": role,
                        "location": location,
                        "skills": list(skills),
                        "experience": experience,
                        "summary": summary,
                        "score": score_val,
                        "raw_text": text,
                    }

                def render_candidate_card(candidate, idx):
                                    # Build safe display values
                                    name = candidate.get("name") or f"Candidate {idx+1}"
                                    role = candidate.get("role") or "Software Developer"
                                    location = candidate.get("location") or "Not specified"
                                    experience = candidate.get("experience") or "Not specified"
                                    score_val = float(candidate.get("score", 0) or 0)
                                    score_percent = int(round(score_val * 100))

                                    # Summary (truncate)
                                    text_src = candidate.get("summary") or candidate.get("raw_text") or ""
                                    summary = text_src if len(text_src) <= 250 else text_src[:250].rsplit(" ", 1)[0] + "..."

                                    # Color emoji logic
                                    if score_percent >= 80:
                                        color_emoji = "🟢"
                                    elif score_percent >= 60:
                                        color_emoji = "🟡"
                                    else:
                                        color_emoji = "🔴"

                                    candidate_id = str(candidate.get("id") or idx)

                                    # CARD LAYOUT (NO HTML)
                                    with st.container():
                                        st.markdown("---")

                                        col1, col2 = st.columns([3, 1])

                                        with col1:
                                            st.subheader(f"👤 {name}")
                                            st.write(f"💼 {role}")
                                            # ✅ FIXED: Show location prominently with proper formatting
                                            if location != "unknown" and location != "Not specified":
                                                st.write(f"📍 Location: {location.title()}")
                                            else:
                                                st.write(f"📍 Location: Not specified")

                                        with col2:
                                            st.metric(label="Match", value=f"{score_percent}%", delta=color_emoji)

                                        st.write(f"📅 Experience: {experience}")
                                        st.write("🧠 Summary:")
                                        st.caption(summary)

                                        # Buttons (unique keys using candidate id)
                                        b1, b2 = st.columns(2)
                                        with b1:
                                            if st.button("📄 View Resume", key=f"view_{candidate_id}_{idx}"):
                                                st.session_state["view_resume_id"] = candidate_id
                                                st.session_state["view_resume_text"] = candidate.get("raw_text", "")
                                        with b2:
                                            if st.button("⭐ Shortlist", key=f"short_{candidate_id}_{idx}"):
                                                ss = st.session_state.get("shortlist", [])
                                                smap = st.session_state.get("shortlist_map", {})
                                                if candidate_id not in ss:
                                                    ss.append(candidate_id)
                                                    smap[candidate_id] = {"name": name, "role": role}
                                                    st.session_state["shortlist"] = ss
                                                    st.session_state["shortlist_map"] = smap
                                                    st.success("Added to shortlist")
                                                else:
                                                    st.info("Already shortlisted")


                # Category badge
                st.markdown(
                    f"<div style='display:inline-block;padding:6px 12px;border-radius:12px;background:#EEF6FF;color:#1D4ED8;font-weight:600;'>Category: {category}</div>",
                    unsafe_allow_html=True,
                )

                # Answer and Top Candidates side-by-side (minimal recruiter UI)
                left_col, right_col = st.columns([1, 3])

                # Build structured candidates once and enforce limits
                candidates = [build_candidate(d, (d.get("score") or 0) / 100.0, i) for i, d in enumerate(docs[:10])]
                # truncate summaries and skills per rules
                for c in candidates:
                    if c.get("summary"):
                        c["summary"] = (c["summary"][:80] + "...") if len(c["summary"]) > 80 else c["summary"]
                    if c.get("skills") and isinstance(c.get("skills"), list):
                        c["skills"] = c["skills"][:5]

                top_candidates = candidates[:10]

                # compute global top skills (simple frequency)
                from collections import Counter

                skill_counter = Counter()
                for c in candidates:
                    for s in (c.get("skills") or []):
                        if s:
                            skill_counter[s] += 1
                global_top_skills = [s for s, _ in skill_counter.most_common(6)]

                # LEFT: Search summary (compact, metrics only) + recruiter controls
                with left_col:
                    st.subheader("🔍 Search Summary")
                    st.write(f"**Query:** {user_query}")
                    st.metric("Candidates Found", len(top_candidates))
                    avg_score = 0
                    if top_candidates:
                        try:
                            avg_score = int(sum([c.get("score", 0) for c in top_candidates]) / len(top_candidates) * 100)
                        except Exception:
                            avg_score = 0
                    st.metric("Avg Match", f"{avg_score}%")

                    st.markdown("### Top Skills")
                    st.caption(", ".join(global_top_skills or []))

                    st.markdown("---")
                    st.markdown("### Recruiter Controls")

                    num_candidates = st.selectbox(
                        "Number of Candidates",
                        [5, 10, 15, 20],
                        index=1,
                    )

                    st.markdown("### 📅 Experience Range")
                    min_exp, max_exp = st.slider(
                        "Years of Experience",
                        0,
                        20,
                        (0, 5),
                    )

                    # AI skill suggestions and selected skills
                    st.markdown("### 💡 Suggested Skills")
                    suggestions = get_skill_suggestions(user_query)
                    cols = st.columns(5)
                    for i, skill in enumerate(suggestions):
                        col = cols[i % 5]
                        if col.button(f"+ {skill}", key=f"skill_add_{skill}"):
                            if skill not in st.session_state.selected_skills:
                                st.session_state.selected_skills.append(skill)

                    # Show selected skills with remove buttons
                    if st.session_state.selected_skills:
                        st.markdown("**Selected Skills:**")
                        rem_cols = st.columns(5)
                        for i, sk in enumerate(list(st.session_state.selected_skills)):
                            with rem_cols[i % 5]:
                                st.write(sk)
                                if st.button("Remove", key=f"skill_remove_{sk}"):
                                    st.session_state.selected_skills = [s for s in st.session_state.selected_skills if s != sk]

                    # Reset filters
                    if st.button("Reset Filters"):
                        st.session_state.selected_skills = []
                        # reset sliders by rerunning with defaults
                        min_exp, max_exp = 0, 5

                    st.markdown("---")
                    st.success("Top candidates ranked based on skills and relevance")

                # build refined query using selected skills
                refined_query = user_query + " " + " ".join(st.session_state.selected_skills or [])

                # Scoring-based ranking (avoid hard AND filters)
                def compute_candidate_score(candidate: dict, q: dict) -> tuple:
                    text = (candidate.get("raw_text") or "").lower()
                    score = 0.0
                    matched = []

                    # Skill score (HIGH weight)
                    for sk in (q.get("skills") or []):
                        if sk and sk.lower() in text:
                            score += 10
                            matched.append(sk)

                    # Experience score
                    yrs = get_experience_years(candidate.get("raw_text", "") or candidate.get("summary", ""))
                    exp_min = q.get("experience_min", 0)
                    exp_max = q.get("experience_max", 100)
                    try:
                        if exp_min <= yrs <= exp_max:
                            score += 15
                        else:
                            score += max(0, 10 - abs(yrs - exp_min))
                    except Exception:
                        pass

                    # ✅ FIXED LOCATION LOGIC
                    query_location = (q.get("location") or "").lower()
                    candidate_location = (candidate.get("location") or "").lower()
                    
                    if query_location and candidate_location:
                        if query_location in candidate_location or candidate_location in query_location:
                            score += 15  # Bonus for matching location
                        else:
                            score -= 10  # Penalize wrong location
                    elif query_location and query_location in text:
                        # Fallback to text search if location not extracted
                        score += 5

                    # Query relevance (keyword match)
                    qtext = (q.get("text") or "").lower()
                    if qtext and qtext in text:
                        score += 20

                    # include original vector score as small boost
                    try:
                        orig = float(candidate.get("score", 0) or 0) * 100
                        score += orig * 0.05
                    except Exception:
                        pass

                    return score, matched

                scored = []
                for c in top_candidates:
                    s, matched = compute_candidate_score(c, {
                        "skills": st.session_state.selected_skills or [],
                        "experience_min": min_exp,
                        "experience_max": max_exp,
                        "location": location or "",
                        "text": refined_query,
                    })
                    c["_raw_score"] = s
                    c["_matched_skills"] = list(dict.fromkeys(matched))
                    scored.append(c)

                # sort by score desc
                scored_sorted = sorted(scored, key=lambda x: x.get("_raw_score", 0), reverse=True)

                # safety: if fewer candidates than requested, return what we have
                display_candidates = scored_sorted[:int(num_candidates)] if scored_sorted else top_candidates[:int(num_candidates)]

                # normalize to percent based on max observed
                max_raw = max([c.get("_raw_score", 0) for c in display_candidates]) if display_candidates else 0
                for c in display_candidates:
                    raw = c.get("_raw_score", 0)
                    pct = int(round((raw / max_raw) * 100)) if max_raw > 0 else 0
                    c["_match_pct"] = pct

                # show active filters
                with left_col:
                    st.markdown("### 🎯 Active Filters")
                    if st.session_state.selected_skills:
                        st.write("Skills:", ", ".join(st.session_state.selected_skills or []))
                    st.write(f"Experience: {min_exp} - {max_exp} years")
                    st.write(f"Showing top {num_candidates} candidates")

                # RIGHT: Clean candidate cards (structured view)
                with right_col:
                    st.subheader("Top Candidates")
                    if not display_candidates:
                        st.warning("No matching candidates found")
                    else:
                        for i, c in enumerate(display_candidates[:20]):
                            info = c.get("_info")
                            if info is None:
                                try:
                                    from src.parser import extract_candidate_info as _eci

                                    info = _eci(c.get("raw_text", ""), user_query)
                                except Exception:
                                    info = {
                                        "role": c.get("role", "Software Engineer"),
                                        "experience": c.get("experience", "Not specified"),
                                        "skills": c.get("skills", []),
                                        "matched_skills": [],
                                        "location": c.get("location", "Not specified"),
                                    }

                            # enforce UX rules: max 6 skills, one-line summary
                            skills_display = (info.get("skills") or [])[:6]
                            summary_line = (c.get("summary") or "")
                            summary_line = summary_line.replace("\n", " ")
                            if len(summary_line) > 100:
                                summary_line = summary_line[:100].rsplit(" ", 1)[0] + "..."

                            name = c.get("name") or f"Candidate {i+1}"
                            role = info.get("role") or c.get("role") or "Software Engineer"
                            # ✅ FIXED: Use location from candidate if available (extracted by query pipeline)
                            location = c.get("location") or info.get("location") or "Not specified"
                            experience = info.get("experience") or c.get("experience") or "Not specified"
                            # prefer calculated match percent if available
                            score_pct = int(c.get("_match_pct", int(round(float(c.get("score", 0) or 0) * 100))))

                            with st.container():
                                col1, col2 = st.columns([4, 1])
                                with col1:
                                    st.markdown(f"### 👤 {name}")
                                    # ✅ FIXED: Show location prominently with proper formatting
                                    if location != "unknown" and location != "Not specified":
                                        st.caption(f"{role} • 📍 {location.title()}")
                                    else:
                                        st.caption(f"{role} • 📍 Location not specified")
                                with col2:
                                    st.metric("Match", f"{score_pct}%")

                                g1, g2, g3 = st.columns(3)
                                with g1:
                                    st.markdown("**💼 Experience**")
                                    st.write(experience)
                                with g2:
                                    st.markdown("**🧠 Skills**")
                                    st.write(", ".join(skills_display) if skills_display else "—")
                                with g3:
                                    st.markdown("**🎯 Match Skills**")
                                    matched_sk = c.get("_matched_skills") or info.get("matched_skills") or []
                                    if matched_sk:
                                        st.success(", ".join(matched_sk))
                                    else:
                                        st.caption("No strong match")

                                # one-line summary only
                                if summary_line:
                                    st.caption(summary_line)

                                # Actions: View Details (shows full resume only on demand) + Shortlist
                                b1, b2 = st.columns(2)
                                with b1:
                                    if st.button("View Details", key=f"view_details_{i}"):
                                        st.session_state["view_resume_text"] = c.get("raw_text", "")
                                        st.session_state["view_resume_name"] = name
                                with b2:
                                    if st.button("⭐ Shortlist", key=f"short_{i}"):
                                        add_to_shortlist(c)

                                st.markdown("---")

                    # Show full resume only when requested
                    if st.session_state.get("view_resume_text"):
                        vname = st.session_state.get("view_resume_name", "Candidate")
                        with st.expander(f"Full Resume — {vname}"):
                            st.text(st.session_state.get("view_resume_text"))

                # Developer debug / processing trace (collapsed) - no raw texts
                with st.expander("🐛 Debug / Processing Trace"):
                    trace = retrieved.get("trace", {})
                    st.markdown("**Trace / steps**")
                    try:
                        st.json(trace)
                    except Exception:
                        st.write(trace)

                    st.markdown("---")
                    st.markdown("**Retrieved documents (developer view)**")
                    try:
                        for i, d in enumerate(docs[:10], start=1):
                            st.markdown(f"**{i}. ID: {d.get('id')}  |  Score: {d.get('score')}**")
                            meta = d.get('meta', {}) or {}
                            if meta:
                                st.json(meta)
                            # do NOT display full resume text or raw document content here
                            st.caption("Full document text hidden in developer view")
                            st.markdown("---")
                    except Exception:
                        st.write([{"id": d.get('id'), "score": d.get('score')} for d in docs])
                
            # Analytics rendering will be available in the Analytics tab (below)
                
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
        
        with st.spinner("Processing resumes and computing match scores..."):
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
                    
                    valid_results.append({
                        "name": file.name,
                        "score": score,
                        "skills": parsed["skills"],
                        "matched_skills": matched_skills,
                        "experience": parsed["experience"]
                    })
                
                except Exception:
                    # Never let individual file failures break the whole pipeline
                    failed_files.append(file.name)
                    continue
            
            # ✅ Sort results by score (descending) - FIXES SAME SCORE ISSUE
            valid_results = sorted(valid_results, key=lambda x: x["score"], reverse=True)
            
            # 🧪 Add debug panel
            if valid_results:
                with st.expander("Debug Info"):
                    st.write("JD Skills:", jd_data["skills"])
                    st.write("JD Experience:", jd_data["experience"])
                    if valid_results:
                        st.write("Sample Resume Skills:", valid_results[0]["skills"])
                        st.write("Sample Resume Experience:", valid_results[0]["experience"])
            
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
                
                # Display candidate cards with clean UI
                for r in valid_results:
                    st.markdown("---")
                    
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.subheader(f"👤 {r['name']}")
                        
                        # ✨ Optional: Show file type badge
                        file_type = r['name'].split(".")[-1].upper()
                        st.caption(f"📎 {file_type} file")
                        
                        st.write(f"💼 Experience: {r['experience']} yrs")
                        
                        st.write("🧠 Skills:")
                        st.write(", ".join(r["skills"][:5]))
                        
                        st.write("✅ Matched Skills:")
                        if r["matched_skills"]:
                            st.success(", ".join(r["matched_skills"]))
                        else:
                            st.caption("No strong match")
                        
                    with col2:
                        st.metric("Match", f"{r['score']}%")
                    
                    # Shortlist button
                    if st.button("⭐ Shortlist", key=f"upload_shortlist_{r['name']}"):
                        # Add to shortlist
                        if "upload_shortlist" not in st.session_state:
                            st.session_state.upload_shortlist = []
                        
                        if r['name'] not in st.session_state.upload_shortlist:
                            st.session_state.upload_shortlist.append(r['name'])
                            st.success(f"Added {r['name']} to shortlist!")
                        else:
                            st.info(f"{r['name']} already in shortlist")
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


