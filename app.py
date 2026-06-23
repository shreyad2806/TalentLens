print("STEP 1 Loading Config")
import os
import warnings
from dotenv import load_dotenv
load_dotenv() # Load environment variables

print("STEP 2 Loading UI")
import streamlit as st
st.set_page_config(page_title="Talentlens", page_icon="🎯", layout="wide")

import re
import html
import time
import pandas as pd
from PyPDF2 import PdfReader
import io
import docx

print("STEP 3 Loading Retrieval Services")

@st.cache_resource
def get_metadata_service():
    try:
        from src.retrieval.metadata import MetadataService
        return MetadataService(cache_enabled=True)
    except Exception as e:
        print(f"Warning: Metadata service failed to load: {e}")
        return None

@st.cache_resource
def get_hybrid_service():
    try:
        from src.retrieval.dense import DenseRetrievalService
        from src.retrieval.sparse import SparseRetrievalService, BM25Index
        from src.retrieval.hybrid import HybridRetrievalService
        from src.indexing.pipeline import IndexingPipeline
        from pathlib import Path
        
        dense = DenseRetrievalService()
        
        # Try to load BM25 index from persistent storage
        bm25_index_path = Path("data/indexes/bm25")
        if (bm25_index_path / "metadata.json").exists():
            print("Loading BM25 index from persistent storage for retrieval...")
            bm25_index = BM25Index()
            bm25_index.load_from_disk(bm25_index_path)
            print(f"BM25 loaded: {bm25_index.num_documents} documents")
        else:
            print("BM25 index not found, creating empty index")
            bm25_index = BM25Index()
        
        sparse = SparseRetrievalService(index=bm25_index)
        return HybridRetrievalService(
            dense_retrieval_service=dense,
            sparse_retrieval_service=sparse
        )
    except Exception as e:
        print(f"Warning: Hybrid service failed to load: {e}")
        return None

@st.cache_resource
def get_reranker_service():
    try:
        from src.retrieval.reranker import RerankerService
        return RerankerService()
    except Exception as e:
        print(f"Warning: Reranker service failed to load: {e}")
        return None

metadata_service = get_metadata_service()
hybrid_service = get_hybrid_service()
reranker_service = get_reranker_service()

print("STEP 4 Bootstrap System")
@st.cache_resource
def get_bootstrap_service():
    try:
        from src.bootstrap import BootstrapService
        return BootstrapService(verbose=True)
    except Exception as e:
        print(f"Warning: Bootstrap service failed to load: {e}")
        return None

@st.cache_resource
def run_bootstrap():
    """Run bootstrap once per session with logging and timing metrics."""
    import time
    from src.bootstrap import BootstrapService
    from src.vector_store.qdrant.qdrant_adapter import QdrantAdapter
    from src.vector_store.config import VectorStoreConfig, VectorStoreProvider
    
    print("STARTUP HEALTH CHECK")
    start_time = time.time()
    
    # Check vector store configuration
    config = VectorStoreConfig()
    print(f"Vector Store Provider: {config.provider.value}")
    
    # Check Qdrant if it's the provider
    if config.provider == VectorStoreProvider.QDRANT:
        try:
            qdrant_adapter = QdrantAdapter()
            health_status = qdrant_adapter.health_check()
            
            print(f"Qdrant URL: {qdrant_adapter.url}")
            print(f"Qdrant Collection: {qdrant_adapter.collection_name}")
            print(f"Qdrant Connected: {health_status.connection_healthy}")
            print(f"Qdrant Collection Exists: {health_status.collection_exists}")
            print(f"Qdrant Vector Count: {health_status.vector_count}")
            
            # If collection exists and has vectors, skip bootstrap
            if health_status.connection_healthy and health_status.collection_exists and health_status.vector_count > 0:
                elapsed_time = time.time() - start_time
                print(f"✓ INDEX FOUND - Skipping bootstrap (startup time: {elapsed_time:.2f}s)")
                
                # Load BM25 index from cache
                import pickle
                from pathlib import Path
                bm25_cache_path = Path("data/cache/bm25.pkl")
                if bm25_cache_path.exists():
                    print(f"✓ Loading BM25 index from cache...")
                    with open(bm25_cache_path, 'rb') as f:
                        bm25_index = pickle.load(f)
                    print(f"✓ BM25 index loaded")
                else:
                    print(f"⚠ BM25 index cache not found at {bm25_cache_path}")
                
                return {"bootstrapped": False, "vector_count": health_status.vector_count}
            
        except Exception as e:
            print(f"⚠ Qdrant health check failed: {e}")
            print("Continuing with bootstrap...")
    
    # Run bootstrap if Qdrant check failed or collection doesn't exist
    bootstrap_service = None
    try:
        bootstrap_service = BootstrapService(verbose=True)
    except Exception as e:
        print(f"Warning: Bootstrap service failed to load: {e}")
        return None
    
    if not bootstrap_service:
        return None
    
    print("BOOTSTRAP START")
    bootstrap_start_time = time.time()
    
    try:
        bootstrap_result = bootstrap_service.bootstrap()
        bootstrap_elapsed_time = time.time() - bootstrap_start_time
        total_elapsed_time = time.time() - start_time
        
        if not bootstrap_result.get('bootstrapped', True):
            print(f"BOOTSTRAP SKIPPED - index already contains data (bootstrap time: {bootstrap_elapsed_time:.2f}s, total time: {total_elapsed_time:.2f}s)")
        else:
            print(f"BOOTSTRAP COMPLETE (bootstrap time: {bootstrap_elapsed_time:.2f}s, total time: {total_elapsed_time:.2f}s)")
            print("⚠ INDEX NOT FOUND - Run 'python scripts/build_index.py' to build the production index")
        
        return bootstrap_result
    except Exception as e:
        bootstrap_elapsed_time = time.time() - bootstrap_start_time
        total_elapsed_time = time.time() - start_time
        print(f"BOOTSTRAP FAILED: {e} (bootstrap time: {bootstrap_elapsed_time:.2f}s, total time: {total_elapsed_time:.2f}s)")
        import traceback
        traceback.print_exc()
        return None

# Run bootstrap once per session
bootstrap_result = run_bootstrap()

print("STEP 5 Rendering Streamlit")

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
        skills = extract_skills_from_text(text)
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
        skills = extract_jd_skills(jd_text)
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
                
                # Create MetadataFilter directly from UI fields
                from src.retrieval.metadata import MetadataFilter
                try:
                    m_filter = MetadataFilter(
                        minimum_experience=exp_range[0],
                        maximum_experience=exp_range[1],
                        location=location if location and location.strip() else None,
                        skills=jd_skills if jd_skills else None
                    )
                    hybrid_filters = m_filter.model_dump(exclude_none=True)
                except Exception as e:
                    hybrid_filters = None
                    print(f"Filter validation failed: {e}")
                
                if not hybrid_service:
                    st.warning("⚠️ Hybrid Retrieval Service unavailable. Check backend.")
                    st.stop()
                
                if not reranker_service:
                    st.warning("⚠️ Cross Encoder Reranker unavailable. Check backend.")
                    st.stop()
                
                # 1. Hybrid Retrieval
                retrieved = hybrid_service.search(
                    query=user_query,
                    top_k=num_candidates * 3, # Fetch more for reranker
                    filters=hybrid_filters
                )
                
                if not retrieved:
                    st.info("No candidates found matching criteria.")
                    st.session_state.search_results = []
                else:
                    # 2. Cross Encoder Reranker
                    reranked = reranker_service.rerank(
                        query=user_query,
                        candidates=retrieved,
                        top_k=num_candidates
                    )
                    
                    # 3. Map to UI Schema
                    scored_results = []
                    for i, r in enumerate(reranked):
                        # Extract skills either from metadata or text
                        candidate_skills = r.metadata.get("skills", [])
                        if not candidate_skills or len(candidate_skills) == 0:
                            candidate_skills = extract_skills_from_text(r.matched_text)
                        
                        if not candidate_skills:
                            candidate_skills = ["unknown"]
                        
                        # Match skills for highlighting
                        _, matched_skills = compute_match_score(candidate_skills, jd_skills)
                        
                        # Scale score to percentage
                        score_pct = int(max(0.0, min(1.0, r.rerank_score)) * 100) if r.rerank_score <= 1.0 else int(r.rerank_score)
                        
                        # Format experience
                        exp_val = r.metadata.get("minimum_experience")
                        exp_str = f"{exp_val} years" if exp_val is not None else "Not specified"
                        
                        candidate = {
                            "id": f"{i}_{r.resume_id}",
                            "name": r.candidate_name,
                            "role": r.section if r.section else "Software Developer",
                            "location": r.metadata.get("location", "Not specified"),
                            "experience": exp_str,
                            "skills": candidate_skills,
                            "text": r.matched_text,
                            "score": score_pct,
                            "matched_skills": matched_skills,
                            "reasons": [
                                f"Cross-Encoder Score: {r.rerank_score:.2f}",
                                f"Hybrid Rank: {r.original_rank}"
                            ]
                        }
                        scored_results.append(candidate)
                    
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
                            
                            st.write("---")
                            st.write("**First 3 Candidates:**")
                            for i, c in enumerate(scored_results[:3]):
                                st.write(f"{i+1}. {c['name']}: {c['skills']} → {c['matched_skills']} ({c['score']}%)")
                
            except Exception as e:
                st.warning(f"Search failed: {str(e)}")

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
