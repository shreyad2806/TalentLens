import os
import warnings
from dotenv import load_dotenv
import streamlit as st
import re
import html
import time

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
# Tabs: Resume Search and Analytics Dashboard
tab_search, tab_analytics = st.tabs(["Resume Search", "Analytics Dashboard"]) 
if "shortlist" not in st.session_state:
    st.session_state.shortlist = []
    st.session_state.shortlist_map = {}


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
        submitted = st.form_submit_button("Search 🚀")

    if submitted and user_query.strip():
        # Import here to avoid heavy model loading at Streamlit startup.
        from src.query_pipeline import retrieve, answer

        with st.spinner("Searching candidates and generating answer..."):
            try:
                retrieved = retrieve(user_query)
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
                    location = meta.get("location") or meta.get("city") or extract_location(text)
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
                                            st.write(f"📍 {location}")

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

                # LEFT: Search summary (compact, metrics only)
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
                    st.caption(", ".join(global_top_skills))

                    # minimal success note replacing verbose AI-recommendation
                    st.success("Top candidates ranked based on skills and relevance")

                # RIGHT: Clean candidate cards
                with right_col:
                    st.subheader("Top Candidates")
                    if not top_candidates:
                        st.warning("No matching candidates found")
                    else:
                        for i, c in enumerate(top_candidates):
                            with st.container():
                                col1, col2 = st.columns([4, 1])
                                with col1:
                                    st.markdown(f"### {c.get('name','Candidate')}")
                                    st.caption(f"{c.get('role','')} • {c.get('location','Not specified')}")
                                with col2:
                                    st.metric("Match", f"{int((c.get('score') or 0)*100)}%")

                                # Skills (only top 5)
                                if c.get("skills"):
                                    skills_line = " ".join([f"`{s}`" for s in (c.get("skills") or [])])
                                    st.markdown(skills_line)

                                # Experience and one-line summary
                                st.caption(f"📅 {c.get('experience','Not specified')}")
                                st.caption(c.get('summary',''))

                                # Actions
                                b1, b2 = st.columns(2)
                                with b1:
                                    st.button("View Profile", key=f"view_{i}")
                                with b2:
                                    if st.button("⭐ Shortlist", key=f"short_{i}"):
                                        add_to_shortlist(c)

                                st.markdown("---")

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

# Analytics tab (outside of submit flow)
with tab_analytics:
    try:
        import plotly.express as px  # type: ignore[reportMissingImports]
    except Exception:
        st.warning("Plotly is not installed — install with `pip install plotly` to enable Analytics charts.")
    else:
        try:
            from src.analytics import load_data, extract_skills, extract_locations, category_distribution

            st.title("📊 Analytics Dashboard")

            @st.cache_data
            def get_data():
                return load_data()

            with st.spinner("Loading analytics..."):
                df = get_data()

            if df is None or df.empty:
                st.info("No resume dataset found at Resume/Resume.csv")
            else:
                text_col = "Resume" if "Resume" in df.columns else next((c for c in df.columns if df[c].dtype == object), df.columns[0])

                skills = extract_skills(df[text_col])
                locations = extract_locations(df[text_col])
                categories = category_distribution(df)

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Top Skills")
                    if skills:
                        fig = px.bar(
                            x=list(skills.keys()),
                            y=list(skills.values()),
                            labels={"x": "Skill", "y": "Count"},
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No skills detected in the dataset.")

                with col2:
                    st.subheader("Top Locations")
                    if locations:
                        fig = px.bar(
                            x=list(locations.keys()),
                            y=list(locations.values()),
                            labels={"x": "Location", "y": "Count"},
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No locations detected in the dataset.")

                st.subheader("Candidate Categories")
                if getattr(categories, "empty", False) or (isinstance(categories, dict) and not categories):
                    st.info("No category data available.")
                else:
                    if isinstance(categories, dict):
                        names = list(categories.keys())
                        values = list(categories.values())
                    else:
                        names = list(categories.index)
                        values = list(categories.values)

                    fig = px.pie(values=values, names=names)
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Analytics module error: {e}")


