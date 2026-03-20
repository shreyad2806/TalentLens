import os
import warnings
from dotenv import load_dotenv
import streamlit as st
import re


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

with tab_search:
    # Shortlist sidebar (persistent within session)
    with st.sidebar:
        st.header("⭐ Shortlist")
        ss = st.session_state.get("shortlist", [])
        smap = st.session_state.get("shortlist_map", {})
        if not ss:
            st.write("No shortlisted candidates.")
        else:
            for sid in list(ss):
                item = smap.get(sid, {})
                st.markdown(f"- {item.get('name', sid)} — {item.get('role', '')}")
                if st.button("Remove", key=f"remove_{sid}"):
                    ss.remove(sid)
                    smap.pop(sid, None)
                    st.session_state["shortlist"] = ss
                    st.session_state["shortlist_map"] = smap
                    st.experimental_rerun()

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

                def _extract_experience_from_text(text):
                    if not text:
                        return "Not specified"
                    # look for patterns like '5 years', '3+ years', '10 yrs'
                    m = re.search(r"(\d{1,2})\s*(?:\+)?\s*(?:years|yrs|year)", str(text).lower())
                    if m:
                        return f"{m.group(1)}+ years"
                    return "Not specified"

                def build_candidate_object(doc, idx=0):
                    meta = (doc.get("meta") or {})
                    text = doc.get("text") or ""

                    name = meta.get("name") or doc.get("name") or f"Candidate {idx+1}"
                    # role: meta -> title -> first line of resume -> fallback
                    role = meta.get("role") or meta.get("title")
                    if not role and text:
                        first_line = str(text).strip().splitlines()
                        role = first_line[0] if first_line else None
                    role = role or "Software Developer"

                    location = meta.get("location") or meta.get("city") or _extract_location_from_text(text) or "Not specified"

                    raw_skills = meta.get("skills") or meta.get("keywords") or doc.get("skills") or ""
                    skills = _safe_list(raw_skills)
                    # if no explicit skills, try lightweight extraction from text
                    if not skills and text:
                        from src.analytics import extract_skills as _ext_skills

                        c = _ext_skills([text])
                        skills = list(c.keys())

                    experience = meta.get("experience") or meta.get("years") or _extract_experience_from_text(text)

                    summary = (meta.get("summary") or meta.get("short") or str(text)[:250]).strip()

                    score = _normalize_score(doc.get("score") or meta.get("score") or 0)

                    return {
                        "id": doc.get("id") or meta.get("id") or f"doc_{idx}",
                        "name": name,
                        "role": role,
                        "location": location,
                        "skills": skills,
                        "experience": experience,
                        "summary": summary,
                        "score": score,
                        "raw_text": text,
                    }

                def render_candidate_card(candidate, idx):
                    # Build safe display values
                    name = candidate.get("name") or f"Candidate {idx+1}"
                    role = candidate.get("role") or "Software Developer"
                    location = candidate.get("location") or "Not specified"
                    score = int(round(float(candidate.get("score", 0)) * 100))
                    skills = candidate.get("skills") or []
                    experience = candidate.get("experience") or "Not specified"
                    summary = str(candidate.get("summary") or "")
                    if len(summary) > 220:
                        summary = summary[:220].rsplit(" ", 1)[0] + "..."

                    # Score color logic
                    if score >= 80:
                        color = "#10b981"  # green-500
                    elif score >= 60:
                        color = "#f59e0b"  # amber-500
                    else:
                        color = "#ef4444"  # red-500

                    candidate_id = str(candidate.get("id") or idx)

                    st.markdown(
                        f"""
                    <div style="background-color:#0b1220;padding:16px;border-radius:12px;margin-bottom:14px;border:1px solid #1f2937;">
                        <div style="display:flex;align-items:center;justify-content:space-between;">
                            <div style="flex:1">
                                <h4 style="margin:0;color:#f3f4f6;">👤 {name}</h4>
                                <div style="color:#cbd5e1;margin-top:6px;">💼 {role} • 📍 {location}</div>
                            </div>
                            <div style="text-align:right;min-width:110px;margin-left:12px;">
                                <div style="font-size:20px;color:{color};font-weight:700;">⭐ {score}%</div>
                                <div style="color:#94a3b8;font-size:12px;">Match Score</div>
                            </div>
                        </div>

                        <div style="margin-top:10px;color:#e6eef8;">
                            <b>Skills:</b> {(' | '.join(skills)) if skills else 'N/A'}
                        </div>
                        <div style="margin-top:6px;color:#cbd5e1;"> <b>Experience:</b> {experience} </div>
                        <div style="margin-top:8px;color:#9ca3af;">{summary}</div>

                        <div style="margin-top:12px;display:flex;gap:8px;">
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    c1, c2 = st.columns([1, 1])
                    with c1:
                        if st.button("📄 View Resume", key=f"view_{candidate_id}_{idx}"):
                            st.session_state["view_resume_id"] = candidate_id
                            st.session_state["view_resume_text"] = candidate.get("raw_text", "")
                    with c2:
                        if st.button("⭐ Shortlist", key=f"shortlist_{candidate_id}_{idx}"):
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

                # Answer and Top Candidates side-by-side (clean recruiter layout)
                col_left, col_right = st.columns([2, 1])

                # Left: AI Recommendation (clean recruiter view)
                with col_left:
                    st.subheader("🧠 AI Recommendation")
                    ans = answer(user_query, retrieved)

                    # Present a concise AI summary (limit length)
                    answer_text = ans.get("answer", "") or ""
                    summary = answer_text[:800] + ("..." if len(answer_text) > 800 else "")

                    with st.container():
                        st.markdown(summary)
                        st.markdown("---")

                    # Structured recommended candidates list
                    st.markdown("**Recommended Candidates**")
                    if docs:
                        for i, d in enumerate(docs[:5], start=1):
                            candidate_name = d.get("meta", {}).get("name") or d.get("name") or f"Candidate {i}"
                            candidate_id = d.get("id") or d.get("meta", {}).get("id") or "-"

                            # Extract structured fields if available
                            meta = d.get("meta", {}) or {}
                            # Skills may be a list or comma-separated string
                            raw_skills = meta.get("skills") or meta.get("keywords") or d.get("skills") or ""
                            if isinstance(raw_skills, (list, tuple)):
                                skills = [s.strip() for s in raw_skills]
                            else:
                                skills = [s.strip() for s in str(raw_skills).split(",") if s.strip()]

                            experience = meta.get("experience") or meta.get("years") or meta.get("seniority") or "-"
                            role = meta.get("role") or meta.get("title") or category or "-"

                            with st.container():
                                st.markdown(f"### {i}. {candidate_name}  —  ID: {candidate_id}")

                                st.markdown("**Key Match Factors**")
                                st.markdown(f"• Experience: {experience}")
                                st.markdown(f"• Role Match: {role}")

                                # Skills badges
                                if skills:
                                    badges = " ".join([f"`{s}`" for s in skills[:6]])
                                    st.markdown(f"**Skills:** {badges}")

                                # Short explanation (if available)
                                explain = d.get("explain", {}) or {}
                                explanation_text = explain.get("explanation") or explain.get("summary") or ""
                                if explanation_text:
                                    short_ex = explanation_text[:300] + ("..." if len(explanation_text) > 300 else "")
                                    st.caption(short_ex)

                                st.markdown("---")
                    else:
                        st.info("No candidate recommendations available for this query.")

                # Right: Top candidates as cards
                with col_right:
                    st.subheader("📋 Top Candidates")
                    if not docs:
                        st.warning("No matching candidates found")
                    else:
                        candidates = [build_candidate_object(d, idx=i) for i, d in enumerate(docs[:8])]
                        for i, cand in enumerate(candidates):
                            render_candidate_card(cand, i)

                        # Resume viewer modal-like area
                        view_id = st.session_state.get("view_resume_id")
                        if view_id:
                            view_text = st.session_state.get("view_resume_text", "")
                            st.markdown(
                                f"""
                            <div style='background-color:#071025;padding:14px;border-radius:10px;border:1px solid #22303a;margin-top:10px;'>
                                <div style='display:flex;justify-content:space-between;align-items:center;'>
                                    <h4 style='margin:0;color:#e6eef8;'>📄 Resume: {view_id}</h4>
                                </div>
                                <div style='margin-top:8px;color:#cbd5e1;white-space:pre-wrap;max-height:360px;overflow:auto;padding-top:8px;'>
                                    {st.escape(view_text) if hasattr(st, 'escape') else view_text}
                                </div>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                            if st.button("Close", key=f"close_view_{view_id}"):
                                st.session_state["view_resume_id"] = None
                                st.session_state["view_resume_text"] = ""

                # Developer debug / processing trace (collapsed)
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
                        # Show docs with full text for debugging
                        for i, d in enumerate(docs, start=1):
                            st.markdown(f"**{i}. ID: {d.get('id')}  |  Score: {d.get('score')}**")
                            meta = d.get('meta', {}) or {}
                            if meta:
                                st.json(meta)
                            # Full resume text (developer only)
                            st.text_area(f"Resume text ({d.get('id')})", value=d.get('text', '') or "", height=250)
                            st.markdown("---")
                    except Exception:
                        st.write(docs)
                
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


