import os
import warnings
from dotenv import load_dotenv
import streamlit as st


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
                        st.info("No documents found. Try another query.")
                    else:
                        for i, d in enumerate(docs[:5], start=1):
                            score = d.get("score")
                            try:
                                score_pct = f"{int(round(float(score)))}%"
                            except Exception:
                                score_pct = str(score)

                            candidate_name = d.get("meta", {}).get("name") or d.get("name") or f"Candidate {i}"
                            category_label = d.get("meta", {}).get("category") or category or "-"

                            with st.container():
                                st.subheader(f"{candidate_name}")
                                st.write(f"Category: {category_label}")
                                try:
                                    st.metric("Match Score", score_pct)
                                except Exception:
                                    st.write(f"Match Score: {score_pct}")

                                # Brief explanation
                                explain = d.get("explain", {}) or {}
                                if explain.get("explanation"):
                                    st.caption(explain.get("explanation"))

                                # Resume preview (limit length)
                                resume_text = d.get("text", "") or ""
                                preview = resume_text[:400] + ("..." if len(resume_text) > 400 else "")
                                st.write(preview)
                                st.markdown("---")

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

            st.subheader("Analytics Dashboard")
            df = load_data()

            if df is None or df.empty:
                st.info("No resume dataset found at Resume/Resume.csv")
            else:
                text_col = "Resume" if "Resume" in df.columns else df.columns[0]
                skills = extract_skills(df[text_col])
                locations = extract_locations(df[text_col])
                categories = category_distribution(df)

                if skills:
                    fig = px.bar(x=list(dict(skills).keys()), y=list(dict(skills).values()), title="Top Skills in Resume Database")
                    st.plotly_chart(fig, use_container_width=True)

                if locations:
                    fig = px.bar(x=list(dict(locations).keys()), y=list(dict(locations).values()), title="Candidate Locations")
                    st.plotly_chart(fig, use_container_width=True)

                if not categories.empty:
                    fig = px.pie(values=categories.values, names=categories.index, title="Candidate Category Distribution")
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Analytics module error: {e}")


