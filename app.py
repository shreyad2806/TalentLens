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

                # Use the LLM's final answer as an intro, but limit length
                answer_text = ans.get("answer", "") or ""
                clean_intro = answer_text[:1200] + ("..." if len(answer_text) > 1200 else "")
                st.markdown(clean_intro)
                st.markdown("---")

                # Build a clean numbered list of recommended candidates from retrieved docs
                if docs:
                    st.markdown("Based on the provided resumes, the following candidates match your criteria:")
                    for i, d in enumerate(docs[:5], start=1):
                        candidate_name = d.get("meta", {}).get("name") or d.get("name") or f"Candidate {i}"
                        candidate_id = d.get("id") or d.get("meta", {}).get("id") or "-"

                        # derive short bullets from explanation or meta
                        explain = d.get("explain", {}) or {}
                        explanation_text = explain.get("explanation") or explain.get("summary") or ""
                        # create 2 short bullets if possible
                        bullets = []
                        if explanation_text:
                            parts = [p.strip() for p in explanation_text.split(".") if p.strip()]
                            for part in parts[:2]:
                                bullets.append(part)
                        else:
                            # fallback: attempt to extract from meta
                            meta_skills = d.get("meta", {}).get("skills") or d.get("meta", {}).get("keywords") or ""
                            if meta_skills:
                                bullets = [s.strip() for s in str(meta_skills).split(",")][:2]

                        # render numbered candidate entry
                        st.markdown(f"**{i}. {candidate_name} (ID: {candidate_id})**")
                        for b in bullets:
                            st.markdown(f"- {b}")
                        st.markdown("\n")
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
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please try again or check your configuration.")


