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

            # Left: AI Answer
            with col_left:
                st.subheader("🧠 AI Answer")
                ans = answer(user_query, retrieved)
                st.write(ans.get("answer", ""))

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

            # Processing details in an expander
            st.subheader("🛠️ Processing Details")
            with st.expander("See processing details (tools, filters, models)"):
                trace = retrieved.get("trace", {})
                # Show trace as JSON inside the expander
                try:
                    st.json(trace)
                except Exception:
                    st.write(trace)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please try again or check your configuration.")


