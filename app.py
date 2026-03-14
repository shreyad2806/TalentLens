import os
import warnings
from dotenv import load_dotenv
import streamlit as st


# Suppress warnings to avoid threading issues
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Fix tokenizer threading issues

load_dotenv()

st.set_page_config(page_title="Resume QA", page_icon="📄", layout="wide")
st.title("📄 Resume QA powered by Local RAG")

st.markdown(
    """
    - This system is powered by local RAG, which uses open-source models and tools to answer questions.
    - Enter a question; the system will classify it into a category, filter resumes by that category, search semantically, and return both the raw docs and a readable answer.
    """
)

with st.form("query_form"):
    user_query = st.text_area("Your question", height=120, placeholder="e.g., Find senior Java developers with Spring experience in Bangalore")
    submitted = st.form_submit_button("Search 🚀")

if submitted and user_query.strip():
    # Import here to avoid heavy model loading at Streamlit startup.
    from src.query_pipeline import retrieve, answer

    with st.spinner("Classifying, searching, and generating answer..."):
        try:
            retrieved = retrieve(user_query)
            category = retrieved.get("category")
            docs = retrieved.get("docs", [])

            # Category badge
            st.markdown(
                f"<div style='display:inline-block;padding:6px 12px;border-radius:16px;background:#EEF6FF;color:#1D4ED8;font-weight:600;'>Category: {category}</div>",
                unsafe_allow_html=True,
            )

            # Answer and Docs side-by-side
            col_left, col_right = st.columns([1.1, 0.9])

            with col_left:
                st.subheader("Answer")
                ans = answer(user_query, retrieved)
                st.write(ans.get("answer", ""))

            with col_right:
                st.subheader("Top 5 Candidates (documents as-it-is)")
                if not docs:
                    st.info("No documents found. Try another query.")
                else:
                    for i, d in enumerate(docs[:5], start=1):
                        score = d.get("score")
                        score_str = f"{score}%" if isinstance(score, (int, float)) else str(score)
                        with st.expander(f"{i}. ID: {d['id']}  |  Score: {score_str}"):
                            st.subheader("Candidate Match")
                            try:
                                st.metric("Match Score", f"{d.get('score')}%")
                            except Exception:
                                st.write(f"Match Score: {score_str}")

                            st.markdown("### Why selected")
                            explain = d.get("explain", {}) or {}
                            st.write(explain.get("explanation", ""))

                            st.markdown("---")
                            st.subheader("Resume (raw)")
                            st.write(d.get("text", ""))

            # Processing trace - expanded by default
            st.subheader("Processing Trace")
            with st.expander("See detailed steps (tools, filters, models)", expanded=True):
                for step in retrieved.get("trace", []):
                    st.markdown(f"**{step.get('step')}**")
                    st.json(step)
                if 'ans' not in locals():
                    ans = answer(user_query, retrieved)
                st.markdown("**LLM answer generation**")
                st.json(ans.get("trace", {}))
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please try again or check your configuration.")


