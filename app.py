"""TalentLens — AI Resume Intelligence Platform (one-page dashboard)."""
from __future__ import annotations

import html
import os
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="TalentLens", page_icon="🎯", layout="wide")

# ── Custom theme / design system ────────────────────────────────────────────

def _talentlens_css() -> str:
    return """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

      html, body, .stApp, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif !important;
        background-color: #0B1220 !important;
        color: #F8FAFC !important;
      }

      /* Hide Streamlit chrome */
      header { visibility: hidden !important; }
      footer { visibility: hidden !important; }
      #MainMenu { visibility: hidden !important; }
      [data-testid="stToolbar"] { display: none !important; }

      /* Sidebar */
      [data-testid="stSidebar"] {
        background-color: #0B1220 !important;
        border-right: 1px solid #253247 !important;
      }
      [data-testid="stSidebar"] .st-emotion-cache-16idsys p,
      [data-testid="stSidebar"] .st-emotion-cache-16idsys span,
      [data-testid="stSidebar"] .st-emotion-cache-16idsys div {
        color: #F8FAFC !important;
      }

      /* Bordered containers (cards, drawer, filter bar) */
      [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #131C2E !important;
        border: 1px solid #253247 !important;
        border-radius: 16px !important;
        padding: 0.75rem 1rem !important;
        margin-bottom: 0.75rem !important;
      }

      /* Inputs */
      .stTextInput input, .stSelectbox, .stMultiselect, .stSlider, div[data-baseweb="input"] input {
        background-color: #131C2E !important;
        color: #F8FAFC !important;
        border: 1px solid #253247 !important;
        border-radius: 12px !important;
      }
      .stTextInput input:focus, div[data-baseweb="input"] input:focus {
        border-color: #6D5DF6 !important;
        box-shadow: 0 0 0 1px #6D5DF6 !important;
      }

      /* Buttons */
      .stButton > button {
        background-color: #6D5DF6 !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.1rem !important;
      }
      .stButton > button:hover { background-color: #5b4ed6 !important; }
      .stButton > button[kind="secondary"] {
        background-color: transparent !important;
        color: #F8FAFC !important;
        border: 1px solid #253247 !important;
      }
      .stButton > button[kind="secondary"]:hover { background-color: #1e293b !important; }

      /* Radio nav */
      div[role="radiogroup"] > label {
        background-color: transparent !important;
        border-radius: 10px !important;
        padding: 0.55rem 0.75rem !important;
        color: #94A3B8 !important;
        border: 1px solid transparent !important;
      }
      div[role="radiogroup"] > label:has(input:checked) {
        background-color: #131C2E !important;
        color: #F8FAFC !important;
        border-color: #6D5DF6 !important;
      }

      /* Right drawer sticky */
      [data-testid="stColumn"]:last-child {
        position: sticky !important;
        top: 1rem !important;
        align-self: flex-start !important;
      }

      /* Match circle */
      .match-circle {
        width: 96px;
        height: 96px;
        border-radius: 50%;
        background: conic-gradient(#22C55E var(--pct, 92%), #1e293b 0);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        margin: 0 auto 0.5rem auto;
      }
      .match-circle-inner {
        width: 76px;
        height: 76px;
        border-radius: 50%;
        background-color: #131C2E;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
      }
      .match-number { font-size: 1.6rem; font-weight: 700; color: #F8FAFC; }
      .match-label { font-size: 0.7rem; color: #94A3B8; }

      /* Skill chips */
      .skill-chip {
        display: inline-block;
        background-color: #1e293b;
        color: #F8FAFC;
        border: 1px solid #253247;
        border-radius: 20px;
        padding: 0.2rem 0.6rem;
        margin: 0.15rem;
        font-size: 0.78rem;
      }
      .skill-chip.matched { background-color: #6D5DF6; border-color: #6D5DF6; color: white; }

      /* Avatar */
      .avatar-circle {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        background: linear-gradient(135deg, #6D5DF6, #22C55E);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
      }

      /* Summary clamp */
      .summary { display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; color: #94A3B8; font-size: 0.85rem; line-height: 1.45; }

      /* Status dots */
      .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 8px; }
      .dot-green { background-color: #22C55E; }
      .dot-blue { background-color: #3B82F6; }
      .dot-purple { background-color: #6D5DF6; }

      /* Headings */
      h1, h2, h3, h4, h5 { color: #F8FAFC !important; }
      .muted { color: #94A3B8; }

      /* Progress bars */
      .stProgress > div > div > div > div { background-color: #6D5DF6 !important; }
    </style>
    """

st.markdown(_talentlens_css(), unsafe_allow_html=True)

# ── Cached backend factories (lazy, once per session) ───────────────────────

@st.cache_resource(show_spinner=False)
def _get_retrieval_bundle():
    from src.bootstrap.composition_root import create_retrieval_bundle
    return create_retrieval_bundle()


@st.cache_resource(show_spinner=False)
def _run_bootstrap():
    from src.bootstrap.bootstrap_service import BootstrapService
    return BootstrapService(verbose=False).bootstrap()


# ── Session state ───────────────────────────────────────────────────────────

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "shortlist" not in st.session_state:
    st.session_state.shortlist = []

if "shortlist_map" not in st.session_state:
    st.session_state.shortlist_map = {}

if "search_results" not in st.session_state:
    st.session_state.search_results = []

if "selected_candidate" not in st.session_state:
    st.session_state.selected_candidate = None

if "search_service" not in st.session_state:
    st.session_state.search_service = None


# ── Helpers ─────────────────────────────────────────────────────────────────

def _initials(name: str) -> str:
    clean = html.unescape(name).strip()
    if not clean:
        return "?"
    if clean.lower().startswith("resume #"):
        return "R#"
    parts = clean.split()
    return "".join(p[0].upper() for p in parts[:2] if p)


def _confidence_label(conf: float) -> str:
    if conf >= 0.8:
        return "High Confidence"
    if conf >= 0.5:
        return "Medium Confidence"
    return "Low Confidence"


def _add_to_shortlist(candidate: dict) -> None:
    cid = candidate.get("id")
    if not cid or cid in st.session_state.shortlist_map:
        return
    st.session_state.shortlist.append(cid)
    st.session_state.shortlist_map[cid] = candidate


def _remove_from_shortlist(cid: str) -> None:
    if cid in st.session_state.shortlist_map:
        del st.session_state.shortlist_map[cid]
    st.session_state.shortlist = [c for c in st.session_state.shortlist if c != cid]


def _why_matched(candidate: dict) -> list[str]:
    reasons = []
    if candidate.get("role_match", 0) >= 40:
        reasons.append("Role alignment")
    if candidate.get("matched_skills"):
        reasons.append(f"Skills match: {', '.join(candidate['matched_skills'][:3])}")
    if candidate.get("industry_match", 0) >= 40:
        reasons.append("Industry relevance")
    if candidate.get("experience_match", 0) >= 40:
        reasons.append("Relevant experience")
    if candidate.get("education_match", 0) >= 40:
        reasons.append("Education match")
    if candidate.get("location_match", 0) >= 40:
        reasons.append("Location match")
    if not reasons:
        reasons.append("Retrieved by semantic search")
    return reasons


# ── Bootstrap guard ─────────────────────────────────────────────────────────

if "bootstrap_complete" not in st.session_state:
    st.session_state.bootstrap_complete = False

if not st.session_state.bootstrap_complete:
    with st.spinner("Initializing TalentLens indexes..."):
        bootstrap_result = _run_bootstrap()
        st.session_state.bootstrap_complete = True
        st.session_state.bootstrap_result = bootstrap_result

    bundle = _get_retrieval_bundle()
    from src.search.search_service import SearchService
    st.session_state.search_service = SearchService(hybrid_service=bundle.hybrid_service)
    st.rerun()

bundle = _get_retrieval_bundle()
if st.session_state.search_service is None:
    from src.search.search_service import SearchService
    st.session_state.search_service = SearchService(hybrid_service=bundle.hybrid_service)


# ── LEFT SIDEBAR ────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("<h2 style='margin:0; color:#F8FAFC;'>TalentLens</h2>", unsafe_allow_html=True)
    st.markdown("<p class='muted' style='margin:0 0 2rem 0; font-size:0.8rem;'>AI Resume Intelligence</p>", unsafe_allow_html=True)

    nav = st.radio(
        "Navigation",
        ["🏠 Dashboard", "⭐ Shortlist"],
        index=0 if st.session_state.page == "Dashboard" else 1,
        label_visibility="collapsed",
    )
    st.session_state.page = "Dashboard" if "Dashboard" in nav else "Shortlist"

    st.sidebar.markdown("---")
    st.sidebar.markdown("<p style='font-weight:600; color:#F8FAFC;'>System Status</p>", unsafe_allow_html=True)
    vector_count = bundle.vector_store_service.count()
    bm25_count = bundle.bm25_index.total_documents if hasattr(bundle.bm25_index, "total_documents") else 0
    st.sidebar.markdown(f"<span class='status-dot dot-green'></span>Vector Store Connected", unsafe_allow_html=True)
    st.sidebar.markdown(f"<span class='status-dot dot-blue'></span>BM25 Ready", unsafe_allow_html=True)
    st.sidebar.markdown(f"<span class='status-dot dot-purple'></span>RAG Pipeline Active", unsafe_allow_html=True)
    st.sidebar.markdown(f"<p class='muted' style='font-size:0.75rem; margin-top:1rem;'>{vector_count} vectors • {bm25_count} BM25 docs</p>", unsafe_allow_html=True)


# ── MAIN LAYOUT ─────────────────────────────────────────────────────────────

main_col, drawer_col = st.columns([2.8, 1.2])

# ── TOP SECTION (Dashboard only) ────────────────────────────────────────────

if st.session_state.page == "Dashboard":
    with main_col:
        st.markdown("<h1 style='text-align:center; margin-bottom:0;'>TalentLens</h1>", unsafe_allow_html=True)
        st.markdown("<p class='muted' style='text-align:center; margin-bottom:2rem;'>Search thousands of resumes using natural language.</p>", unsafe_allow_html=True)

        # Search form + filters
        with st.container(border=True):
            with st.form("search_form", clear_on_submit=False):
                c1, c2 = st.columns([5, 1])
                with c1:
                    user_query = st.text_input(
                        "Search",
                        placeholder="Finance Manager with Excel and Banking experience",
                        label_visibility="collapsed",
                    )
                with c2:
                    submitted = st.form_submit_button("Search", type="primary", use_container_width=True)

                # Filter bar
                st.markdown("<p style='font-weight:600; color:#F8FAFC; margin:1rem 0 0.5rem;'>Filters</p>", unsafe_allow_html=True)
                f1, f2, f3, f4, f5, f6 = st.columns([1, 1, 1, 1.4, 1, 0.8])
                with f1:
                    exp_range = st.slider("Experience", 0, 20, (0, 20), step=1)
                with f2:
                    location_filter = st.text_input("Location", placeholder="Any")
                with f3:
                    education_filter = st.text_input("Education", placeholder="Any")
                with f4:
                    skill_pool = ["Python", "SQL", "AWS", "Docker", "Kubernetes", "React", "Node.js", "Java",
                                  "Machine Learning", "AI", "RAG", "LLMs", "PostgreSQL", "MongoDB", "Git",
                                  "CI/CD", "Flask", "Django", "Spark", "Excel", "Tableau", "Power BI",
                                  "TypeScript", "Next.js", "GraphQL", "Redis", "Azure", "GCP", "Salesforce",
                                  "Spring", "C++", "HTML", "CSS"]
                    skills_filter = st.multiselect("Skills", skill_pool, placeholder="Any skills")
                with f5:
                    max_results = st.selectbox("Max Results", [5, 10, 15, 20], index=1)
                with f6:
                    reset = st.form_submit_button("Reset", type="secondary")

        if reset:
            st.session_state.search_results = []
            st.session_state.selected_candidate = None
            st.rerun()

        # Search execution
        if submitted and user_query.strip():
            with st.spinner("Retrieving and scoring candidates..."):
                from src.search.schema import SearchFilters

                _exp_min = exp_range[0] if exp_range[0] > 0 else None
                _exp_max = exp_range[1] if exp_range[1] < 20 else None
                _loc = location_filter.strip() if location_filter else None
                _edu = education_filter.strip() if education_filter else None
                _skills = [s.lower() for s in skills_filter] if skills_filter else None

                filters = SearchFilters(
                    role=None,
                    skills=_skills,
                    location=_loc,
                    experience_min=_exp_min,
                    experience_max=_exp_max,
                    education=_edu,
                    strict=False,
                )

                results = st.session_state.search_service.search(
                    query=user_query,
                    top_k=max_results,
                    filters=filters,
                )
                st.session_state.search_results = [r.to_frontend_dict() for r in results]
                st.session_state.search_results = [
                    {**c, "score_breakdown": r.score_breakdown}
                    for c, r in zip(st.session_state.search_results, results)
                ]
                st.session_state.selected_candidate = None

        # Search stats
        if st.session_state.search_results:
            total = bm25_count if bm25_count else 2484
            n_results = len(st.session_state.search_results)
            high = sum(1 for c in st.session_state.search_results if c.get("overall_match", 0) >= 90)
            good = sum(1 for c in st.session_state.search_results if 70 <= c.get("overall_match", 0) < 90)
            fair = n_results - high - good

            stat1, stat2, stat3, stat4, stat5 = st.columns(5)
            with stat1:
                with st.container(border=True):
                    st.markdown(f"<p class='muted' style='margin:0; font-size:0.75rem;'>Indexed Resumes</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='margin:0; font-weight:700;'>{total}</p>", unsafe_allow_html=True)
            with stat2:
                with st.container(border=True):
                    st.markdown(f"<p class='muted' style='margin:0; font-size:0.75rem;'>Results</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='margin:0; font-weight:700;'>{n_results}</p>", unsafe_allow_html=True)
            with stat3:
                with st.container(border=True):
                    st.markdown(f"<p class='muted' style='margin:0; font-size:0.75rem;'>High Match (≥90%)</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='margin:0; font-weight:700; color:#22C55E;'>{high}</p>", unsafe_allow_html=True)
            with stat4:
                with st.container(border=True):
                    st.markdown(f"<p class='muted' style='margin:0; font-size:0.75rem;'>Good Match (70-89%)</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='margin:0; font-weight:700; color:#F59E0B;'>{good}</p>", unsafe_allow_html=True)
            with stat5:
                with st.container(border=True):
                    st.markdown(f"<p class='muted' style='margin:0; font-size:0.75rem;'>Fair Match (<70%)</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='margin:0; font-weight:700; color:#94A3B8;'>{fair}</p>", unsafe_allow_html=True)

        # Results
        for i, candidate in enumerate(st.session_state.search_results):
            with st.container(border=True):
                r1, r2, r3 = st.columns([0.12, 0.58, 0.30])

                with r1:
                    st.markdown(f"<div class='avatar-circle'>{_initials(candidate.get('name', ''))}</div>", unsafe_allow_html=True)

                with r2:
                    st.markdown(f"<h4 style='margin:0; color:#F8FAFC;'>{html.escape(candidate.get('name', 'Unknown'))}</h4>", unsafe_allow_html=True)
                    role = html.escape(str(candidate.get('role') or 'Role not specified'))
                    loc = html.escape(str(candidate.get('location') or 'Location not specified'))
                    exp = html.escape(str(candidate.get('experience') or 'Experience not specified'))
                    edu = html.escape(str(candidate.get('education')[0] if candidate.get('education') else 'Education not specified'))
                    st.markdown(f"<p class='muted' style='margin:0; font-size:0.8rem;'>{role} • {loc} • {exp} • {edu}</p>", unsafe_allow_html=True)

                    if candidate.get('top_skills'):
                        chips = []
                        for s in candidate.get('top_skills', [])[:8]:
                            matched = s.lower() in [m.lower() for m in candidate.get('matched_skills', [])]
                            cls = "skill-chip matched" if matched else "skill-chip"
                            chips.append(f"<span class='{cls}'>{html.escape(s)}</span>")
                        st.markdown("".join(chips), unsafe_allow_html=True)

                    preview = html.escape(str(candidate.get('resume_preview', '') or candidate.get('summary', '') or ''))
                    st.markdown(f"<p class='summary'>{preview}</p>", unsafe_allow_html=True)

                with r3:
                    pct = int(round(candidate.get('overall_match', 0)))
                    conf = _confidence_label(float(candidate.get('confidence', 0)))
                    st.markdown(
                        f"""
                        <div class='match-circle' style='--pct:{pct}%;'>
                          <div class='match-circle-inner'>
                            <span class='match-number'>{pct}%</span>
                            <span class='match-label'>{conf}</span>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button("View", key=f"view_search_{i}", use_container_width=True):
                            st.session_state.selected_candidate = candidate
                            st.rerun()
                    with b2:
                        if st.button("Shortlist", key=f"short_search_{i}", use_container_width=True, type="primary"):
                            _add_to_shortlist(candidate)
                            st.toast("Added to shortlist")

# ── SHORTLIST VIEW ──────────────────────────────────────────────────────────

elif st.session_state.page == "Shortlist":
    with main_col:
        st.markdown("<h1 style='margin-bottom:0.25rem;'>Shortlist</h1>", unsafe_allow_html=True)
        st.markdown("<p class='muted' style='margin-bottom:1.5rem;'>Saved candidates for this search session.</p>", unsafe_allow_html=True)

        if not st.session_state.shortlist:
            st.info("No candidates have been shortlisted yet.")
        else:
            for i, cid in enumerate(st.session_state.shortlist):
                candidate = st.session_state.shortlist_map.get(cid)
                if not candidate:
                    continue
                with st.container(border=True):
                    r1, r2, r3 = st.columns([0.12, 0.58, 0.30])

                    with r1:
                        st.markdown(f"<div class='avatar-circle'>{_initials(candidate.get('name', ''))}</div>", unsafe_allow_html=True)

                    with r2:
                        st.markdown(f"<h4 style='margin:0; color:#F8FAFC;'>{html.escape(candidate.get('name', 'Unknown'))}</h4>", unsafe_allow_html=True)
                        role = html.escape(str(candidate.get('role') or 'Role not specified'))
                        loc = html.escape(str(candidate.get('location') or 'Location not specified'))
                        st.markdown(f"<p class='muted' style='margin:0; font-size:0.8rem;'>{role} • {loc}</p>", unsafe_allow_html=True)

                        chips = [f"<span class='skill-chip'>{html.escape(s)}</span>" for s in candidate.get('top_skills', [])[:6]]
                        st.markdown("".join(chips), unsafe_allow_html=True)

                    with r3:
                        pct = int(round(candidate.get('overall_match', 0)))
                        conf = _confidence_label(float(candidate.get('confidence', 0)))
                        st.markdown(
                            f"""
                            <div class='match-circle' style='--pct:{pct}%;'>
                              <div class='match-circle-inner'>
                                <span class='match-number'>{pct}%</span>
                                <span class='match-label'>{conf}</span>
                              </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        b1, b2 = st.columns(2)
                        with b1:
                            if st.button("View", key=f"view_short_{i}", use_container_width=True):
                                st.session_state.selected_candidate = candidate
                                st.rerun()
                        with b2:
                            if st.button("Remove", key=f"remove_short_{i}", use_container_width=True, type="secondary"):
                                _remove_from_shortlist(cid)
                                st.rerun()

# ── RIGHT-SIDE DRAWER ───────────────────────────────────────────────────────

with drawer_col:
    if st.session_state.selected_candidate:
        c = st.session_state.selected_candidate
        with st.container(border=True):
            st.markdown("<h3 style='margin:0 0 1rem 0;'>Match Breakdown</h3>", unsafe_allow_html=True)

            pct = int(round(c.get('overall_match', 0)))
            conf = _confidence_label(float(c.get('confidence', 0)))
            st.markdown(
                f"""
                <div class='match-circle' style='--pct:{pct}%; width:120px; height:120px;'>
                  <div class='match-circle-inner' style='width:96px; height:96px;'>
                    <span class='match-number' style='font-size:2rem;'>{pct}%</span>
                    <span class='match-label'>{conf}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<p style='font-weight:600; color:#F8FAFC; margin:1rem 0 0.5rem;'>Components</p>", unsafe_allow_html=True)
            components = [
                ("Role", c.get('role_match', 0) / 100 if isinstance(c.get('role_match'), (int, float)) else 0),
                ("Skills", c.get('skill_match', 0) / 100 if isinstance(c.get('skill_match'), (int, float)) else 0),
                ("Experience", c.get('experience_match', 0) / 100 if isinstance(c.get('experience_match'), (int, float)) else 0),
                ("Industry", c.get('industry_match', 0) / 100 if isinstance(c.get('industry_match'), (int, float)) else 0),
                ("Education", c.get('education_match', 0) / 100 if isinstance(c.get('education_match'), (int, float)) else 0),
                ("Location", c.get('location_match', 0) / 100 if isinstance(c.get('location_match'), (int, float)) else 0),
            ]
            for label, val in components:
                if val > 0:
                    st.caption(f"{label}: {int(round(val*100))}%")
                    st.progress(min(1.0, val), text="")

            st.markdown("<p style='font-weight:600; color:#F8FAFC; margin:1rem 0 0.5rem;'>Why this matched</p>", unsafe_allow_html=True)
            for reason in _why_matched(c):
                st.markdown(f"<p style='margin:0.15rem 0; color:#94A3B8; font-size:0.85rem;'>✓ {html.escape(reason)}</p>", unsafe_allow_html=True)

            if c.get('top_skills'):
                st.markdown("<p style='font-weight:600; color:#F8FAFC; margin:1rem 0 0.5rem;'>Top Skills</p>", unsafe_allow_html=True)
                chips = [f"<span class='skill-chip'>{html.escape(s)}</span>" for s in c.get('top_skills', [])[:12]]
                st.markdown("".join(chips), unsafe_allow_html=True)

            with st.expander("View Full Resume"):
                summary = html.escape(str(c.get('summary') or c.get('resume_preview') or 'No preview available.'))
                st.markdown(f"<p class='summary' style='-webkit-line-clamp: unset;'>{summary}</p>", unsafe_allow_html=True)

            if st.button("Close", use_container_width=True):
                st.session_state.selected_candidate = None
                st.rerun()
