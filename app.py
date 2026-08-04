"""TalentLens — AI Resume Intelligence Platform (one-page dashboard)."""
from __future__ import annotations

import hashlib
import html
import logging
import os
import re
import time
from pathlib import Path
import streamlit as st

from src.recruiter_report import build_recruiter_report, format_markdown, format_text

st.set_page_config(page_title="TalentLens", page_icon="🎯", layout="wide")

APP_START = time.perf_counter()


def _talentlens_css() -> str:
    return """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

      html, body, .stApp, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif !important;
        background-color: #0B1220 !important;
        color: #F8FAFC !important;
      }

      header, footer, #MainMenu, [data-testid="stToolbar"] { display: none !important; }

      .block-container {
        max-width: 1500px !important;
        width: 100% !important;
        margin: 0 auto !important;
        padding: 0.5rem 1.25rem 1.25rem 1.25rem !important;
      }

      [data-testid="stSidebar"] {
        min-width: 280px !important;
        max-width: 300px !important;
        width: 300px !important;
      }

      [data-testid="stSidebarContent"] { padding: 1rem !important; }

      [data-testid="stSidebar"] {
        background-color: #0B1220 !important;
        border-right: 1px solid #253247 !important;
      }

      [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #131C2E !important;
        border: 1px solid #253247 !important;
        border-radius: 12px !important;
        padding: 0.5rem 0.75rem !important;
        margin-bottom: 0.5rem !important;
      }

      .stTextInput input, .stSelectbox, .stMultiselect, .stSlider, div[data-baseweb="input"] input,
      .stTextInput input:focus, div[data-baseweb="input"] input:focus {
        background-color: #131C2E !important;
        color: #F8FAFC !important;
        border: 1px solid #253247 !important;
        border-radius: 8px !important;
      }

      .stButton > button {
        color: #F8FAFC !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.45rem 0.9rem !important;
        font-size: 0.78rem !important;
      }
      .stButton > button[data-testid="baseButton-primary"] {
        background-color: #6D5DF6 !important;
        border: 1px solid #6D5DF6 !important;
      }
      .stButton > button[data-testid="baseButton-primary"]:hover { background-color: #5b4ed6 !important; }
      .stButton > button[kind="secondary"], .stButton > button[data-testid="baseButton-secondary"] {
        background-color: transparent !important;
        border: 1px solid #253247 !important;
        color: #F8FAFC !important;
      }
      .stButton > button[kind="secondary"]:hover, .stButton > button[data-testid="baseButton-secondary"]:hover { background-color: #1e293b !important; }

      div[role="radiogroup"] > label {
        background-color: transparent !important;
        border-radius: 8px !important;
        padding: 0.45rem 0.65rem !important;
        color: #94A3B8 !important;
        border: 1px solid transparent !important;
      }
      div[role="radiogroup"] > label:has(input:checked) {
        background-color: #131C2E !important;
        color: #F8FAFC !important;
        border-color: #6D5DF6 !important;
      }

      .match-circle {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        background: conic-gradient(#22C55E var(--pct, 92%), #1e293b 0);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 0.5rem 0 auto;
      }
      .match-circle-inner {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background-color: #131C2E;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        text-align: center;
      }
      .match-number { font-size: 0.78rem; font-weight: 700; color: #F8FAFC; line-height: 1; white-space: nowrap; }
      .match-label { font-size: 0.48rem; color: #94A3B8; line-height: 1; white-space: nowrap; margin-top: 1px; }

      .match-circle-lg {
        width: 110px;
        height: 110px;
        border-radius: 50%;
        background: conic-gradient(#22C55E var(--pct, 92%), #1e293b 0);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 0.75rem auto;
      }
      .match-circle-lg .match-circle-inner { width: 86px; height: 86px; }
      .match-circle-lg .match-number { font-size: 1.6rem; }
      .match-circle-lg .match-label { font-size: 0.68rem; margin-top: 3px; }

      .skill-chip {
        display: inline-block;
        background-color: #1e293b;
        color: #F8FAFC;
        border: 1px solid #253247;
        border-radius: 20px;
        padding: 0.12rem 0.5rem;
        margin: 0.1rem 0.15rem 0.1rem 0;
        font-size: 0.7rem;
      }
      .skill-chip.matched { background-color: #6D5DF6; border-color: #6D5DF6; color: white; }
      .skill-chip.more { background-color: transparent; color: #6D5DF6; border-color: #6D5DF6; }

      .avatar-circle {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: linear-gradient(135deg, #6D5DF6, #22C55E);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 0.8rem;
      }

      .ai-summary { display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; color: #94A3B8; font-size: 0.78rem; line-height: 1.35; margin: 0.15rem 0 0 0; }

      .status-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 6px; }
      .dot-green { background-color: #22C55E; }
      .dot-blue { background-color: #3B82F6; }
      .dot-purple { background-color: #6D5DF6; }

      h1, h2, h3, h4, h5 { color: #F8FAFC !important; margin: 0 !important; }
      .muted { color: #94A3B8; }

      .stProgress > div > div > div > div { background-color: #6D5DF6 !important; }

      .stat-card { display: flex; flex-direction: column; justify-content: space-between; min-height: 72px; }
      .stat-icon {
        width: 22px; height: 22px; border-radius: 6px; display: inline-flex; align-items: center; justify-content: center;
        font-size: 0.65rem; font-weight: 700; margin-bottom: 0.35rem;
      }
      .stat-icon.idx { background: #1e293b; color: #F8FAFC; }
      .stat-icon.res { background: #1e293b; color: #F8FAFC; }
      .stat-icon.high { background: #14532d; color: #22C55E; }
      .stat-icon.good { background: #451a03; color: #F59E0B; }
      .stat-icon.fair { background: #1e293b; color: #94A3B8; }

      mark.query-hit {
        background-color: rgba(109, 93, 246, 0.35);
        color: #F8FAFC;
        border-radius: 3px;
        padding: 0 2px;
      }

      pre.full-resume {
        white-space: pre-wrap;
        max-height: 55vh;
        overflow-y: auto;
        font-size: 0.78rem;
        color: #F8FAFC;
        background-color: #0B1220;
        padding: 0.75rem;
        border-radius: 8px;
        border: 1px solid #253247;
      }

      /* Tighten Streamlit vertical whitespace */
      .st-emotion-cache-1y4p8pa, .st-emotion-cache-1n76uvr { gap: 0.5rem !important; }
      p, .stMarkdown p { margin: 0.15rem 0 !important; }

      /* Skeleton loader */
      .skeleton-card {
        background: linear-gradient(90deg, #131C2E 25%, #1E293B 50%, #131C2E 75%);
        background-size: 200% 100%;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border: 1px solid #253247;
        animation: shimmer 1.6s infinite;
      }
      .skeleton-line { background: #253247; border-radius: 4px; margin-bottom: 0.6rem; }
      .skeleton-title { width: 45%; height: 1rem; }
      .skeleton-subtitle { width: 70%; height: 0.75rem; }
      .skeleton-chip { width: 30%; height: 0.75rem; }
      .skeleton-circle { width: 40px; height: 40px; border-radius: 50%; background: #253247; }
      @keyframes shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
      }

      /* Focus and accessibility */
      .stButton > button:focus, .stTextInput input:focus, .stSelectbox:focus-visible, .stSlider [role="slider"]:focus {
        outline: 2px solid #6D5DF6 !important;
        outline-offset: 2px !important;
      }
      [role="button"]:focus, [tabindex]:focus { outline: 2px solid #6D5DF6 !important; }
    </style>
    """

st.markdown(_talentlens_css(), unsafe_allow_html=True)


def _compute_cache_key() -> str:
    """Build a cache key from dataset content and active configuration."""
    base = Path(__file__).resolve().parent
    dataset_path = base / "combined" / "production_dataset.json"
    parts: list[str] = []
    if dataset_path.exists():
        h = hashlib.md5()
        with open(dataset_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        parts.append(h.hexdigest())
    else:
        parts.append("no-dataset")

    parts.extend([
        os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        os.getenv("VECTOR_STORE_PROVIDER", "qdrant"),
        str(os.getenv("EMBEDDING_DIM", "")),
    ])
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]


@st.cache_resource(show_spinner=False)
def _get_retrieval_bundle(cache_key: str):
    logging.info("[CACHE-MISS] Creating retrieval bundle for cache_key=%s", cache_key)
    from src.bootstrap.composition_root import create_retrieval_bundle
    bundle = create_retrieval_bundle()
    print("[STARTUP] Retrieval bundle ready")
    for key, val in bundle.startup_metrics.items():
        print(f"[STARTUP] {key}: {val:.2f}ms")
    return bundle


@st.cache_resource(show_spinner=False)
def _get_search_service(cache_key: str):
    logging.info("[CACHE-MISS] Creating search service for cache_key=%s", cache_key)
    from src.search.search_service import SearchService
    bundle = _get_retrieval_bundle(cache_key)
    return SearchService(
        hybrid_service=bundle.hybrid_service,
        reranker=bundle.reranker,
    )


@st.cache_resource(show_spinner=False)
def _run_bootstrap(cache_key: str):
    logging.info("[CACHE-MISS] Running bootstrap for cache_key=%s", cache_key)
    from src.bootstrap.bootstrap_service import BootstrapService
    bundle = _get_retrieval_bundle(cache_key)
    return BootstrapService(verbose=False, bundle=bundle).bootstrap()


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
if "bootstrap_complete" not in st.session_state:
    st.session_state.bootstrap_complete = False
if "is_loading" not in st.session_state:
    st.session_state.is_loading = False
if "displayed_count" not in st.session_state:
    st.session_state.displayed_count = 10
if "search_error" not in st.session_state:
    st.session_state.search_error = None


def _initials(name: str) -> str:
    clean = html.unescape(name).strip()
    if not clean:
        return "?"
    words = clean.split()
    return "".join(w[0].upper() for w in words[:2] if w)


def _render_skeletons(n: int = 3):
    for _ in range(n):
        with st.container():
            c1, c2 = st.columns([0.1, 0.9])
            with c1:
                st.markdown('<div class="skeleton-card" style="width:50px;height:50px;padding:0;"><div class="skeleton-circle"></div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown('''
                    <div class="skeleton-card">
                        <div class="skeleton-line skeleton-title"></div>
                        <div class="skeleton-line skeleton-subtitle"></div>
                        <div class="skeleton-line skeleton-chip"></div>
                    </div>
                ''', unsafe_allow_html=True)


def _clean_confidence(candidate: dict) -> str:
    conf = float(candidate.get("confidence", 0))
    overall = float(candidate.get("overall_match", 0))
    matched = [s for s in candidate.get("matched_skills", []) if s]
    if conf >= 0.75 and overall >= 70 and matched:
        return "High"
    if conf >= 0.5 and overall >= 50:
        return "Medium"
    return "Low"


def _render_recruiter_report(candidate: dict) -> None:
    """Render the AI-generated recruiter report in the drawer."""
    query = st.session_state.get("last_query", "")
    parsed = st.session_state.get("parsed_query") or {}
    report = build_recruiter_report(candidate, query, parsed)

    st.markdown(
        f"<h3 style='margin:0; font-size:1rem; color:#F8FAFC;'>Recruiter Report</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p class='muted' style='margin:0 0 0.6rem 0; font-size:0.75rem;'>{html.escape(report['name'])} • {html.escape(report['role'])}</p>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Match", f"{report['match']['overall']:.0f}%")
    c2.metric("Role Match", f"{report['match']['role']:.0f}%")
    c3.metric("Skill Match", f"{report['match']['skill']:.0f}%")
    c4.metric("Experience Match", f"{report['match']['experience']:.0f}%")

    st.markdown(
        f"<h4 style='margin:0.6rem 0 0.2rem; color:#F8FAFC; font-size:0.85rem;'>Hiring Recommendation</h4>"
        f"<p style='margin:0; color:#22C55E; font-size:0.9rem; font-weight:600;'>{html.escape(report['hiring_recommendation'])}</p>"
        f"<p style='margin:0 0 0.6rem; color:#94A3B8; font-size:0.75rem;'>{html.escape(report['hiring_explanation'])}</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h4 style='margin:0.6rem 0 0.2rem; color:#F8FAFC; font-size:0.85rem;'>Candidate Overview</h4>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<p style='margin:0; color:#CBD5E1; font-size:0.77rem; line-height:1.45;'>{html.escape(report['overview'])}</p>", unsafe_allow_html=True)

    def _bullets(items: list[str]) -> str:
        return "".join(f"<p style='margin:0.05rem 0; color:#CBD5E1; font-size:0.75rem;'>• {html.escape(str(i))}</p>" for i in items) or "<p style='font-size:0.75rem; color:#94A3B8;'>None identified.</p>"

    st.markdown("<h4 style='margin:0.6rem 0 0.2rem; color:#F8FAFC; font-size:0.85rem;'>Strengths</h4>", unsafe_allow_html=True)
    st.markdown(_bullets(report["strengths"]), unsafe_allow_html=True)

    st.markdown("<h4 style='margin:0.6rem 0 0.2rem; color:#F8FAFC; font-size:0.85rem;'>Potential Gaps</h4>", unsafe_allow_html=True)
    st.markdown(_bullets(report["gaps"]), unsafe_allow_html=True)

    st.markdown("<h4 style='margin:0.6rem 0 0.2rem; color:#F8FAFC; font-size:0.85rem;'>Relevant Skills</h4>", unsafe_allow_html=True)
    st.markdown(_bullets(report["relevant_skills"]), unsafe_allow_html=True)

    st.markdown("<h4 style='margin:0.6rem 0 0.2rem; color:#F8FAFC; font-size:0.85rem;'>Missing Skills</h4>", unsafe_allow_html=True)
    st.markdown(_bullets(report["missing_skills"]), unsafe_allow_html=True)

    st.markdown("<h4 style='margin:0.6rem 0 0.2rem; color:#F8FAFC; font-size:0.85rem;'>Relevant Projects</h4>", unsafe_allow_html=True)
    st.markdown(_bullets(report["relevant_projects"]), unsafe_allow_html=True)

    st.markdown("<h4 style='margin:0.6rem 0 0.2rem; color:#F8FAFC; font-size:0.85rem;'>Suggested Interview Questions</h4>", unsafe_allow_html=True)
    st.markdown(
        "".join(
            f"<p style='margin:0.1rem 0; color:#CBD5E1; font-size:0.75rem;'><b>{n}.</b> {html.escape(str(q))}</p>"
            for n, q in enumerate(report["interview_questions"], start=1)
        ),
        unsafe_allow_html=True,
    )

    st.markdown("---")
    md = format_markdown(report)
    text = format_text(report)
    e1, e2 = st.columns(2)
    with e1:
        st.download_button(
            label="Download Markdown",
            data=md,
            file_name=f"{report['name'].replace(' ', '_')}_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with e2:
        st.download_button(
            label="Download Text",
            data=text,
            file_name=f"{report['name'].replace(' ', '_')}_report.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with st.expander("Copy plain text"):
        st.text_area("Report text", text, height=200, label_visibility="collapsed")

    if st.button("Back to details", use_container_width=True, type="secondary"):
        st.session_state.drawer_tab = "details"
        st.rerun()


def _render_drawer(candidate: dict | None) -> None:
    if not candidate:
        return
    c = candidate
    hl_terms = list(c.get('matched_keywords', [])) + list(c.get('matched_skills', []))

    with st.container(border=True):
        # ── Header ──
        st.markdown(f"<h3 style='margin:0; font-size:1rem;'>{html.escape(c.get('name', 'Candidate'))}</h3>", unsafe_allow_html=True)
        role = html.escape(str(c.get('role') or 'Role not specified'))
        loc = html.escape(str(c.get('location') or ''))
        sub = f"{role} • {loc}" if loc else role
        st.markdown(f"<p class='muted' style='margin:0 0 0.6rem 0; font-size:0.72rem;'>{sub}</p>", unsafe_allow_html=True)

        pct = int(round(c.get('overall_match', 0)))
        conf = _clean_confidence(c)
        st.markdown(
            f"""
            <div class='match-circle-lg' style='--pct:{pct}%;'>
              <div class='match-circle-inner'>
                <span class='match-number'>{pct}%</span>
                <span class='match-label'>{conf}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Professional Summary ──
        _drawer_section("Professional Summary")
        summary = str(c.get('summary') or '').strip() or str(c.get('ai_summary') or '').split("\n\n")[0]
        if summary:
            st.markdown(f"<p style='margin:0; color:#94A3B8; font-size:0.77rem; line-height:1.4;'>{_highlight(summary[:400], hl_terms)}</p>", unsafe_allow_html=True)
        else:
            st.caption("No summary extracted.")

        # ── Experience ──
        _drawer_section("Experience")
        exp = str(c.get('experience') or 'Not specified')
        st.markdown(f"<p style='margin:0; color:#94A3B8; font-size:0.77rem;'>{html.escape(exp)}{' • ' + role if c.get('role') else ''}</p>", unsafe_allow_html=True)

        # ── Education ──
        _drawer_section("Education")
        education = [e for e in c.get('education', []) if e]
        if education:
            for e in education[:3]:
                st.markdown(f"<p style='margin:0.05rem 0; color:#94A3B8; font-size:0.77rem;'>🎓 {_highlight(str(e)[:120], hl_terms)}</p>", unsafe_allow_html=True)
        else:
            st.caption("No education extracted.")

        # ── Projects ──
        projects = [p for p in c.get('projects', []) if p]
        if projects:
            _drawer_section("Projects")
            for p in projects[:4]:
                st.markdown(f"<p style='margin:0.05rem 0; color:#94A3B8; font-size:0.77rem;'>• {_highlight(str(p)[:120], hl_terms)}</p>", unsafe_allow_html=True)

        # ── Skills ──
        skills = c.get('skills') or c.get('top_skills') or []
        if skills:
            _drawer_section("Skills")
            matched_lower = [m.lower() for m in c.get('matched_skills', [])]
            chips = []
            for s in skills[:15]:
                cls = "skill-chip matched" if s.lower() in matched_lower else "skill-chip"
                chips.append(f"<span class='{cls}'>{html.escape(s)}</span>")
            if len(skills) > 15:
                chips.append(f"<span class='skill-chip more'>+{len(skills) - 15}</span>")
            st.markdown("".join(chips), unsafe_allow_html=True)

        # ── Retrieved Chunks ──
        with st.expander("Retrieved Chunks"):
            chunks = c.get('retrieved_chunks', [])
            if not chunks:
                st.caption("No retrieved chunks available.")
            for i, chunk in enumerate(chunks, start=1):
                src = html.escape(str(chunk.get('source', 'hybrid')).replace('RetrievalSource.', '').lower())
                score = float(chunk.get('score', 0.0))
                text = _highlight(str(chunk.get('text', '')), hl_terms)
                st.markdown(f"<p style='margin:0 0 0.15rem 0; font-size:0.72rem; color:#F8FAFC; font-weight:600;'>Chunk {i} — {src} (score {score:.4f})</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='margin:0 0 0.5rem 0; font-size:0.72rem; color:#94A3B8;'>{text}</p>", unsafe_allow_html=True)

        # ── Why this matched ──
        _drawer_section("Why this matched")
        match_details = c.get('match_details') or []
        for detail in match_details:
            label = html.escape(str(detail.get('label', '')))
            score = detail.get('score', 0)
            value = str(detail.get('value', '')).strip()
            value_html = _highlight(html.escape(value), hl_terms) if value else ""
            if value_html:
                st.markdown(
                    f"<p style='margin:0.05rem 0; color:#94A3B8; font-size:0.77rem;'>"
                    f"✓ {label}: {value_html} <span style='color:#22C55E; font-weight:600;'>{score}%</span></p>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<p style='margin:0.05rem 0; color:#94A3B8; font-size:0.77rem;'>"
                    f"✓ {label} <span style='color:#22C55E; font-weight:600;'>{score}%</span></p>",
                    unsafe_allow_html=True,
                )

        # ── Matched metadata ──
        _drawer_section("Matched Details")

        def _fmt_list(values: list | tuple | None, label: str) -> str | None:
            values = [str(v).strip() for v in (values or []) if v]
            if not values:
                return None
            return f"<span style='color:#F8FAFC; font-weight:600;'>{html.escape(label)}:</span> {_highlight(html.escape(', '.join(values)), hl_terms)}"

        def _fmt_scalar(value, label: str) -> str | None:
            text = str(value).strip() if value is not None else ""
            if not text:
                return None
            return f"<span style='color:#F8FAFC; font-weight:600;'>{html.escape(label)}:</span> {_highlight(html.escape(text), hl_terms)}"

        for label, key in [
            ("Matched Role", "matched_role"),
            ("Matched Skills", "matched_skills"),
            ("Matched Experience", "matched_experience"),
            ("Matched Education", "matched_education"),
            ("Matched Industry", "matched_industry"),
        ]:
            raw = c.get(key)
            if isinstance(raw, (list, tuple)):
                html_out = _fmt_list(raw, label)
            else:
                html_out = _fmt_scalar(raw, label)
            if html_out:
                st.markdown(
                    f"<p style='margin:0.05rem 0; color:#94A3B8; font-size:0.77rem;'>{html_out}</p>",
                    unsafe_allow_html=True,
                )

        sections = c.get('retrieved_sections') or c.get('matched_sections') or []
        if sections:
            joined = _highlight(html.escape(", ".join(sections)), hl_terms)
            st.markdown(
                f"<p style='margin:0.05rem 0; color:#94A3B8; font-size:0.77rem;'>"
                f"<span style='color:#F8FAFC; font-weight:600;'>Retrieved Resume Sections:</span> {joined}</p>",
                unsafe_allow_html=True,
            )

        chunk_ids = c.get('retrieved_chunk_ids') or []
        if chunk_ids:
            joined = html.escape(", ".join(str(cid) for cid in chunk_ids if cid))
            st.markdown(
                f"<p style='margin:0.05rem 0; color:#94A3B8; font-size:0.77rem;'>"
                f"<span style='color:#F8FAFC; font-weight:600;'>Retrieved Chunk IDs:</span> {joined}</p>",
                unsafe_allow_html=True,
            )

        metadata_score = c.get('metadata_score')
        if metadata_score is not None:
            st.markdown(
                f"<p style='margin:0.05rem 0; color:#94A3B8; font-size:0.77rem;'>"
                f"<span style='color:#F8FAFC; font-weight:600;'>Matched Metadata:</span> {round(metadata_score, 2)}</p>",
                unsafe_allow_html=True,
            )

        confidence = c.get('confidence')
        if confidence is not None:
            st.markdown(
                f"<p style='margin:0.05rem 0; color:#94A3B8; font-size:0.77rem;'>"
                f"<span style='color:#F8FAFC; font-weight:600;'>Confidence score:</span> {round(confidence, 2)}</p>",
                unsafe_allow_html=True,
            )

        if st.button("Close", use_container_width=True):
            st.session_state.selected_candidate = None
            st.rerun()


def _render_performance_panel(bundle, search_service) -> None:
    def _table(title: str, rows: dict[str, float]) -> str:
        html_rows = ""
        for label, ms in rows.items():
            style = "color:#F87171; font-weight:600;" if ms > 200 else "color:#F8FAFC;"
            html_rows += (
                f"<tr>"
                f"<td style='padding:0.2rem 0.5rem; color:#94A3B8; white-space:nowrap;'>{html.escape(label)}</td>"
                f"<td style='padding:0.2rem 0.5rem; text-align:right; {style}'>{ms:.1f} ms</td>"
                f"</tr>"
            )
        return (
            f"<h5 style='margin:0.5rem 0 0.25rem 0; color:#F8FAFC; font-size:0.85rem;'>{html.escape(title)}</h5>"
            f"<table style='width:100%; font-size:0.78rem; border-collapse:collapse;'>"
            f"{html_rows}"
            f"</table>"
        )

    startup = getattr(bundle, "startup_metrics", {}) or {}
    startup_rows = {
        "Vector Store": startup.get("vector_store_ms", 0.0),
        "Embedding Service Init": startup.get("embedding_service_ms", 0.0),
        "BM25 Init": startup.get("bm25_init_ms", 0.0),
        "BM25 Load": startup.get("bm25_load_ms", 0.0),
        "Dense Service Init": startup.get("dense_service_ms", 0.0),
        "Sparse Service Init": startup.get("sparse_service_ms", 0.0),
        "Hybrid Service Init": startup.get("hybrid_service_ms", 0.0),
        "Reranker Init": startup.get("reranker_init_ms", 0.0),
        "Embedding Model Load": startup.get("embedding_model_load_ms", 0.0),
        "Cross-Encoder Load": startup.get("cross_encoder_load_ms", 0.0),
        "Total Startup": startup.get("total_ms", 0.0),
    }
    st.markdown(_table("Startup", startup_rows), unsafe_allow_html=True)

    if not (search_service and getattr(search_service, "last_search_metrics", None)):
        st.caption("No search has been run this session.")
        return

    m = search_service.last_search_metrics
    ui_ms = 0.0
    if "last_search_end" in st.session_state:
        ui_ms = (time.perf_counter() - st.session_state.last_search_end) * 1000
        m["ui_ms"] = ui_ms

    search_rows = {
        "Query Parse": m.get("query_parse_ms", 0.0),
        "Embedding": m.get("embedding_time_ms", 0.0),
        "Dense Retrieval": m.get("dense_retrieval_ms", 0.0),
        "Sparse Retrieval": m.get("sparse_retrieval_ms", 0.0),
        "Fusion": m.get("fusion_ms", 0.0),
        "Metadata Scoring": m.get("metadata_scoring_ms", 0.0),
        "Summary Generation": m.get("summary_ms", 0.0),
        "Reranking": m.get("rerank_time_ms", 0.0),
        "UI Rendering": ui_ms,
        "Total Search": m.get("latency_ms", 0.0),
        "Total (with UI)": m.get("latency_ms", 0.0) + ui_ms,
    }
    st.markdown(_table("Last Search", search_rows), unsafe_allow_html=True)


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


def _highlight(text: str, terms: list[str]) -> str:
    """HTML-escape text and wrap query-term matches in a highlight mark."""
    import re as _re
    escaped = html.escape(text)
    clean_terms = sorted({t.strip() for t in terms if t and len(t.strip()) >= 2}, key=len, reverse=True)
    for term in clean_terms:
        pattern = _re.compile(rf"(?<![\w])({_re.escape(html.escape(term))})(?![\w])", _re.IGNORECASE)
        escaped = pattern.sub(r"<mark class='query-hit'>\1</mark>", escaped)
    return escaped


def _drawer_section(title: str) -> None:
    st.markdown(f"<p style='font-weight:600; color:#F8FAFC; margin:0.6rem 0 0.25rem; font-size:0.8rem;'>{title}</p>", unsafe_allow_html=True)


def _why_matched(candidate: dict) -> list[str]:
    reasons = []
    matched_skills = [s for s in candidate.get("matched_skills", []) if s]
    if matched_skills:
        reasons.append("Skills: " + ", ".join(matched_skills[:4]))
    keywords = [k for k in candidate.get("matched_keywords", []) if k]
    if keywords:
        reasons.append("Terms: " + ", ".join(keywords[:4]))
    sections = [s for s in candidate.get("matched_sections", []) if s]
    if sections:
        reasons.append("Evidence: " + ", ".join(sections[:3]))
    if not reasons:
        reasons.append("Semantic match")
    return reasons


def _render_explainability_panel(candidate: dict, idx: int) -> None:
    """Render an expandable, recruiter-friendly explanation of why a candidate matched."""

    # --- Why this candidate? ---
    st.markdown(
        "<p style='font-weight:600; color:#F8FAFC; margin:0.6rem 0 0.35rem; font-size:0.82rem;'>Why This Candidate?</p>",
        unsafe_allow_html=True,
    )
    match_details = candidate.get("match_details") or []
    if not match_details:
        st.caption("No specific match details available.")
    else:
        n_cols = 2
        for i in range(0, len(match_details), n_cols):
            cols = st.columns(n_cols)
            for j, detail in enumerate(match_details[i : i + n_cols]):
                label = html.escape(str(detail.get("label", "")))
                value = html.escape(str(detail.get("value", "")))
                with cols[j]:
                    if value and value.lower() != label.lower():
                        st.markdown(
                            f"<p style='margin:0.05rem 0; font-size:0.74rem; color:#F8FAFC;'>"
                            f"<span style='color:#22C55E; margin-right:0.3rem;'>✓</span>"
                            f"<b>{label}</b>: {value}"
                            f"</p>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"<p style='margin:0.05rem 0; font-size:0.74rem; color:#F8FAFC;'>"
                            f"<span style='color:#22C55E; margin-right:0.3rem;'>✓</span>"
                            f"<b>{label}</b>"
                            f"</p>",
                            unsafe_allow_html=True,
                        )

    # --- Matched resume sections ---
    sections = [s for s in candidate.get("retrieved_sections", []) if s]
    if sections:
        st.markdown(
            "<p style='font-weight:600; color:#F8FAFC; margin:0.6rem 0 0.35rem; font-size:0.82rem;'>Matched Resume Sections</p>",
            unsafe_allow_html=True,
        )
        display = " • ".join(html.escape(str(s).title()) for s in sections[:6])
        st.markdown(
            f"<p style='margin:0; font-size:0.74rem; color:#94A3B8;'>{display}</p>",
            unsafe_allow_html=True,
        )

    # --- Matched resume chunks ---
    chunks = candidate.get("retrieved_chunks", [])[:3]
    if chunks:
        st.markdown(
            "<p style='font-weight:600; color:#F8FAFC; margin:0.6rem 0 0.35rem; font-size:0.82rem;'>Matched Resume Chunks</p>",
            unsafe_allow_html=True,
        )
        for n, chunk in enumerate(chunks, start=1):
            section = html.escape(str(chunk.get("section") or chunk.get("source") or "Unknown"))
            text = chunk.get("matched_text") or chunk.get("text") or chunk.get("preview") or chunk.get("content") or ""
            text = " ".join(text.split())
            snippet = text[:180] + "…" if len(text) > 180 else text
            snippet = html.escape(snippet)
            score = chunk.get("score")
            if isinstance(score, (int, float)):
                score_text = f"{score:.2f}"
            else:
                score_text = "N/A"
            st.markdown(
                f"<p style='margin:0.15rem 0 0.05rem; font-size:0.75rem; color:#F8FAFC;'>"
                f"<b>Chunk {n}</b> <span style='color:#94A3B8;'>| {section}</span>"
                f"</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='margin:0; font-size:0.72rem; color:#CBD5E1; line-height:1.35;'>"
                f"{snippet}"
                f"</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='margin:0 0 0.3rem; font-size:0.7rem; color:#94A3B8;'>Score: {score_text}</p>",
                unsafe_allow_html=True,
            )

    # --- Ranking breakdown ---
    st.markdown(
        "<p style='font-weight:600; color:#F8FAFC; margin:0.6rem 0 0.35rem; font-size:0.82rem;'>Ranking Breakdown</p>",
        unsafe_allow_html=True,
    )
    breakdown = [
        ("Dense Similarity", candidate.get("semantic_match")),
        ("Sparse Match", candidate.get("keyword_match")),
        ("Role Match", candidate.get("role_match")),
        ("Skill Match", candidate.get("skill_match")),
        ("Experience Match", candidate.get("experience_match")),
        ("Cross Encoder", candidate.get("rerank_match")),
        ("Final Score", candidate.get("overall_match")),
    ]
    for label, value in breakdown:
        if value is None or value == "N/A":
            continue
        if not isinstance(value, (int, float)):
            continue
        pct = min(1.0, max(0.0, value / 100.0))
        st.markdown(
            f"<p style='margin:0.05rem 0; font-size:0.72rem; color:#F8FAFC;'>"
            f"{html.escape(label)}: <b>{value:.0f}%</b>"
            f"</p>",
            unsafe_allow_html=True,
        )
        st.progress(pct, text=None)


cache_key = _compute_cache_key()

if st.session_state.get("_bundle_cache_key") == cache_key:
    logging.info("[CACHE-HIT] Reusing cached retrieval bundle for cache_key=%s", cache_key)
st.session_state._bundle_cache_key = cache_key

if not st.session_state.bootstrap_complete:
    with st.spinner("Initializing TalentLens indexes..."):
        st.session_state.bootstrap_result = _run_bootstrap(cache_key)
        st.session_state.bootstrap_complete = True
    st.rerun()

bundle = _get_retrieval_bundle(cache_key)
st.session_state.search_service = _get_search_service(cache_key)
app_ready_ms = (time.perf_counter() - APP_START) * 1000
print(f"[STARTUP] App Ready: {app_ready_ms:.2f}ms")


with st.sidebar:
    st.markdown("<h2 style='margin:0; color:#F8FAFC; font-size:1.35rem;'>TalentLens</h2>", unsafe_allow_html=True)
    st.markdown("<p class='muted' style='margin:0 0 1.5rem 0; font-size:0.75rem;'>AI Resume Intelligence</p>", unsafe_allow_html=True)

    nav = st.radio(
        "Navigation",
        ["🏠 Dashboard", "⭐ Shortlist"],
        index=0 if st.session_state.page == "Dashboard" else 1,
        label_visibility="collapsed",
    )
    st.session_state.page = "Dashboard" if "Dashboard" in nav else "Shortlist"

    st.markdown("---")
    st.markdown("<p style='font-weight:600; color:#F8FAFC; font-size:0.8rem;'>System Status</p>", unsafe_allow_html=True)
    vector_count = bundle.vector_store_service.count()
    bm25_count = bundle.bm25_index.total_documents if hasattr(bundle.bm25_index, "total_documents") else 0
    st.markdown("<span class='status-dot dot-green'></span>Vector Store Connected", unsafe_allow_html=True)
    st.markdown("<span class='status-dot dot-blue'></span>BM25 Ready", unsafe_allow_html=True)
    st.markdown("<span class='status-dot dot-purple'></span>RAG Pipeline Active", unsafe_allow_html=True)
    st.markdown(f"<p class='muted' style='font-size:0.7rem; margin-top:0.5rem;'>{vector_count} vectors • {bm25_count} indexed</p>", unsafe_allow_html=True)


selected = st.session_state.selected_candidate
if selected:
    main_col, drawer_col = st.columns([3.0, 1.0])
else:
    main_col = st.container()
    drawer_col = None

with main_col:
    if st.session_state.page == "Dashboard":
        st.markdown("<h1 style='font-size:1.5rem; margin:0;'>TalentLens</h1>", unsafe_allow_html=True)
        st.markdown("<p class='muted' style='margin:0 0 0.25rem 0; font-size:0.78rem;'>Search thousands of resumes using natural language.</p>", unsafe_allow_html=True)

        with st.container(border=True):
            with st.form("search_form", clear_on_submit=False):
                c1, c2 = st.columns([4, 1])
                with c1:
                    user_query = st.text_input(
                        "Search",
                        placeholder="Finance Manager with Excel and Banking experience",
                        label_visibility="collapsed",
                    )
                with c2:
                    submitted = st.form_submit_button("Search", type="primary", use_container_width=True, help="Press Enter in the query box to search")

                st.markdown("<p style='font-weight:600; color:#F8FAFC; margin:0.6rem 0 0.25rem; font-size:0.8rem;'>Filters</p>", unsafe_allow_html=True)
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
                    skills_filter = st.multiselect("Skills", skill_pool, placeholder="Any")
                with f5:
                    max_results = st.selectbox("Max Results", [5, 10, 15, 20], index=1)
                with f6:
                    reset = st.form_submit_button("Reset", type="secondary")

        if reset:
            st.session_state.search_results = []
            st.session_state.selected_candidate = None
            st.rerun()

        if submitted and user_query.strip():
            st.session_state.is_loading = True
            st.session_state.search_error = None
            st.session_state.displayed_count = max_results
            st.rerun()

        # Execute pending search (used to support retry after an error)
        if st.session_state.is_loading:
            with st.spinner("Retrieving candidates..."):
                from src.search.schema import SearchFilters

                _exp_min = exp_range[0] if exp_range[0] > 0 else None
                _exp_max = exp_range[1] if exp_range[1] < 20 else None
                _loc = location_filter.strip() if location_filter else None
                _edu = education_filter.strip() if education_filter else None
                _skills = [s.lower() for s in skills_filter] if skills_filter else None

                filters = SearchFilters(
                    skills=_skills,
                    location=_loc,
                    experience_min=_exp_min,
                    experience_max=_exp_max,
                    education=_edu,
                    strict=False,
                )

                try:
                    # Parse query intent for display.
                    parsed = st.session_state.search_service.parse_query(user_query)
                    st.session_state.parsed_query = parsed.display_dict()
                    st.session_state.last_query = user_query

                    results = st.session_state.search_service.search(
                        query=user_query,
                        top_k=max_results,
                        filters=filters,
                    )
                    st.session_state.search_results = [r.to_frontend_dict() for r in results]
                    st.session_state.last_search_end = time.perf_counter()
                    st.session_state.selected_candidate = None
                    st.session_state.is_loading = False
                    st.toast(f"Found {len(results)} candidates", icon="🎯")
                except Exception:
                    st.session_state.is_loading = False
                    st.session_state.search_error = "Search failed. Please try again."
                    st.session_state.search_results = []
                    logging.error("Search failed for query: %r", user_query, exc_info=True)
                    st.toast("Search failed. Please try again.", icon="⚠️")

        if st.session_state.get("search_error"):
            with st.container(border=True):
                st.error(st.session_state.search_error)
                if st.button("Retry search", type="primary"):
                    st.session_state.search_error = None
                    st.session_state.is_loading = True
                    st.rerun()

        # Skeleton loaders while the first batch of results is loading
        if st.session_state.get("is_loading") and not st.session_state.search_results:
            _render_skeletons(n=3)

        # Display parsed query intent
        if st.session_state.get("parsed_query"):
            with st.container(border=True):
                st.markdown("<p style='font-weight:600; color:#F8FAFC; margin:0 0 0.35rem 0; font-size:0.8rem;'>Parsed query</p>", unsafe_allow_html=True)
                cols = st.columns(6)
                labels = ["Role", "Industry", "Skills", "Experience", "Education", "Location"]
                for col, label in zip(cols, labels):
                    with col:
                        val = st.session_state.parsed_query.get(label, "Not specified")
                        st.markdown(f"<p style='margin:0; font-size:0.65rem; color:#94A3B8;'>{label}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='margin:0; font-weight:600; font-size:0.77rem; color:#F8FAFC;'>{html.escape(str(val))}</p>", unsafe_allow_html=True)

        if st.session_state.search_results:
            total = bm25_count if bm25_count else 2484
            n_results = len(st.session_state.search_results)
            high = sum(1 for c in st.session_state.search_results if c.get("overall_match", 0) >= 90)
            good = sum(1 for c in st.session_state.search_results if 70 <= c.get("overall_match", 0) < 90)
            fair = n_results - high - good

            stat1, stat2, stat3, stat4, stat5 = st.columns(5)
            stat_data = [
                ("Indexed", total, "idx", "I"),
                ("Results", n_results, "res", "R"),
                ("High (≥90%)", high, "high", "H"),
                ("Good (70-89%)", good, "good", "G"),
                ("Fair (<70%)", fair, "fair", "F"),
            ]
            for col, (label, val, cls, icon) in zip([stat1, stat2, stat3, stat4, stat5], stat_data):
                with col:
                    with st.container(border=True):
                        st.markdown(f"""
                            <div class='stat-card'>
                                <span class='stat-icon {cls}'>{icon}</span>
                                <p class='muted' style='margin:0; font-size:0.65rem;'>{label}</p>
                                <p style='margin:0; font-weight:700; font-size:1.1rem;'>{val}</p>
                            </div>
                        """, unsafe_allow_html=True)

            # Advanced retrieval analytics
            analytics = getattr(st.session_state.search_service, "last_search_metrics", None)
            if analytics:
                with st.expander("Advanced Analytics"):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Dense Candidates", analytics.get("dense_candidates", 0))
                    c2.metric("Sparse Candidates", analytics.get("sparse_candidates", 0))
                    c3.metric("RRF Candidates", analytics.get("rrf_candidates", 0))
                    c4.metric("Cross-Encoder Candidates", analytics.get("cross_encoder_candidates", 0))
                    c5, c6, c7, c8, c9 = st.columns(5)
                    c5.metric("Latency", f"{analytics.get('latency_ms', 0):.1f} ms")
                    c6.metric("Embedding Time", f"{analytics.get('embedding_time_ms', 0):.1f} ms")
                    c7.metric("Retrieval Time", f"{analytics.get('retrieval_time_ms', 0):.1f} ms")
                    c8.metric("Rerank Time", f"{analytics.get('rerank_time_ms', 0):.1f} ms")
                    c9.metric("Generation Time", f"{analytics.get('generation_time_ms', 0):.1f} ms")

            for i, c in enumerate(st.session_state.search_results[:st.session_state.displayed_count]):
                with st.container(border=True):
                    r1, r2, r3 = st.columns([0.08, 0.70, 0.22])

                    with r1:
                        st.markdown(f"<div class='avatar-circle' aria-label='{html.escape(c.get('name', 'Candidate'))} avatar' title='{html.escape(c.get('name', 'Candidate'))}'>{_initials(c.get('name', ''))}</div>", unsafe_allow_html=True)

                    with r2:
                        st.markdown(f"<h4 style='margin:0; color:#F8FAFC; font-size:0.92rem;'>{html.escape(c.get('name', 'Unknown'))}</h4>", unsafe_allow_html=True)
                        role = html.escape(str(c.get('role') or 'Role not specified'))
                        loc = html.escape(str(c.get('location') or 'Location not specified'))
                        exp = html.escape(str(c.get('experience') or 'Experience not specified'))
                        st.markdown(f"<p class='muted' style='margin:0; font-size:0.7rem;'>{role} • {loc} • {exp}</p>", unsafe_allow_html=True)

                        chips = []
                        for s in c.get('top_skills', [])[:4]:
                            matched = s.lower() in [m.lower() for m in c.get('matched_skills', [])]
                            cls = "skill-chip matched" if matched else "skill-chip"
                            chips.append(f"<span class='{cls}'>{html.escape(s)}</span>")
                        if c.get('extra_skills', 0) > 0:
                            chips.append(f"<span class='skill-chip more'>+{c['extra_skills']}</span>")
                        st.markdown("".join(chips), unsafe_allow_html=True)

                        summary = html.escape(str(c.get('ai_summary') or ''))
                        st.markdown(f"<p class='ai-summary'>{summary}</p>", unsafe_allow_html=True)

                    with r3:
                        pct = int(round(c.get('overall_match', 0)))
                        conf = _clean_confidence(c)
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
                            if st.button("View", key=f"v_{c['id']}_{i}", use_container_width=True, help="Open candidate details"):
                                st.session_state.selected_candidate = c
                                st.session_state.drawer_tab = "details"
                                st.rerun()
                        with b2:
                            if st.button("Shortlist", key=f"s_{c['id']}_{i}", use_container_width=True, type="primary", help="Save this candidate to your shortlist"):
                                _add_to_shortlist(c)
                                st.toast("Added to shortlist")
                        if st.button("Report", key=f"r_{c['id']}_{i}", use_container_width=True, type="secondary", help="Generate recruiter report"):
                            st.session_state.selected_candidate = c
                            st.session_state.drawer_tab = "report"
                            st.rerun()

                    with st.expander("Explainability"):
                        _render_explainability_panel(c, i)

            if st.session_state.displayed_count < len(st.session_state.search_results):
                if st.button("Load more", key="load_more", use_container_width=True, type="secondary"):
                    st.session_state.displayed_count = min(
                        st.session_state.displayed_count + 10,
                        len(st.session_state.search_results)
                    )
                    st.rerun()

    else:
        st.markdown("<h1 style='font-size:1.5rem; margin-bottom:0.1rem;'>Shortlist</h1>", unsafe_allow_html=True)
        st.markdown("<p class='muted' style='margin-bottom:0.6rem; font-size:0.78rem;'>Saved candidates for this search session.</p>", unsafe_allow_html=True)

        if not st.session_state.shortlist:
            st.info("No candidates have been shortlisted yet.")
        else:
            for i, cid in enumerate(st.session_state.shortlist):
                c = st.session_state.shortlist_map.get(cid)
                if not c:
                    continue
                with st.container(border=True):
                    r1, r2, r3 = st.columns([0.08, 0.70, 0.22])
                    with r1:
                        st.markdown(f"<div class='avatar-circle' aria-label='{html.escape(c.get('name', 'Candidate'))} avatar' title='{html.escape(c.get('name', 'Candidate'))}'>{_initials(c.get('name', ''))}</div>", unsafe_allow_html=True)
                    with r2:
                        st.markdown(f"<h4 style='margin:0; color:#F8FAFC; font-size:0.92rem;'>{html.escape(c.get('name', 'Unknown'))}</h4>", unsafe_allow_html=True)
                        role = html.escape(str(c.get('role') or 'Role not specified'))
                        loc = html.escape(str(c.get('location') or 'Location not specified'))
                        st.markdown(f"<p class='muted' style='margin:0; font-size:0.7rem;'>{role} • {loc}</p>", unsafe_allow_html=True)
                        chips = []
                        for s in c.get('top_skills', [])[:4]:
                            matched = s.lower() in [m.lower() for m in c.get('matched_skills', [])]
                            cls = "skill-chip matched" if matched else "skill-chip"
                            chips.append(f"<span class='{cls}'>{html.escape(s)}</span>")
                        if c.get('extra_skills', 0) > 0:
                            chips.append(f"<span class='skill-chip more'>+{c['extra_skills']}</span>")
                        st.markdown("".join(chips), unsafe_allow_html=True)
                        summary = html.escape(str(c.get('ai_summary') or ''))
                        st.markdown(f"<p class='ai-summary'>{summary}</p>", unsafe_allow_html=True)
                    with r3:
                        pct = int(round(c.get('overall_match', 0)))
                        conf = _clean_confidence(c)
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
                            if st.button("View", key=f"sv_{c['id']}_{i}", use_container_width=True, help="Open candidate details"):
                                st.session_state.selected_candidate = c
                                st.rerun()
                        with b2:
                            if st.button("Remove", key=f"sr_{c['id']}_{i}", use_container_width=True, type="secondary"):
                                _remove_from_shortlist(cid)
                                st.toast("Removed from shortlist", icon="🗑️")
                                st.rerun()


    with st.expander("Performance", expanded=False):
        _render_performance_panel(bundle, st.session_state.search_service)


if drawer_col is not None:
    with drawer_col:
        if st.session_state.get("selected_candidate") and st.session_state.get("drawer_tab") == "report":
            _render_recruiter_report(st.session_state.selected_candidate)
        else:
            _render_drawer(st.session_state.selected_candidate)
