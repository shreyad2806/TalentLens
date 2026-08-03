import csv
import io
import random
import sys
import time
import warnings
from typing import Any

import streamlit as st

# Force UTF-8 stdout to avoid emoji print errors from underlying services.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

warnings.filterwarnings("ignore")

st.set_page_config(page_title="TalentLens Retrieval Benchmark", layout="wide")


@st.cache_resource(show_spinner=True)
def _load_search_service():
    from src.bootstrap.composition_root import create_retrieval_bundle
    from src.search.search_service import SearchService

    bundle = create_retrieval_bundle()
    return SearchService(hybrid_service=bundle.hybrid_service)


def _build_sample_query(resume: Any, skill_pool: list[str], location_pool: list[str], exp_pool: list[str]) -> str:
    """Build a query that should match the chosen resume."""
    parts: list[str] = []

    role = (resume.role or "").strip()
    if role:
        parts.append(role)
    else:
        parts.append("professional")

    skills = [s for s in (resume.skills or [])]
    if skills:
        parts.append(f"with {skills[0]}")
    elif skill_pool:
        parts.append(f"with {random.choice(skill_pool)}")

    loc = (resume.location or "").strip()
    if loc:
        parts.append(f"in {loc}")
    elif random.random() < 0.4 and location_pool:
        parts.append(f"in {random.choice(location_pool)}")

    if random.random() < 0.3 and exp_pool:
        parts.append(random.choice(exp_pool))

    if resume.education:
        edu = next((e for e in resume.education if getattr(e, "degree", None)), None)
        if edu and getattr(edu, "degree", None) and random.random() < 0.2:
            parts.append(str(edu.degree))

    return " ".join(parts).strip()


def _generate_sample_queries(resumes: list[Any], n: int = 100, seed: int = 42) -> list[tuple[str, str]]:
    """Return (query, golden_resume_id) pairs."""
    random.seed(seed)
    location_pool = ["Bangalore", "Mumbai", "Delhi", "Hyderabad", "Pune", "Chennai", "Noida", "Gurgaon"]
    exp_pool = ["2+ years", "3-5 years", "5+ years"]
    skill_pool = [
        "Python", "SQL", "AWS", "Docker", "Kubernetes", "React", "Node.js", "Java",
        "Machine Learning", "AI", "RAG", "LLMs", "PostgreSQL", "MongoDB", "Git",
        "CI/CD", "Flask", "Django", "Spark", "Excel", "Tableau", "Power BI",
        "TypeScript", "Next.js", "GraphQL", "Redis", "Azure", "GCP", "Salesforce",
        "Spring", "C++", "HTML", "CSS",
    ]

    samples = random.choices(resumes, k=n) if len(resumes) < n else random.sample(resumes, n)
    queries: list[tuple[str, str]] = []
    for resume in samples:
        q = _build_sample_query(resume, skill_pool, location_pool, exp_pool)
        queries.append((q, str(resume.resume_id)))
    return queries


def _run_benchmark(search_service, queries: list[tuple[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, (query, golden_id) in enumerate(queries, start=1):
        t0 = time.perf_counter()
        results = search_service.search(query, top_k=10)
        query_latency = (time.perf_counter() - t0) * 1000

        result_ids = [r.resume_metadata.resume_id for r in results]
        golden_rank = result_ids.index(golden_id) + 1 if golden_id in result_ids else 0
        recall_at_10 = 1.0 if golden_rank > 0 and golden_rank <= 10 else 0.0
        precision_at_10 = 1.0 / 10.0 if recall_at_10 else 0.0
        mrr = 1.0 / golden_rank if golden_rank > 0 else 0.0

        metrics = getattr(search_service, "last_search_metrics", {}) or {}
        hybrid_metrics = getattr(search_service.hybrid_service, "last_metrics", None)
        if hybrid_metrics is not None:
            hybrid_overlap = getattr(hybrid_metrics, "overlap_count", 0)
            hybrid_fused = getattr(hybrid_metrics, "fused_candidate_count", 1) or 1
            hybrid_recall = hybrid_overlap / hybrid_fused
        else:
            hybrid_recall = 0.0

        rows.append({
            "query": query,
            "golden_id": golden_id,
            "found": golden_rank > 0,
            "rank": golden_rank,
            "recall@10": recall_at_10,
            "precision@10": precision_at_10,
            "mrr": mrr,
            "hybrid_recall": hybrid_recall,
            "latency_ms": query_latency,
            "dense_candidates": metrics.get("dense_candidates", 0),
            "sparse_candidates": metrics.get("sparse_candidates", 0),
            "rrf_candidates": metrics.get("rrf_candidates", 0),
            "cross_encoder_candidates": metrics.get("cross_encoder_candidates", 0),
        })
    return rows


def _to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main():
    st.title("TalentLens Retrieval Benchmark")
    st.markdown(
        "Evaluate the retrieval pipeline with 100 synthetic known-item queries. "
        "For each query a resume is chosen from the index and a query is generated from its role, "
        "skills, location and education."
    )

    search_service = _load_search_service()
    resume_cache = getattr(search_service, "_resume_cache", {})
    if not resume_cache:
        st.error("No resumes available to build sample queries.")
        st.stop()

    resumes = list(resume_cache.values())
    st.write(f"Loaded {len(resumes)} resumes.")

    n_queries = st.number_input("Number of sample queries", min_value=10, max_value=500, value=100, step=10)
    seed = st.number_input("Random seed", min_value=0, max_value=100000, value=42, step=1)

    queries = _generate_sample_queries(resumes, n=n_queries, seed=seed)

    if st.button("Run Benchmark", type="primary"):
        with st.spinner("Running benchmark... this may take a few minutes."):
            rows = _run_benchmark(search_service, queries)

        st.divider()

        # Summary metrics
        avg_recall = _mean([r["recall@10"] for r in rows])
        avg_mrr = _mean([r["mrr"] for r in rows])
        avg_precision = _mean([r["precision@10"] for r in rows])
        avg_hybrid_recall = _mean([r["hybrid_recall"] for r in rows])
        avg_latency = _mean([r["latency_ms"] for r in rows])

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Recall@10", f"{avg_recall:.3f}")
        c2.metric("MRR", f"{avg_mrr:.3f}")
        c3.metric("Precision@10", f"{avg_precision:.3f}")
        c4.metric("Hybrid Recall", f"{avg_hybrid_recall:.3f}")
        c5.metric("Avg Latency", f"{avg_latency:.1f} ms")

        st.subheader("Metric Averages")
        st.bar_chart(
            {
                "Metric": ["Recall@10", "MRR", "Precision@10", "Hybrid Recall"],
                "Value": [avg_recall, avg_mrr, avg_precision, avg_hybrid_recall],
            },
            x="Metric",
            y="Value",
        )

        st.subheader("Per-Query Latency")
        st.line_chart(
            {
                "query index": list(range(len(rows))),
                "latency_ms": [r["latency_ms"] for r in rows],
            },
            x="query index",
            y="latency_ms",
        )

        st.subheader("Per-Query Recall@10")
        st.bar_chart(
            {
                "query index": list(range(len(rows))),
                "recall@10": [r["recall@10"] for r in rows],
            },
            x="query index",
            y="recall@10",
        )

        st.subheader("Results")
        st.dataframe(rows, use_container_width=True)

        st.download_button(
            label="Export results as CSV",
            data=_to_csv(rows),
            file_name="retrieval_benchmark.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
