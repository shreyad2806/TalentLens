"""Test search and UI card rendering for the required queries.

Outputs a clean report to test_search_report.txt (diagnostic stdout is suppressed
during the search calls).
"""
from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging

logging.basicConfig(level=logging.CRITICAL)
logging.disable(logging.CRITICAL)

from src.bootstrap.composition_root import create_retrieval_bundle
from src.search.search_service import SearchService
from src.cards.candidate_card import build_candidate_card


def main():
    bundle = create_retrieval_bundle()
    search_service = SearchService(hybrid_service=bundle.hybrid_service)

    queries = ["python", "java", "banking", "spring boot", "excel"]
    report_path = PROJECT_ROOT / "test_search_report.txt"

    with open(report_path, "w", encoding="utf-8") as report:
        for query in queries:
            report.write("\n" + "=" * 80 + "\n")
            report.write(f"QUERY: {query!r}\n")
            report.write("=" * 80 + "\n")

            # Silence noisy diagnostic prints during the retrieval + scoring pipeline.
            with redirect_stdout(io.StringIO()):
                results = search_service.search(query=query, top_k=10)

            report.write(f"\nReturned {len(results)} results\n\n")

            for i, r in enumerate(results, 1):
                m = r.resume_metadata
                s = r.score_breakdown

                # This is the exact UI call that was failing with `query_terms`.
                with redirect_stdout(io.StringIO()):
                    card = build_candidate_card(
                        resume_id=m.resume_id,
                        rrf_score=r.rrf_score,
                        jd_skills=r.matched_skills,
                        matched_text=r.matched_text or "",
                        section=r.matched_sections[0] if r.matched_sections else "",
                        dense_score=r.dense_score,
                        bm25_score=r.bm25_score,
                        query=query,
                    )

                report.write(f"--- Rank {i} ---\n")
                report.write(f"  resume_id:      {m.resume_id}\n")
                report.write(f"  candidate_name: {m.candidate_name or 'N/A'}\n")
                report.write(f"  display_name:   {card['name'] if card else 'N/A'}\n")
                report.write(f"  skills:         {m.skills[:15] if m.skills else []}\n")
                report.write(f"  matched_skills: {r.matched_skills}\n")
                report.write(f"  skill_match:    {round(s.get('skill', 0.0) * 100, 2)}%\n")
                report.write(f"  overall_match:  {round(s.get('overall', 0.0) * 100, 2)}%\n")
                report.write(f"  applicable:     {s.get('applicable', [])}\n")
                report.write(f"  raw_scores:     {s.get('raw_scores', {})}\n")
                report.write(f"  denominator:    {s.get('denominator', 0.0)}\n")
                report.write(f"  final_score:    {r.final_score:.4f}\n")

        report.write("\n" + "=" * 80 + "\n")
        report.write("All search tests complete.\n")
        report.write("=" * 80 + "\n")

    print(f"Report written to: {report_path}")
    print(f"Exit code: 0")


if __name__ == "__main__":
    main()
