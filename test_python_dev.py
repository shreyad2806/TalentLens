"""Quick check for 'python developer' combined role + skill + retrieval."""
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
    query = "python developer"

    with redirect_stdout(io.StringIO()):
        results = search_service.search(query=query, top_k=5)

    print(f"QUERY: {query!r}\n")
    for i, r in enumerate(results, 1):
        m = r.resume_metadata
        s = r.score_breakdown
        print(f"--- Rank {i} ---")
        print(f"  resume_id:      {m.resume_id}")
        print(f"  display_name:   {getattr(m, 'candidate_name') or 'Resume #' + m.resume_id}")
        print(f"  role:           {m.role}")
        print(f"  skills:         {m.skills[:10] if m.skills else []}")
        print(f"  matched_skills: {r.matched_skills}")
        print(f"  applicable:     {s.get('applicable', [])}")
        print(f"  raw_scores:     {s.get('raw_scores', {})}")
        print(f"  denominator:    {s.get('denominator', 0.0)}")
        print(f"  overall_match:  {round(s.get('overall', 0.0) * 100, 2)}%")
        print(f"  final_score:    {r.final_score:.4f}")


if __name__ == "__main__":
    main()
