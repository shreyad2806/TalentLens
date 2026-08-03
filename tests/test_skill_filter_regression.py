"""Regression tests for skill normalization in dense Qdrant retrieval."""
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ResumeMetadata
from src.resume_parser.normalizer import MetadataNormalizer
from src.retrieval.hybrid.fusion_service import FusionService
from src.retrieval.metadata.filter_engine import FilterEngine
from src.retrieval.metadata.schema import FilterCondition, FilterOperator
from src.vector_store.adapters.qdrant_adapter import QdrantAdapter
from src.vector_store.schema import VectorRecord


def normalize_test():
    """Verify the shared normalizer collapses Java variants to 'java'."""
    cases = [
        (["Java"], ["java"]),
        (["JAVA"], ["java"]),
        (["java"], ["java"]),
        (["Core Java"], ["java"]),
        (["Java SE"], ["java"]),
        (["J2EE"], ["java"]),
        (["Java/J2EE"], ["java"]),
        (["Python"], ["python"]),
        (["AWS"], ["aws"]),
        (["SQL"], ["sql"]),
        (["React"], ["react"]),
        (["machine learning"], ["ml"]),
    ]
    for raw, expected in cases:
        got = MetadataNormalizer.normalize_skills_for_qdrant(raw)
        assert got == expected, f"{raw!r}: expected {expected!r}, got {got!r}"
    print("[PASS] normalize_skills_for_qdrant")


def dense_retrieval_test():
    """Verify Qdrant dense retrieval returns non-zero results for skill filters."""
    test_dir = Path(tempfile.gettempdir()) / f"qdrant_skill_regression_{uuid.uuid4().hex}"
    os.environ["QDRANT_PATH"] = str(test_dir)
    os.environ["QDRANT_COLLECTION"] = "skill_regression"

    try:
        adapter = QdrantAdapter(config=None)

        dimension = 384
        queries = ["python", "java", "aws", "sql", "react", "ml"]

        records = []
        for i, skill in enumerate(queries):
            # Use the unnormalized raw forms on the index side to prove
            # the adapter normalizes them before upsert.
            raw_skill = skill.upper() if skill in ("aws", "sql") else skill.title()
            if skill == "java":
                raw_skill = "Core Java" if i % 2 == 0 else "J2EE"
            elif skill == "ml":
                raw_skill = "machine learning"

            metadata = ResumeMetadata(
                resume_id=f"resume-{skill}",
                candidate_name=f"Candidate {skill}",
                role="Engineer",
                skills=[raw_skill],
            )
            v = np.random.randn(dimension)
            v = v / np.linalg.norm(v)
            vec = v.tolist()
            records.append(
                VectorRecord(
                    id=str(uuid.uuid4()),
                    chunk_id=f"chunk-{skill}",
                    section="skills",
                    text=f"Experience with {raw_skill}",
                    chunk_text=f"Experience with {raw_skill}",
                    original_text=f"Experience with {raw_skill}",
                    vector=vec,
                    resume_metadata=metadata,
                )
            )

        result = adapter.upsert(records)
        print(f"[INFO] Upserted {result['upserted_count']} records")

        q = np.random.randn(dimension)
        q = q / np.linalg.norm(q)
        query_vec = q.tolist()

        print("\n=== Dense retrieval skill-filter results (after fix) ===")
        for skill in queries:
            # Send the canonical/variant query forms that users might type.
            filter_input = [skill]
            results = adapter.query(query_vec, k=5, filters={"skills": filter_input})
            print(f"  skills={filter_input!r}: returned {len(results)} results")
            assert len(results) >= 1, f"skills={filter_input!r} returned zero results"
            for r in results:
                print(f"    resume_id={r['metadata']['resume_id']} score={r['score']:.4f}")

        print("\n[PASS] dense_retrieval_test")

    finally:
        try:
            adapter._adapter.client.close()
        except Exception:
            pass
        del os.environ["QDRANT_PATH"]
        del os.environ["QDRANT_COLLECTION"]
        if test_dir.exists():
            shutil.rmtree(test_dir)


def sparse_filter_test():
    """Verify FilterEngine normalizes candidate and query skills before matching."""
    engine = FilterEngine()
    condition = FilterCondition(
        field="skills",
        operator=FilterOperator.INTERSECTS,
        value=["java"],
    )

    # Candidate resumes use unnormalized, mixed-casing variants
    candidates = [
        ResumeMetadata(resume_id="r1", skills=["Java"], candidate_name="A"),
        ResumeMetadata(resume_id="r2", skills=["Core Java"], candidate_name="B"),
        ResumeMetadata(resume_id="r3", skills=["Java SE"], candidate_name="C"),
        ResumeMetadata(resume_id="r4", skills=["J2EE"], candidate_name="D"),
        ResumeMetadata(resume_id="r5", skills=["Java/J2EE"], candidate_name="E"),
        ResumeMetadata(resume_id="r6", skills=["Python"], candidate_name="F"),
    ]

    print("\n=== Sparse FilterEngine results (after fix) ===")
    passed = []
    for cand in candidates:
        ok = engine._eval_list_field(cand, condition)
        print(f"  resume_id={cand.resume_id} skills={cand.skills!r}: {ok}")
        if ok:
            passed.append(cand.resume_id)

    assert "r1" in passed
    assert "r2" in passed
    assert "r3" in passed
    assert "r4" in passed
    assert "r5" in passed
    assert "r6" not in passed
    print("[PASS] sparse_filter_test")


def hybrid_fusion_test():
    """Verify FusionService produces a candidate when dense and sparse agree."""
    fuser = FusionService()

    dense_results = [
        {
            "chunk_id": "chunk-java",
            "candidate_name": "Candidate java",
            "resume_id": "resume-java",
            "section": "skills",
            "score": 0.8,
            "metadata": {"resume_id": "resume-java", "skills": ["java"]},
        }
    ]

    sparse_results = [
        {
            "chunk_id": "chunk-java",
            "candidate_name": "Candidate java",
            "resume_id": "resume-java",
            "section": "skills",
            "score": 0.7,
            "metadata": {"resume_id": "resume-java", "skills": ["java"]},
            "matched_chunks": [{"text": "Core Java", "offset": 0}],
        }
    ]

    fused, metrics = fuser.fuse_results(dense_results, sparse_results, "Java developer")

    print("\n=== Hybrid fusion results (after fix) ===")
    print(f"  dense={len(dense_results)} sparse={len(sparse_results)} fused={len(fused)}")
    for r in fused:
        print(f"  resume_id={r.resume_id}")

    assert len(fused) >= 1, "Hybrid fusion returned zero results"
    assert any(r.resume_id == "resume-java" for r in fused)
    print("[PASS] hybrid_fusion_test")


if __name__ == "__main__":
    normalize_test()
    sparse_filter_test()
    dense_retrieval_test()
    hybrid_fusion_test()
    print("\nAll skill-filter regression tests passed.")
