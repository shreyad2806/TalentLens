"""
Metadata Filtering Engine Test.

Loads sample candidates, runs recruiter queries, validates all filter types
(location, experience, salary, skills, education, language, notice period,
AND/OR/NOT logic, range filtering, cache), and reports latency and metrics.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from src.retrieval.metadata import (
    CandidateMetadata,
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataService,
    OrFilterGroup,
    ValidationError,
)


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def print_success(message: str) -> None:
    print(f"\033[92m✅ {message}\033[0m")


def print_failure(message: str) -> None:
    print(f"\033[91m❌ {message}\033[0m")


def print_info(message: str) -> None:
    print(f"\033[94mℹ️  {message}\033[0m")


def print_header(message: str) -> None:
    print(f"\033[93m{'=' * 80}\033[0m")
    print(f"\033[93m{message}\033[0m")
    print(f"\033[93m{'=' * 80}\033[0m")


def print_filters(filters: MetadataFilter) -> None:
    payload = filters.model_dump(exclude_none=True)
    print(json.dumps(payload, indent=2, default=str))


def print_metrics(label: str, accuracy: float, precision: float, recall: float) -> None:
    print(f"\n  {label}")
    print(f"    Filter Accuracy : {accuracy:.2%}")
    print(f"    Filter Precision: {precision:.2%}")
    print(f"    Filter Recall   : {recall:.2%}")


def compute_filter_metrics(
    all_ids: Set[str],
    predicted: Set[str],
    relevant: Set[str],
) -> Tuple[float, float, float]:
    """
    Compute accuracy, precision, recall for a filtering operation.

    TP = correctly retained relevant candidates
    FP = incorrectly retained non-relevant candidates
    FN = incorrectly excluded relevant candidates
    TN = correctly excluded non-relevant candidates
    """
    tp = len(predicted & relevant)
    fp = len(predicted - relevant)
    fn = len(relevant - predicted)
    tn = len(all_ids - predicted - relevant)

    total = len(all_ids)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if not relevant else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return accuracy, precision, recall


# ---------------------------------------------------------------------------
# Sample candidate pool
# ---------------------------------------------------------------------------

def load_sample_candidates() -> List[CandidateMetadata]:
    """Load a diverse sample candidate pool for metadata filtering tests."""
    return [
        CandidateMetadata(
            candidate_id="c001",
            resume_id="r001",
            candidate_name="Alice Sharma",
            experience_years=6.0,
            location="Bangalore",
            skills=["Python", "Django", "AWS", "PostgreSQL"],
            education=["B.Tech Computer Science, IIT Delhi"],
            degree="B.TECH",
            salary_expectation=22.0,
            notice_period_days=30,
            work_mode="hybrid",
            languages=["English", "Hindi"],
        ),
        CandidateMetadata(
            candidate_id="c002",
            resume_id="r002",
            candidate_name="Bob Patel",
            experience_years=3.0,
            location="Mumbai",
            skills=["Java", "Spring", "Kafka"],
            education=["B.E. Information Technology, Mumbai University"],
            degree="B.E.",
            salary_expectation=18.0,
            notice_period_days=60,
            work_mode="onsite",
            languages=["English", "Marathi"],
        ),
        CandidateMetadata(
            candidate_id="c003",
            resume_id="r003",
            candidate_name="Carol Reddy",
            experience_years=8.0,
            location="Bangalore",
            skills=["Python", "FastAPI", "Kubernetes", "Docker"],
            education=["M.Tech Software Engineering, IIIT Hyderabad"],
            degree="M.TECH",
            salary_expectation=30.0,
            notice_period_days=15,
            work_mode="remote",
            languages=["English", "Telugu"],
        ),
        CandidateMetadata(
            candidate_id="c004",
            resume_id="r004",
            candidate_name="Dave Kumar",
            experience_years=5.0,
            location="Bangalore",
            skills=["Python", "React", "AWS", "GraphQL"],
            education=["B.Tech Electronics, NIT Trichy"],
            degree="B.TECH",
            salary_expectation=24.0,
            notice_period_days=0,
            work_mode="hybrid",
            languages=["English"],
            availability="immediate",
        ),
        CandidateMetadata(
            candidate_id="c005",
            resume_id="r005",
            candidate_name="Eve Desai",
            experience_years=4.0,
            location="Pune",
            skills=["Python", "FastAPI", "MongoDB"],
            education=["B.Tech Computer Science, Pune University"],
            degree="B.TECH",
            salary_expectation=19.0,
            notice_period_days=45,
            work_mode="remote",
            languages=["English", "Marathi"],
        ),
        CandidateMetadata(
            candidate_id="c006",
            resume_id="r006",
            candidate_name="Frank Iyer",
            experience_years=7.0,
            location="Delhi",
            skills=["React", "Node.js", "AWS", "TypeScript"],
            education=["MCA, Delhi University"],
            degree="MCA",
            salary_expectation=28.0,
            notice_period_days=30,
            work_mode="hybrid",
            languages=["English", "Hindi"],
        ),
        CandidateMetadata(
            candidate_id="c007",
            resume_id="r007",
            candidate_name="Grace Nair",
            experience_years=6.0,
            location="Bangalore",
            skills=["Python", "Machine Learning", "PyTorch", "NLP"],
            education=["B.Tech Computer Science, BITS Pilani"],
            degree="B.TECH",
            salary_expectation=26.0,
            notice_period_days=20,
            work_mode="remote",
            languages=["English", "Malayalam"],
        ),
        CandidateMetadata(
            candidate_id="c008",
            resume_id="r008",
            candidate_name="Henry Thomas",
            experience_years=5.0,
            location="Chennai",
            skills=["Java", "Spring Boot", "Microservices"],
            education=["B.E. Computer Science, Anna University"],
            degree="B.E.",
            salary_expectation=21.0,
            notice_period_days=30,
            work_mode="onsite",
            languages=["English", "Tamil"],
        ),
        CandidateMetadata(
            candidate_id="c009",
            resume_id="r009",
            candidate_name="Iris Khan",
            experience_years=3.0,
            location="Hyderabad",
            skills=["Python", "FastAPI", "Redis"],
            education=["B.Tech Information Technology, JNTU"],
            degree="B.TECH",
            salary_expectation=16.0,
            notice_period_days=90,
            work_mode="remote",
            languages=["English", "Telugu", "Urdu"],
        ),
        CandidateMetadata(
            candidate_id="c010",
            resume_id="r010",
            candidate_name="Jack Mehta",
            experience_years=9.0,
            location="Bangalore",
            skills=["Python", "TensorFlow", "Deep Learning", "NLP"],
            education=["Ph.D. Artificial Intelligence, IISc Bangalore"],
            degree="PH.D",
            salary_expectation=35.0,
            notice_period_days=60,
            work_mode="remote",
            languages=["English", "Gujarati"],
        ),
    ]


# Ground-truth relevance sets for recruiter queries
QUERY_GROUND_TRUTH: Dict[str, Set[str]] = {
    # Python + Bangalore (AND across fields)
    "Python Developer in Bangalore": {"c001", "c003", "c004", "c007", "c010"},
    # Experience >= 5 (no skill extracted for "Backend")
    "Backend Engineer with 5+ years": {
        "c001", "c003", "c004", "c006", "c007", "c008", "c010",
    },
    # FastAPI skill intersection + salary <= 20 LPA
    "FastAPI Developer under 20 LPA": {"c005", "c009"},
    # work_mode=remote (AI not in known-skills lexicon)
    "Remote AI Engineer": {"c003", "c005", "c007", "c009", "c010"},
    # skills use list intersection (>=1 match): React OR AWS
    "React Developer with AWS": {"c001", "c004", "c006"},
    # Machine Learning + B.Tech degree
    "Machine Learning Engineer with B.Tech": {"c007"},
}


RECRUITER_QUERIES = list(QUERY_GROUND_TRUTH.keys())


# ---------------------------------------------------------------------------
# Validation scenarios
# ---------------------------------------------------------------------------

def validate_location_filter(service: MetadataService, candidates: List[CandidateMetadata]) -> bool:
    print_info("Validating location filter...")
    filters = MetadataFilter(location="Bangalore")
    result = service.apply_filters(filters, candidates)
    expected = {"c001", "c003", "c004", "c007", "c010"}
    passed = set(result.candidate_ids) == expected
    print(f"  Expected: {sorted(expected)}")
    print(f"  Got     : {sorted(result.candidate_ids)}")
    return passed


def validate_experience_filter(service: MetadataService, candidates: List[CandidateMetadata]) -> bool:
    print_info("Validating experience filter...")
    filters = MetadataFilter(minimum_experience=5.0)
    result = service.apply_filters(filters, candidates)
    expected = {"c001", "c003", "c004", "c006", "c007", "c008", "c010"}
    passed = set(result.candidate_ids) == expected
    print(f"  Expected: {sorted(expected)}")
    print(f"  Got     : {sorted(result.candidate_ids)}")
    return passed


def validate_salary_filter(service: MetadataService, candidates: List[CandidateMetadata]) -> bool:
    print_info("Validating salary filter...")
    filters = MetadataFilter(salary_max=20.0)
    result = service.apply_filters(filters, candidates)
    expected = {"c002", "c005", "c009"}
    passed = set(result.candidate_ids) == expected
    print(f"  Expected: {sorted(expected)}")
    print(f"  Got     : {sorted(result.candidate_ids)}")
    return passed


def validate_skills_filter(service: MetadataService, candidates: List[CandidateMetadata]) -> bool:
    print_info("Validating skills filter...")
    filters = MetadataFilter(skills=["FastAPI"])
    result = service.apply_filters(filters, candidates)
    expected = {"c003", "c005", "c009"}
    passed = set(result.candidate_ids) == expected
    print(f"  Expected: {sorted(expected)}")
    print(f"  Got     : {sorted(result.candidate_ids)}")
    return passed


def validate_education_filter(service: MetadataService, candidates: List[CandidateMetadata]) -> bool:
    print_info("Validating education filter...")
    filters = MetadataFilter(degree="B.TECH")
    result = service.apply_filters(filters, candidates)
    expected = {"c001", "c004", "c005", "c007", "c009"}
    passed = set(result.candidate_ids) == expected
    print(f"  Expected: {sorted(expected)}")
    print(f"  Got     : {sorted(result.candidate_ids)}")
    return passed


def validate_language_filter(service: MetadataService, candidates: List[CandidateMetadata]) -> bool:
    print_info("Validating language filter...")
    filters = MetadataFilter(languages=["Tamil"])
    result = service.apply_filters(filters, candidates)
    expected = {"c008"}
    passed = set(result.candidate_ids) == expected
    print(f"  Expected: {sorted(expected)}")
    print(f"  Got     : {sorted(result.candidate_ids)}")
    return passed


def validate_notice_period_filter(service: MetadataService, candidates: List[CandidateMetadata]) -> bool:
    print_info("Validating notice period filter (notice_period <= 30 days)...")
    filters = MetadataFilter(notice_period=30)
    result = service.apply_filters(filters, candidates)
    expected = {"c001", "c003", "c004", "c006", "c007", "c008"}
    passed = set(result.candidate_ids) == expected
    print(f"  Expected: {sorted(expected)}")
    print(f"  Got     : {sorted(result.candidate_ids)}")
    return passed


def validate_and_logic(service: MetadataService, candidates: List[CandidateMetadata]) -> bool:
    print_info("Validating AND logic (location + skills + experience)...")
    filters = MetadataFilter(
        location="Bangalore",
        skills=["Python"],
        minimum_experience=5.0,
    )
    result = service.apply_filters(filters, candidates)
    expected = {"c001", "c003", "c004", "c007", "c010"}
    passed = set(result.candidate_ids) == expected
    print(f"  Expected: {sorted(expected)}")
    print(f"  Got     : {sorted(result.candidate_ids)}")
    return passed


def validate_or_logic(service: MetadataService, candidates: List[CandidateMetadata]) -> bool:
    print_info("Validating OR logic (Mumbai OR remote)...")
    filters = MetadataFilter(
        or_groups=[
            OrFilterGroup(
                conditions=[
                    FilterCondition(field="location", operator=FilterOperator.EQ, value="Mumbai"),
                    FilterCondition(field="work_mode", operator=FilterOperator.EQ, value="remote"),
                ]
            )
        ]
    )
    result = service.apply_filters(filters, candidates)
    expected = {"c002", "c003", "c005", "c007", "c009", "c010"}
    passed = set(result.candidate_ids) == expected
    print(f"  Expected: {sorted(expected)}")
    print(f"  Got     : {sorted(result.candidate_ids)}")
    return passed


def validate_not_logic(service: MetadataService, candidates: List[CandidateMetadata]) -> bool:
    print_info("Validating NOT logic (Python AND NOT Java)...")
    filters = MetadataFilter(skills=["Python"], excluded_skills=["Java"])
    result = service.apply_filters(filters, candidates)
    expected = {"c001", "c003", "c004", "c005", "c007", "c009", "c010"}
    passed = set(result.candidate_ids) == expected
    print(f"  Expected: {sorted(expected)}")
    print(f"  Got     : {sorted(result.candidate_ids)}")
    return passed


def validate_range_filtering(service: MetadataService, candidates: List[CandidateMetadata]) -> bool:
    print_info("Validating range filtering (experience 4–7 years)...")
    filters = MetadataFilter(minimum_experience=4.0, maximum_experience=7.0)
    result = service.apply_filters(filters, candidates)
    expected = {"c001", "c004", "c005", "c006", "c007", "c008"}
    passed = set(result.candidate_ids) == expected
    print(f"  Expected: {sorted(expected)}")
    print(f"  Got     : {sorted(result.candidate_ids)}")
    return passed


def validate_cache(service: MetadataService, candidates: List[CandidateMetadata]) -> bool:
    print_info("Validating cache (parse + filter result)...")
    service.clear_cache()

    query = "Cache validation query Python Bangalore 2026"
    parse_1 = service.parse_filters(query)
    parse_2 = service.parse_filters(query)
    parse_hit = parse_1.cache_hit is False and parse_2.cache_hit is True

    filters = MetadataFilter(skills=["TensorFlow"])
    apply_1 = service.apply_filters(filters, candidates)
    apply_2 = service.apply_filters(filters, candidates)
    result_hit = apply_1.cache_hit is False and apply_2.cache_hit is True

    stats = service.get_cache_stats()
    print(f"  Parse cache hit on 2nd call : {parse_2.cache_hit}")
    print(f"  Result cache hit on 2nd call: {apply_2.cache_hit}")
    print(f"  Parse cache stats           : {stats['parse_cache']}")
    print(f"  Result cache stats          : {stats['result_cache']}")
    return parse_hit and result_hit


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------

def run_query_tests(
    service: MetadataService,
    candidates: List[CandidateMetadata],
) -> Tuple[bool, float, float, float]:
    """Run all recruiter queries and aggregate metrics."""
    all_ids = {c.candidate_id for c in candidates}
    total_accuracy = 0.0
    total_precision = 0.0
    total_recall = 0.0
    all_passed = True

    print_header("RECRUITER QUERY TESTS")

    for query in RECRUITER_QUERIES:
        print()
        print_info(f"Query: {query}")
        print()

        start = time.perf_counter()
        parse_result = service.parse_filters(query)
        filter_result = service.filter_candidates(
            candidates=candidates,
            filters=parse_result.filters,
        )
        total_latency_ms = (time.perf_counter() - start) * 1000

        print("Parsed Filters:")
        print_filters(parse_result.filters)
        print()
        print(f"Candidates Before Filtering : {filter_result.total_before}")
        print(f"Candidates After Filtering  : {filter_result.total_after}")
        print(f"Remaining Candidate IDs     : {filter_result.candidate_ids}")
        print()
        print(f"Parse Latency   : {parse_result.parse_latency_ms:.2f} ms")
        print(f"Filter Latency  : {filter_result.filter_latency_ms:.2f} ms")
        print(f"Total Latency   : {total_latency_ms:.2f} ms")
        print(f"Cache Hit       : {filter_result.cache_hit}")

        relevant = QUERY_GROUND_TRUTH[query]
        predicted = set(filter_result.candidate_ids)
        accuracy, precision, recall = compute_filter_metrics(all_ids, predicted, relevant)
        print_metrics("Filter Metrics", accuracy, precision, recall)

        if predicted != relevant:
            print_failure(
                f"Ground-truth mismatch for '{query}' — "
                f"expected {sorted(relevant)}, got {sorted(predicted)}"
            )
            all_passed = False
        else:
            print_success(f"Query '{query}' matched ground truth")

        total_accuracy += accuracy
        total_precision += precision
        total_recall += recall

    n = len(RECRUITER_QUERIES)
    return all_passed, total_accuracy / n, total_precision / n, total_recall / n


def test_metadata_filtering() -> bool:
    """Run the complete metadata filtering test suite."""
    print_header("METADATA FILTERING ENGINE TEST")
    print()

    print_info("Loading sample candidates...")
    candidates = load_sample_candidates()
    print_success(f"Loaded {len(candidates)} sample candidates")
    print()

    for c in candidates:
        print(
            f"  {c.candidate_id} | {c.candidate_name:16} | {c.location:10} | "
            f"{c.experience_years} yrs | {c.salary_expectation} LPA | "
            f"{', '.join(c.skills[:3])}"
        )
    print()

    service = MetadataService(cache_enabled=True, cache_max_size=500, cache_ttl=3600)

    # --- Recruiter queries ---
    queries_passed, avg_accuracy, avg_precision, avg_recall = run_query_tests(
        service, candidates
    )

    # --- Individual filter validations ---
    print()
    print_header("FILTER VALIDATION TESTS")

    validations = [
        ("Location Filter", validate_location_filter),
        ("Experience Filter", validate_experience_filter),
        ("Salary Filter", validate_salary_filter),
        ("Skills Filter", validate_skills_filter),
        ("Education Filter", validate_education_filter),
        ("Language Filter", validate_language_filter),
        ("Notice Period Filter", validate_notice_period_filter),
        ("AND Logic", validate_and_logic),
        ("OR Logic", validate_or_logic),
        ("NOT Logic", validate_not_logic),
        ("Range Filtering", validate_range_filtering),
        ("Cache", validate_cache),
    ]

    validation_results: Dict[str, bool] = {}
    for name, fn in validations:
        print()
        try:
            validation_results[name] = fn(service, candidates)
            if validation_results[name]:
                print_success(f"{name} passed")
            else:
                print_failure(f"{name} failed")
        except Exception as exc:
            validation_results[name] = False
            print_failure(f"{name} raised: {exc}")

    all_validations_passed = all(validation_results.values())

    # --- Aggregate metrics ---
    print()
    print_header("AGGREGATE METRICS")
    print_metrics("Overall Query Performance", avg_accuracy, avg_precision, avg_recall)

    # --- Cache stats ---
    stats = service.get_cache_stats()
    print()
    print_info("Final Cache Statistics:")
    print(f"  Parse cache  — hits: {stats['parse_cache']['hits']}, "
          f"misses: {stats['parse_cache']['misses']}, "
          f"hit_rate: {stats['parse_cache']['hit_rate']:.2%}")
    print(f"  Result cache — hits: {stats['result_cache']['hits']}, "
          f"misses: {stats['result_cache']['misses']}, "
          f"hit_rate: {stats['result_cache']['hit_rate']:.2%}")

    # --- Final verdict ---
    print()
    all_passed = queries_passed and all_validations_passed

    if all_passed:
        print_success("Metadata Filtering Test Passed")
    else:
        print_failure("Metadata Filtering Test Failed")
        failed = [k for k, v in validation_results.items() if not v]
        if failed:
            print_failure(f"Failed validations: {', '.join(failed)}")
        if not queries_passed:
            print_failure("One or more recruiter queries did not match ground truth")

    print()
    print("\033[95m🚀 Metadata Filtering Ready\033[0m")
    print()

    return all_passed


if __name__ == "__main__":
    success = test_metadata_filtering()
    sys.exit(0 if success else 1)
