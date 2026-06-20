import sys
import os
import time
import random
import uuid
import tracemalloc
from typing import List, Dict, Any, Tuple

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from retrieval.metadata.schema import CandidateMetadata, MetadataFilter, FilterCondition, OrFilterGroup, FilterOperator, FilterLogic
from retrieval.metadata.metadata_service import MetadataService

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_candidates(count: int) -> List[CandidateMetadata]:
    locations = ["Bangalore", "Mumbai", "Delhi", "Pune", "Hyderabad"]
    skills_pool = ["Python", "Java", "React", "AWS", "SQL", "Docker", "Kubernetes", "C++", "Go", "Azure"]
    education_pool = ["B.Tech", "M.Tech", "BCA", "MCA", "B.Sc"]
    
    candidates = []
    for _ in range(count):
        candidates.append(CandidateMetadata(
            candidate_id=str(uuid.uuid4()),
            resume_id=str(uuid.uuid4()),
            experience_years=round(random.uniform(0, 15), 1),
            location=random.choice(locations),
            skills=random.sample(skills_pool, k=random.randint(1, 5)),
            education=[random.choice(education_pool)],
            salary_expectation=round(random.uniform(5, 50), 1),
            notice_period_days=random.choice([0, 15, 30, 60, 90])
        ))
    return candidates

def run_benchmarks():
    sys.stdout.reconfigure(encoding='utf-8')
    print("Starting Metadata Filtering Benchmark...")
    service = MetadataService(cache_enabled=True)
    
    sizes = [100, 1000, 10000, 100000]
    datasets = {}
    memory_usage = {}
    
    tracemalloc.start()
    for size in sizes:
        tracemalloc.clear_traces()
        snapshot1 = tracemalloc.take_snapshot()
        datasets[size] = generate_candidates(size)
        snapshot2 = tracemalloc.take_snapshot()
        stats = snapshot2.compare_to(snapshot1, 'lineno')
        diff = sum(stat.size_diff for stat in stats)
        memory_usage[size] = diff / (1024 * 1024) # MB
    tracemalloc.stop()
    print("Datasets generated.")
    
    # 1. Filter Parsing Time
    queries = [
        "Senior Python Developer in Bangalore with 5+ years experience under 25 LPA",
        "React developer in Mumbai",
        "Data Engineer with AWS and Docker"
    ]
    parse_latencies = []
    for q in queries:
        res = service.parse_filters(q)
        parse_latencies.append(res.parse_latency_ms)
    avg_parse_time = sum(parse_latencies) / len(parse_latencies)
    
    # 6. Filter Type Performance & 2. Filter Execution Time & 3. Candidate Reduction Ratio
    filters_to_test = {
        "Location": MetadataFilter(location="Bangalore"),
        "Experience": MetadataFilter(minimum_experience=5.0),
        "Salary": MetadataFilter(salary_max=25.0),
        "Skills": MetadataFilter(skills=["Python"]),
        "Education": MetadataFilter(education=["B.Tech"]),
        "Notice period": MetadataFilter(notice_period=30),
        "Complex AND": MetadataFilter(location="Bangalore", minimum_experience=5.0, skills=["Python"]),
        "Complex OR": MetadataFilter(or_groups=[
            OrFilterGroup(conditions=[
                FilterCondition(field="location", operator=FilterOperator.EQ, value="Bangalore"),
                FilterCondition(field="location", operator=FilterOperator.EQ, value="Mumbai")
            ])
        ]),
        "Complex NOT": MetadataFilter(not_conditions=[
            FilterCondition(field="location", operator=FilterOperator.EQ, value="Delhi")
        ])
    }
    
    results_data = []
    
    fastest_filter = ("", float('inf'))
    slowest_filter = ("", 0.0)
    best_reduction = ("", 0.0)
    
    # For caching
    cache_miss_time = 0
    cache_hit_time = 0
    
    for size in sizes:
        candidates = datasets[size]
        for f_name, f_obj in filters_to_test.items():
            service.clear_cache()
            
            # Miss
            res_miss = service.filter_candidates(candidates=candidates, filters=f_obj)
            latency_miss = res_miss.filter_latency_ms
            
            # Hit
            res_hit = service.filter_candidates(candidates=candidates, filters=f_obj)
            latency_hit = res_hit.filter_latency_ms
            
            if size == 100000:
                cache_miss_time += latency_miss
                cache_hit_time += latency_hit
                
                if latency_miss < fastest_filter[1]:
                    fastest_filter = (f_name, latency_miss)
                if latency_miss > slowest_filter[1]:
                    slowest_filter = (f_name, latency_miss)
                    
                reduction = ((res_miss.total_before - res_miss.total_after) / res_miss.total_before) * 100
                if reduction > best_reduction[1]:
                    best_reduction = (f_name, reduction)
            
            reduction = ((res_miss.total_before - res_miss.total_after) / res_miss.total_before) * 100
            
            results_data.append({
                "Dataset Size": size,
                "Filter Type": f_name,
                "Latency (ms)": f"{latency_miss:.2f}",
                "Reduction %": f"{reduction:.2f}%",
            })
            
    cache_speedup = ((cache_miss_time - cache_hit_time) / cache_miss_time) * 100 if cache_miss_time > 0 else 0
    
    # Generate Markdown Report
    report_content = f"""# Metadata Filtering Benchmark Report

## 1. Filter Parsing Time
- **Average Parse Latency:** {avg_parse_time:.2f} ms

## 2 & 6. Filter Execution Time & Performance
| Dataset Size | Filter Type | Latency (ms) | Reduction % |
|-------------|-------------|--------------|-------------|
"""
    for r in results_data:
        report_content += f"| {r['Dataset Size']} | {r['Filter Type']} | {r['Latency (ms)']} | {r['Reduction %']} |\n"
        
    report_content += f"""
## 4. Cache Performance (at 100,000 candidates)
- **Average Miss Latency:** {cache_miss_time / len(filters_to_test):.2f} ms
- **Average Hit Latency:** {cache_hit_time / len(filters_to_test):.2f} ms
- **Cache Speedup:** {cache_speedup:.2f}%

## 5. Memory Usage
| Dataset Size | RAM Consumption (MB) |
|-------------|----------------------|
"""
    for size, mem in memory_usage.items():
        report_content += f"| {size} | {mem:.2f} |\n"
        
    report_content += """
## Recommendations
- **Optimal Filter Ordering:** Apply the most restrictive filters first (e.g., specific skills, strict notice period) before evaluating complex OR conditions.
- **Cache TTL:** A TTL of 3600 seconds is sufficient for recruiter workflows since candidates' core attributes don't change frequently during active search sessions.
- **Indexing Strategy:** For >100,000 candidates, consider adding an inverted index or database-level filtering before Python-level filtering.
- **Metadata Optimization:** Store fields like `location` as categorical integer codes to reduce memory consumption and speed up equality checks.
- **Large-scale Recruiter Search:** Paginating results and aggressively caching common queries (like "Python Developer Bangalore") will maximize hit rates and keep P99 latency low.
"""

    ensure_dir(os.path.join(os.path.dirname(__file__), 'reports'))
    report_path = os.path.join(os.path.dirname(__file__), 'reports', 'metadata_filter_benchmark.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    print(f"Report generated at: {report_path}")
    print("\n✅ Metadata Benchmark Complete")
    print(f"Fastest Filter: {fastest_filter[0]} ({fastest_filter[1]:.2f} ms)")
    print(f"Slowest Filter: {slowest_filter[0]} ({slowest_filter[1]:.2f} ms)")
    print(f"Best Candidate Reduction: {best_reduction[0]} ({best_reduction[1]:.2f}%)")
    print(f"Cache Speedup: {cache_speedup:.2f}%")
    print("Recommended Production Configuration: Cache Enabled, Max Size: 10000, TTL: 3600s, Pre-filtering before vector search.")

if __name__ == "__main__":
    run_benchmarks()
