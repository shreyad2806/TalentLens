# Metadata Filtering Benchmark Report

## 1. Filter Parsing Time
- **Average Parse Latency:** 4.12 ms

## 2 & 6. Filter Execution Time & Performance
| Dataset Size | Filter Type | Latency (ms) | Reduction % |
|-------------|-------------|--------------|-------------|
| 100 | Location | 0.36 | 84.00% |
| 100 | Experience | 0.11 | 38.00% |
| 100 | Salary | 0.13 | 59.00% |
| 100 | Skills | 0.35 | 69.00% |
| 100 | Education | 0.31 | 78.00% |
| 100 | Notice period | 0.11 | 46.00% |
| 100 | Complex AND | 0.19 | 98.00% |
| 100 | Complex OR | 0.40 | 65.00% |
| 100 | Complex NOT | 0.17 | 17.00% |
| 1000 | Location | 1.06 | 80.20% |
| 1000 | Experience | 0.46 | 31.70% |
| 1000 | Salary | 0.44 | 55.00% |
| 1000 | Skills | 1.76 | 68.40% |
| 1000 | Education | 1.92 | 81.20% |
| 1000 | Notice period | 0.60 | 42.80% |
| 1000 | Complex AND | 0.94 | 95.80% |
| 1000 | Complex OR | 6.33 | 60.40% |
| 1000 | Complex NOT | 2.64 | 20.80% |
| 10000 | Location | 11.64 | 79.98% |
| 10000 | Experience | 10.77 | 33.06% |
| 10000 | Salary | 9.08 | 55.69% |
| 10000 | Skills | 29.29 | 69.90% |
| 10000 | Education | 22.36 | 80.70% |
| 10000 | Notice period | 10.20 | 38.83% |
| 10000 | Complex AND | 15.25 | 95.84% |
| 10000 | Complex OR | 53.15 | 59.53% |
| 10000 | Complex NOT | 31.52 | 19.85% |
| 100000 | Location | 143.65 | 80.02% |
| 100000 | Experience | 150.84 | 33.20% |
| 100000 | Salary | 137.85 | 55.58% |
| 100000 | Skills | 297.24 | 69.92% |
| 100000 | Education | 305.35 | 80.18% |
| 100000 | Notice period | 149.37 | 40.01% |
| 100000 | Complex AND | 208.28 | 96.00% |
| 100000 | Complex OR | 535.52 | 59.96% |
| 100000 | Complex NOT | 323.48 | 20.00% |

## 4. Cache Performance (at 100,000 candidates)
- **Average Miss Latency:** 250.17 ms
- **Average Hit Latency:** 131.00 ms
- **Cache Speedup:** 47.64%

## 5. Memory Usage
| Dataset Size | RAM Consumption (MB) |
|-------------|----------------------|
| 100 | 0.18 |
| 1000 | 1.84 |
| 10000 | 18.41 |
| 100000 | 184.06 |

## Recommendations
- **Optimal Filter Ordering:** Apply the most restrictive filters first (e.g., specific skills, strict notice period) before evaluating complex OR conditions.
- **Cache TTL:** A TTL of 3600 seconds is sufficient for recruiter workflows since candidates' core attributes don't change frequently during active search sessions.
- **Indexing Strategy:** For >100,000 candidates, consider adding an inverted index or database-level filtering before Python-level filtering.
- **Metadata Optimization:** Store fields like `location` as categorical integer codes to reduce memory consumption and speed up equality checks.
- **Large-scale Recruiter Search:** Paginating results and aggressively caching common queries (like "Python Developer Bangalore") will maximize hit rates and keep P99 latency low.
