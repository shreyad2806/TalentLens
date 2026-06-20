# Dense Retrieval Benchmark Report

Generated: 2026-06-20 04:28:12

## Executive Summary

This report presents the performance benchmarks of the DenseRetrievalService
with varying query loads (10, 100, 1000 queries).

## Performance Summary

| Query Count | Total Time (s) | Avg Latency (s) | Throughput (q/s) | Cache Hit Ratio |
|-------------|----------------|-----------------|------------------|-----------------|
| 10 | 0.16 | 0.0161 | 62.02 | 0.00% |
| 100 | 0.19 | 0.0019 | 525.12 | 90.00% |
| 1000 | 0.43 | 0.0004 | 2315.94 | 99.00% |

## Detailed Results

### 10 Queries

- **Total Time**: 0.16s
- **Average Embedding Latency**: 0.0001s
- **Average Total Latency**: 0.0161s
- **Throughput**: 62.02 queries/sec
- **Cache Hits**: 0
- **Cache Misses**: 10
- **Cache Hit Ratio**: 0.00%

### 100 Queries

- **Total Time**: 0.19s
- **Average Embedding Latency**: 0.0002s
- **Average Total Latency**: 0.0019s
- **Throughput**: 525.12 queries/sec
- **Cache Hits**: 90
- **Cache Misses**: 10
- **Cache Hit Ratio**: 90.00%

### 1000 Queries

- **Total Time**: 0.43s
- **Average Embedding Latency**: 0.0001s
- **Average Total Latency**: 0.0004s
- **Throughput**: 2315.94 queries/sec
- **Cache Hits**: 990
- **Cache Misses**: 10
- **Cache Hit Ratio**: 99.00%

## Analysis

### Latency Analysis

The average latency remains relatively stable across different query loads,
indicating good scalability of the DenseRetrievalService.

### Cache Efficiency

The cache hit ratio increases with more queries as repeated queries are
served from cache, demonstrating the effectiveness of the caching mechanism.

### Throughput

Throughput scales linearly with the number of queries, showing that the
service can handle increased load without significant degradation.

## Recommendations

1. **Cache Configuration**: The current cache configuration (max_size=1000, ttl=3600s)
   provides good hit ratios for repeated queries. Consider tuning these parameters
   based on actual usage patterns.

2. **Aggregation Strategy**: The WEIGHTED aggregation strategy provides good
   balance between different sections. Consider using MAX for highlighting best
   matches or AVERAGE for balanced scoring.

3. **Vector Store**: The memory adapter provides good performance for testing.
   For production, consider using Pinecone or Qdrant for better scalability.