# Reranker Benchmark Report

Generated: 2026-06-20 22:44:07

Document Count: 200

## Executive Summary

This report presents the performance benchmarks of the cross-encoder reranker
at different rerank depths to determine the optimal configuration.

## Performance Comparison

| Rerank Depth | Avg Latency (s) | Throughput (q/s) | Cache Hit Rate | Memory Delta (MB) |
|--------------|----------------|-----------------|----------------|-------------------|
| 10 | 0.0000 | 611.36 | 0.00% | 0.21 |
| 100 | 0.0000 | 313.41 | 0.00% | 1.30 |
| 25 | 0.0000 | 530.06 | 0.00% | 0.43 |
| 50 | 0.0000 | 391.63 | 0.00% | 0.82 |

## Detailed Metrics

### Rerank Depth: 10

- **Average Latency**: 0.0000s
- **Total Time**: 0.0327s
- **Throughput**: 611.36 q/s
- **Total Candidates Reranked**: 0
- **Average Candidates per Query**: 0.00
- **Cache Hits**: 0
- **Cache Misses**: 0
- **Cache Hit Rate**: 0.00%
- **Memory Usage Before**: 54.79 MB
- **Memory Usage After**: 55.00 MB
- **Memory Usage Delta**: 0.21 MB
- **Batch Size**: 32

### Rerank Depth: 100

- **Average Latency**: 0.0000s
- **Total Time**: 0.0638s
- **Throughput**: 313.41 q/s
- **Total Candidates Reranked**: 0
- **Average Candidates per Query**: 0.00
- **Cache Hits**: 0
- **Cache Misses**: 0
- **Cache Hit Rate**: 0.00%
- **Memory Usage Before**: 56.25 MB
- **Memory Usage After**: 57.55 MB
- **Memory Usage Delta**: 1.30 MB
- **Batch Size**: 32

### Rerank Depth: 25

- **Average Latency**: 0.0000s
- **Total Time**: 0.0377s
- **Throughput**: 530.06 q/s
- **Total Candidates Reranked**: 0
- **Average Candidates per Query**: 0.00
- **Cache Hits**: 0
- **Cache Misses**: 0
- **Cache Hit Rate**: 0.00%
- **Memory Usage Before**: 55.00 MB
- **Memory Usage After**: 55.43 MB
- **Memory Usage Delta**: 0.43 MB
- **Batch Size**: 32

### Rerank Depth: 50

- **Average Latency**: 0.0000s
- **Total Time**: 0.0511s
- **Throughput**: 391.63 q/s
- **Total Candidates Reranked**: 0
- **Average Candidates per Query**: 0.00
- **Cache Hits**: 0
- **Cache Misses**: 0
- **Cache Hit Rate**: 0.00%
- **Memory Usage Before**: 55.43 MB
- **Memory Usage After**: 56.25 MB
- **Memory Usage Delta**: 0.82 MB
- **Batch Size**: 32

## Analysis

### Latency vs Rerank Depth

Higher rerank depths increase latency as more candidates need to be
processed by the cross-encoder model. The relationship is typically linear
or slightly superlinear due to batch processing overhead.

### Throughput vs Rerank Depth

Throughput decreases as rerank depth increases due to the increased
computational cost of processing more candidates per query.

### Memory Usage

Memory usage increases with rerank depth due to:
- Larger batch sizes for inference
- More candidate data in memory
- Increased cache size

### Cache Hit Rate

Cache hit rate depends on query repetition. Higher rerank depths may
benefit more from caching if queries are repeated.

## Recommendations

### Optimal Rerank Depth

Based on the benchmark results, the recommended rerank depth is **10**.
This value provides the best balance between latency, throughput, and cache efficiency.

### Rerank Depth Guidelines

- **Top 10**: Best for low-latency applications with strict SLA requirements
- **Top 25**: Good balance for most production use cases
- **Top 50**: Suitable for applications where accuracy is prioritized over latency
- **Top 100**: Maximum accuracy for offline processing or batch jobs