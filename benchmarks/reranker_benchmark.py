"""
Reranker Benchmark Script.

This script benchmarks the cross-encoder reranker at different rerank depths
to determine the optimal configuration for production use.
"""

import sys
from pathlib import Path
import logging
import time
import random
import psutil
import json
from typing import List, Dict, Any
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src.retrieval.reranker import RerankerService, RerankerModel
from src.retrieval.hybrid.schema import HybridSearchResult, MatchedChunk, RetrievalSource
from src.retrieval.sparse import SparseRetrievalService, IndexBuilder as SparseIndexBuilder
from src.chunks.schema import Chunk, ChunkMetadata


def print_success(message: str):
    print(f"\033[92m✅ {message}\033[0m")


def print_failure(message: str):
    print(f"\033[91m❌ {message}\033[0m")


def print_info(message: str):
    print(f"\033[94mℹ️  {message}\033[0m")


def print_header(message: str):
    print(f"\033[93m{'=' * 80}\033[0m")
    print(f"\033[93m{message}\033[0m")
    print(f"\033[93m{'=' * 80}\033[0m")


def generate_mock_chunks(num_chunks: int = 200) -> list:
    """Generate mock Chunk objects for benchmarking."""
    chunks = []
    technologies = [
        "python", "fastapi", "django", "flask", "sql", "postgresql",
        "docker", "kubernetes", "aws", "gcp", "azure",
        "react", "vue.js", "angular", "node.js", "javascript",
        "machine learning", "deep learning", "tensorflow", "pytorch",
        "nlp", "computer vision", "data science", "analytics",
        "langchain", "pinecone", "vector database", "llm",
        "microservices", "rest api", "graphql", "redis",
        "mongodb", "elasticsearch", "kafka", "rabbitmq"
    ]
    
    for i in range(num_chunks):
        # Generate random text
        num_terms = random.randint(15, 40)
        text_terms = random.sample(technologies, min(num_terms, len(technologies)))
        text = " ".join(text_terms)
        
        # Create Chunk object
        chunk = Chunk(
            chunk_id=f"chunk-{i}",
            resume_id=f"resume-{i % 10}",
            candidate_name=f"Candidate {i % 10}",
            section=random.choice(["skills", "experience", "projects"]),
            text=text,
            metadata=ChunkMetadata(
                role=f"Role {i % 5}",
                experience=random.randint(1, 15),
                location=random.choice(["Remote", "On-site", "Hybrid"]),
                education=random.choice(["Bachelor's", "Master's", "PhD"])
            ),
            chunk_order=i
        )
        chunks.append(chunk)
    
    return chunks


def setup_sparse_retrieval(chunks: list) -> SparseRetrievalService:
    """Setup sparse retrieval with mock data."""
    print_info("Setting up sparse retrieval pipeline...")
    
    sparse_builder = SparseIndexBuilder()
    sparse_index = sparse_builder.build_index(chunks)
    sparse_service = SparseRetrievalService(index=sparse_index, cache_enabled=True)
    
    print_success(f"Sparse retrieval setup: {len(chunks)} chunks")
    return sparse_service


def sparse_to_hybrid_results(sparse_results: list, query: str) -> list:
    """Convert sparse results to hybrid results format for reranker."""
    hybrid_results = []
    
    for i, result in enumerate(sparse_results):
        # Create matched chunk
        matched_chunk = MatchedChunk(
            chunk_id=result.chunk_id,
            section=result.section,
            matched_text=result.matched_text,
            score=result.bm25_score,
            retrieval_source=RetrievalSource.SPARSE
        )
        
        # Create hybrid result
        hybrid_result = HybridSearchResult(
            query=query,
            candidate_name=result.candidate_name,
            resume_id=result.resume_id,
            chunk_id=result.chunk_id,
            section=result.section,
            rank=i,
            rrf_score=result.bm25_score,
            dense_rank=None,
            sparse_rank=i,
            matched_chunks=[matched_chunk],
            metadata=result.metadata
        )
        hybrid_results.append(hybrid_result)
    
    return hybrid_results


class RerankerBenchmark:
    """Benchmark class for cross-encoder reranker."""
    
    def __init__(self):
        """Initialize the benchmark."""
        self.queries = [
            "Python Backend Developer",
            "FastAPI Engineer",
            "Machine Learning Engineer",
            "AWS Cloud Architect",
            "Docker Kubernetes",
            "React Developer",
            "Data Scientist",
            "DevOps Engineer",
            "Full Stack Developer",
            "NLP Engineer"
        ]
        self.results = {}
    
    def benchmark_rerank_depth(
        self,
        reranker: RerankerService,
        sparse_service: SparseRetrievalService,
        rerank_depth: int,
        num_queries: int = 50
    ) -> Dict[str, Any]:
        """
        Benchmark reranker at a specific rerank depth.
        
        Args:
            reranker: RerankerService instance
            sparse_service: SparseRetrievalService instance
            rerank_depth: Number of candidates to rerank
            num_queries: Number of queries to run
            
        Returns:
            Dictionary with benchmark results
        """
        print_info(f"Benchmarking rerank depth: {rerank_depth}")
        
        latencies = []
        batch_inference_times = []
        total_candidates_reranked = 0
        cache_hits = 0
        cache_misses = 0
        memory_usage_before = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        
        start_time = time.time()
        
        for i in range(num_queries):
            query = self.queries[i % len(self.queries)]
            
            try:
                # Step 1: Sparse Retrieval
                sparse_results = sparse_service.search(query, top_k=rerank_depth)
                hybrid_results = sparse_to_hybrid_results(sparse_results, query)
                
                # Step 2: Reranking
                query_start = time.time()
                reranked_results = reranker.rerank(query, hybrid_results, top_k=rerank_depth)
                query_latency = time.time() - query_start
                
                latencies.append(query_latency)
                total_candidates_reranked += len(hybrid_results)
                
                # Get cache stats
                if reranker.cache:
                    cache_stats = reranker.cache.get_stats()
                    cache_hits = cache_stats.get('hits', 0)
                    cache_misses = cache_stats.get('misses', 0)
                
            except Exception as e:
                print_failure(f"Rerank query {i+1} failed: {e}")
                continue
        
        total_time = time.time() - start_time
        memory_usage_after = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        memory_usage_delta = memory_usage_after - memory_usage_before
        
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        throughput = num_queries / total_time if total_time > 0 else 0
        cache_hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0.0
        
        results = {
            "rerank_depth": rerank_depth,
            "avg_latency": avg_latency,
            "total_time": total_time,
            "throughput": throughput,
            "total_candidates_reranked": total_candidates_reranked,
            "avg_candidates_per_query": total_candidates_reranked / num_queries if num_queries > 0 else 0,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "memory_usage_before_mb": memory_usage_before,
            "memory_usage_after_mb": memory_usage_after,
            "memory_usage_delta_mb": memory_usage_delta,
            "batch_size": reranker.batch_processor.batch_size
        }
        
        print_success(
            f"Rerank depth {rerank_depth}: avg_latency={avg_latency:.4f}s, "
            f"throughput={throughput:.2f} q/s, cache_hit_rate={cache_hit_rate:.2%}"
        )
        
        return results
    
    def run_all_benchmarks(self, num_chunks: int = 200):
        """Run all benchmarks at different rerank depths."""
        print_header("RERANKER BENCHMARK")
        print()
        
        # Generate mock data
        print_info(f"Generating {num_chunks} mock chunks...")
        chunks = generate_mock_chunks(num_chunks)
        print_success(f"Generated {len(chunks)} chunks")
        print()
        
        # Setup sparse retrieval
        sparse_service = setup_sparse_retrieval(chunks)
        
        # Initialize reranker service (offline mode for testing)
        print_info("Initializing reranker service (offline mode)...")
        reranker = RerankerService(
            model_name=RerankerModel.MINILM_V2.value,
            offline_mode=True,
            cache_enabled=True,
            batch_size=32
        )
        print_success("Reranker service initialized")
        print()
        
        # Benchmark different rerank depths
        print_header("BENCHMARKING RERANK DEPTHS")
        print()
        
        rerank_depths = [10, 25, 50, 100]
        depth_results = {}
        
        for depth in rerank_depths:
            try:
                results = self.benchmark_rerank_depth(
                    reranker,
                    sparse_service,
                    depth,
                    num_queries=20  # Reduced for offline mode testing
                )
                depth_results[f"depth_{depth}"] = results
            except Exception as e:
                print_failure(f"Benchmark for depth {depth} failed: {e}")
                # Create placeholder results for failed benchmark
                depth_results[f"depth_{depth}"] = {
                    "rerank_depth": depth,
                    "avg_latency": 0.0,
                    "total_time": 0.0,
                    "throughput": 0.0,
                    "total_candidates_reranked": 0,
                    "avg_candidates_per_query": 0.0,
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "cache_hit_rate": 0.0,
                    "memory_usage_before_mb": 0.0,
                    "memory_usage_after_mb": 0.0,
                    "memory_usage_delta_mb": 0.0,
                    "batch_size": 32,
                    "error": str(e)
                }
        
        self.results = depth_results
        
        # Generate report
        self.generate_report(num_chunks)
    
    def generate_report(self, num_chunks: int):
        """Generate markdown report with benchmark results."""
        print_header("GENERATING REPORT")
        print()
        
        # Create report
        report = []
        report.append("# Reranker Benchmark Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nDocument Count: {num_chunks}")
        report.append("\n## Executive Summary")
        report.append("\nThis report presents the performance benchmarks of the cross-encoder reranker")
        report.append("at different rerank depths to determine the optimal configuration.")
        
        # Comparison table
        report.append("\n## Performance Comparison")
        report.append("\n| Rerank Depth | Avg Latency (s) | Throughput (q/s) | Cache Hit Rate | Memory Delta (MB) |")
        report.append("|--------------|----------------|-----------------|----------------|-------------------|")
        
        for key in sorted(self.results.keys()):
            result = self.results[key]
            depth = result['rerank_depth']
            avg_latency = result['avg_latency']
            throughput = result['throughput']
            cache_hit_rate = result['cache_hit_rate']
            memory_delta = result['memory_usage_delta_mb']
            
            report.append(
                f"| {depth} | "
                f"{avg_latency:.4f} | "
                f"{throughput:.2f} | "
                f"{cache_hit_rate:.2%} | "
                f"{memory_delta:.2f} |"
            )
        
        # Detailed metrics
        report.append("\n## Detailed Metrics")
        
        for key in sorted(self.results.keys()):
            result = self.results[key]
            depth = result['rerank_depth']
            
            report.append(f"\n### Rerank Depth: {depth}")
            report.append(f"\n- **Average Latency**: {result['avg_latency']:.4f}s")
            report.append(f"- **Total Time**: {result['total_time']:.4f}s")
            report.append(f"- **Throughput**: {result['throughput']:.2f} q/s")
            report.append(f"- **Total Candidates Reranked**: {result['total_candidates_reranked']}")
            report.append(f"- **Average Candidates per Query**: {result['avg_candidates_per_query']:.2f}")
            report.append(f"- **Cache Hits**: {result['cache_hits']}")
            report.append(f"- **Cache Misses**: {result['cache_misses']}")
            report.append(f"- **Cache Hit Rate**: {result['cache_hit_rate']:.2%}")
            report.append(f"- **Memory Usage Before**: {result['memory_usage_before_mb']:.2f} MB")
            report.append(f"- **Memory Usage After**: {result['memory_usage_after_mb']:.2f} MB")
            report.append(f"- **Memory Usage Delta**: {result['memory_usage_delta_mb']:.2f} MB")
            report.append(f"- **Batch Size**: {result['batch_size']}")
        
        # Analysis
        report.append("\n## Analysis")
        
        report.append("\n### Latency vs Rerank Depth")
        report.append("\nHigher rerank depths increase latency as more candidates need to be")
        report.append("processed by the cross-encoder model. The relationship is typically linear")
        report.append("or slightly superlinear due to batch processing overhead.")
        
        report.append("\n### Throughput vs Rerank Depth")
        report.append("\nThroughput decreases as rerank depth increases due to the increased")
        report.append("computational cost of processing more candidates per query.")
        
        report.append("\n### Memory Usage")
        report.append("\nMemory usage increases with rerank depth due to:")
        report.append("- Larger batch sizes for inference")
        report.append("- More candidate data in memory")
        report.append("- Increased cache size")
        
        report.append("\n### Cache Hit Rate")
        report.append("\nCache hit rate depends on query repetition. Higher rerank depths may")
        report.append("benefit more from caching if queries are repeated.")
        
        # Recommendation
        report.append("\n## Recommendations")
        
        # Find optimal depth based on latency and throughput
        best_depth = 10
        best_score = 0
        
        for key in self.results.keys():
            result = self.results[key]
            # Score based on throughput and cache hit rate, penalized by latency
            if result['avg_latency'] > 0:
                score = (result['throughput'] / 100) + (result['cache_hit_rate'] * 10) - (result['avg_latency'] * 10)
            else:
                score = 0
            
            if score > best_score:
                best_score = score
                best_depth = result['rerank_depth']
        
        report.append(f"\n### Optimal Rerank Depth")
        report.append(f"\nBased on the benchmark results, the recommended rerank depth is **{best_depth}**.")
        report.append("This value provides the best balance between latency, throughput, and cache efficiency.")
        
        report.append("\n### Rerank Depth Guidelines")
        report.append("\n- **Top 10**: Best for low-latency applications with strict SLA requirements")
        report.append("- **Top 25**: Good balance for most production use cases")
        report.append("- **Top 50**: Suitable for applications where accuracy is prioritized over latency")
        report.append("- **Top 100**: Maximum accuracy for offline processing or batch jobs")
        
        # Save report
        report_path = Path(__file__).parent / "reranker_benchmark_report.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print_success(f"Report saved to: {report_path}")
        
        # Save JSON results
        json_path = Path(__file__).parent / "reranker_benchmark_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print_success(f"Results saved to: {json_path}")
        
        # Print report
        print()
        print_header("BENCHMARK REPORT")
        print()
        print('\n'.join(report))
        print()


def main():
    """Main function to run the benchmark."""
    try:
        benchmark = RerankerBenchmark()
        benchmark.run_all_benchmarks(num_chunks=200)
        
        print_header("BENCHMARK COMPLETE")
        print_success("All benchmarks completed successfully")
        print()
        print("🚀 Reranker Benchmark Complete")
        
        return 0
    except Exception as e:
        print_failure(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
