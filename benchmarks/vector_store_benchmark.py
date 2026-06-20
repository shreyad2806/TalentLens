"""
Vector Store Adapter Benchmarking Script.

This script benchmarks the performance of different vector store adapters:
- Memory (in-memory)
- Pinecone (if configured)
- Qdrant (if configured)

Metrics measured:
- Connect time
- Upsert 100 vectors
- Upsert 1000 vectors
- Query latency
- Delete latency
- Memory usage
- Throughput

Results are saved to a markdown report with recommendations.
"""

import sys
from pathlib import Path
import logging
import os
import time
import psutil
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src.vector_store import VectorRecord, VectorStoreService
from src.vector_store.config import reset_config


def print_success(message: str):
    """Print a success message in green."""
    print(f"\033[92m✅ {message}\033[0m")


def print_failure(message: str):
    """Print a failure message in red."""
    print(f"\033[91m❌ {message}\033[0m")


def print_info(message: str):
    """Print an info message in blue."""
    print(f"\033[94mℹ️  {message}\033[0m")


def print_warning(message: str):
    """Print a warning message in yellow."""
    print(f"\033[93m⚠ {message}\033[0m")


def print_header(message: str):
    """Print a header message in yellow."""
    print(f"\033[93m{'=' * 80}\033[0m")
    print(f"\033[93m{message}\033[0m")
    print(f"\033[93m{'=' * 80}\033[0m")


def get_memory_usage() -> float:
    """
    Get current memory usage in MB.
    
    Returns:
        Memory usage in MB
    """
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def generate_test_vectors(count: int, dimension: int = 1024) -> List[VectorRecord]:
    """
    Generate test vector records.
    
    Args:
        count: Number of vectors to generate
        dimension: Vector dimension (default: 1024)
        
    Returns:
        List of VectorRecord objects
    """
    import random
    
    vectors = []
    for i in range(count):
        vector = [random.random() for _ in range(dimension)]
        record = VectorRecord(
            id=f"benchmark-vector-{i}",
            resume_id=f"benchmark-resume-{i % 10}",
            chunk_id=f"benchmark-chunk-{i}",
            candidate_name=f"Candidate {i % 5}",
            section="benchmark",
            vector=vector,
            metadata={"index": i}
        )
        vectors.append(record)
    
    return vectors


def benchmark_adapter(provider: str) -> Optional[Dict[str, Any]]:
    """
    Benchmark a specific vector store adapter.
    
    Args:
        provider: Provider name ('memory', 'pinecone', 'qdrant')
        
    Returns:
        Dictionary with benchmark results, or None if provider not available
    """
    print_header(f"BENCHMARKING {provider.upper()} ADAPTER")
    print()
    
    # Reset config to ensure fresh environment variable read
    reset_config()
    
    # Set environment for the provider
    os.environ["VECTOR_STORE_PROVIDER"] = provider
    os.environ["VECTOR_STORE_DIMENSION"] = "1024"
    
    # Provider-specific environment variables
    if provider == "pinecone":
        if not os.getenv("PINECONE_API_KEY") or not os.getenv("PINECONE_INDEX_NAME"):
            print_warning("PINECONE_API_KEY or PINECONE_INDEX_NAME not set")
            print_info("Skipping Pinecone adapter benchmark")
            print()
            return None
    elif provider == "qdrant":
        if not os.getenv("QDRANT_COLLECTION"):
            print_warning("QDRANT_COLLECTION not set")
            print_info("Skipping Qdrant adapter benchmark")
            print()
            return None
    
    results = {
        "provider": provider,
        "timestamp": datetime.now().isoformat(),
        "metrics": {}
    }
    
    try:
        # Measure connect time
        print_info("Measuring connect time...")
        memory_before = get_memory_usage()
        connect_start = time.time()
        
        vector_store_service = VectorStoreService()
        
        # Call connect if available
        if hasattr(vector_store_service.vector_store, 'connect'):
            vector_store_service.vector_store.connect()
        
        connect_time = time.time() - connect_start
        memory_after = get_memory_usage()
        memory_usage = memory_after - memory_before
        
        results["metrics"]["connect_time_seconds"] = connect_time
        results["metrics"]["memory_usage_mb"] = memory_usage
        
        print_success(f"Connect time: {connect_time:.3f}s")
        print_info(f"Memory usage: {memory_usage:.2f} MB")
        print()
        
        # Benchmark upsert 100 vectors
        print_info("Benchmarking upsert 100 vectors...")
        vectors_100 = generate_test_vectors(100)
        upsert_100_start = time.time()
        
        result = vector_store_service.upsert(vectors_100)
        
        upsert_100_time = time.time() - upsert_100_start
        upsert_100_throughput = 100 / upsert_100_time
        
        results["metrics"]["upsert_100_time_seconds"] = upsert_100_time
        results["metrics"]["upsert_100_throughput_vectors_per_second"] = upsert_100_throughput
        
        print_success(f"Upsert 100 vectors: {upsert_100_time:.3f}s")
        print_info(f"Throughput: {upsert_100_throughput:.2f} vectors/sec")
        print()
        
        # Benchmark upsert 1000 vectors
        print_info("Benchmarking upsert 1000 vectors...")
        vectors_1000 = generate_test_vectors(1000)
        upsert_1000_start = time.time()
        
        result = vector_store_service.upsert(vectors_1000)
        
        upsert_1000_time = time.time() - upsert_1000_start
        upsert_1000_throughput = 1000 / upsert_1000_time
        
        results["metrics"]["upsert_1000_time_seconds"] = upsert_1000_time
        results["metrics"]["upsert_1000_throughput_vectors_per_second"] = upsert_1000_throughput
        
        print_success(f"Upsert 1000 vectors: {upsert_1000_time:.3f}s")
        print_info(f"Throughput: {upsert_1000_throughput:.2f} vectors/sec")
        print()
        
        # Benchmark query latency
        print_info("Benchmarking query latency...")
        query_vector = vectors_100[0].vector
        query_latencies = []
        
        for _ in range(10):
            query_start = time.time()
            results_query = vector_store_service.query(query_vector, k=10)
            query_latency = time.time() - query_start
            query_latencies.append(query_latency)
        
        avg_query_latency = sum(query_latencies) / len(query_latencies)
        min_query_latency = min(query_latencies)
        max_query_latency = max(query_latencies)
        
        results["metrics"]["query_latency_avg_seconds"] = avg_query_latency
        results["metrics"]["query_latency_min_seconds"] = min_query_latency
        results["metrics"]["query_latency_max_seconds"] = max_query_latency
        
        print_success(f"Query latency (avg): {avg_query_latency:.4f}s")
        print_info(f"Query latency (min): {min_query_latency:.4f}s")
        print_info(f"Query latency (max): {max_query_latency:.4f}s")
        print()
        
        # Benchmark delete latency
        print_info("Benchmarking delete latency...")
        ids_to_delete = [vectors_100[i].id for i in range(100)]
        delete_start = time.time()
        
        result = vector_store_service.delete(ids_to_delete)
        
        delete_time = time.time() - delete_start
        delete_throughput = 100 / delete_time
        
        results["metrics"]["delete_100_time_seconds"] = delete_time
        results["metrics"]["delete_100_throughput_vectors_per_second"] = delete_throughput
        
        print_success(f"Delete 100 vectors: {delete_time:.3f}s")
        print_info(f"Throughput: {delete_throughput:.2f} vectors/sec")
        print()
        
        # Measure final memory usage
        final_memory = get_memory_usage()
        results["metrics"]["final_memory_mb"] = final_memory
        print_info(f"Final memory usage: {final_memory:.2f} MB")
        print()
        
        # Close connection
        vector_store_service.close()
        
        print_success(f"{provider.upper()} benchmark completed successfully")
        print()
        
        return results
        
    except Exception as e:
        print_failure(f"Benchmark failed for {provider}: {e}")
        import traceback
        traceback.print_exc()
        print()
        return None


def print_comparison_table(results: List[Dict[str, Any]]):
    """
    Print a comparison table of benchmark results.
    
    Args:
        results: List of benchmark result dictionaries
    """
    print_header("BENCHMARK COMPARISON TABLE")
    print()
    
    if not results:
        print_warning("No benchmark results to display")
        return
    
    # Print table header
    print(f"{'Metric':<40} {'':<10}", end="")
    for result in results:
        print(f"{result['provider'].upper():<15}", end="")
    print()
    print("-" * (40 + 10 + 15 * len(results)))
    
    # Print metrics
    metrics = [
        ("Connect Time (s)", "connect_time_seconds"),
        ("Memory Usage (MB)", "memory_usage_mb"),
        ("Upsert 100 Time (s)", "upsert_100_time_seconds"),
        ("Upsert 100 Throughput (v/s)", "upsert_100_throughput_vectors_per_second"),
        ("Upsert 1000 Time (s)", "upsert_1000_time_seconds"),
        ("Upsert 1000 Throughput (v/s)", "upsert_1000_throughput_vectors_per_second"),
        ("Query Latency Avg (s)", "query_latency_avg_seconds"),
        ("Query Latency Min (s)", "query_latency_min_seconds"),
        ("Query Latency Max (s)", "query_latency_max_seconds"),
        ("Delete 100 Time (s)", "delete_100_time_seconds"),
        ("Delete 100 Throughput (v/s)", "delete_100_throughput_vectors_per_second"),
        ("Final Memory (MB)", "final_memory_mb"),
    ]
    
    for metric_name, metric_key in metrics:
        print(f"{metric_name:<40} {'':<10}", end="")
        for result in results:
            value = result["metrics"].get(metric_key, "N/A")
            if isinstance(value, float):
                print(f"{value:<15.4f}", end="")
            else:
                print(f"{str(value):<15}", end="")
        print()
    
    print()


def generate_recommendations(results: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Generate recommendations for different use cases.
    
    Args:
        results: List of benchmark result dictionaries
        
    Returns:
        Dictionary of recommendations
    """
    recommendations = {}
    
    if not results:
        return {
            "development": "Memory (fastest, no setup required)",
            "small_projects": "Memory (sufficient for small datasets)",
            "enterprise": "Pinecone or Qdrant (scalable, managed service)",
            "high_throughput": "Pinecone or Qdrant (optimized for scale)"
        }
    
    # Analyze results
    memory_result = next((r for r in results if r["provider"] == "memory"), None)
    pinecone_result = next((r for r in results if r["provider"] == "pinecone"), None)
    qdrant_result = next((r for r in results if r["provider"] == "qdrant"), None)
    
    # Development recommendation
    if memory_result:
        recommendations["development"] = "Memory (fastest, no setup required, ideal for development)"
    else:
        recommendations["development"] = "Memory (not benchmarked, but recommended for development)"
    
    # Small projects recommendation
    if memory_result:
        upsert_100_throughput = memory_result["metrics"].get("upsert_100_throughput_vectors_per_second", 0)
        if upsert_100_throughput > 100:
            recommendations["small_projects"] = "Memory (sufficient throughput for small datasets)"
        else:
            recommendations["small_projects"] = "Memory (good for small datasets, consider Pinecone/Qdrant for scale)"
    else:
        recommendations["small_projects"] = "Memory (good starting point, upgrade as needed)"
    
    # Enterprise recommendation
    if pinecone_result or qdrant_result:
        if pinecone_result and qdrant_result:
            pinecone_throughput = pinecone_result["metrics"].get("upsert_1000_throughput_vectors_per_second", 0)
            qdrant_throughput = qdrant_result["metrics"].get("upsert_1000_throughput_vectors_per_second", 0)
            
            if pinecone_throughput > qdrant_throughput:
                recommendations["enterprise"] = "Pinecone (higher throughput, managed service)"
            else:
                recommendations["enterprise"] = "Qdrant (higher throughput, open-source option)"
        elif pinecone_result:
            recommendations["enterprise"] = "Pinecone (managed service, scalable)"
        elif qdrant_result:
            recommendations["enterprise"] = "Qdrant (open-source, self-hosted option)"
    else:
        recommendations["enterprise"] = "Pinecone or Qdrant (not benchmarked, but recommended for enterprise)"
    
    # High throughput recommendation
    if pinecone_result or qdrant_result:
        if pinecone_result and qdrant_result:
            pinecone_throughput = pinecone_result["metrics"].get("upsert_1000_throughput_vectors_per_second", 0)
            qdrant_throughput = qdrant_result["metrics"].get("upsert_1000_throughput_vectors_per_second", 0)
            
            if pinecone_throughput > qdrant_throughput:
                recommendations["high_throughput"] = "Pinecone (optimized for high throughput)"
            else:
                recommendations["high_throughput"] = "Qdrant (optimized for high throughput)"
        elif pinecone_result:
            recommendations["high_throughput"] = "Pinecone (optimized for scale)"
        elif qdrant_result:
            recommendations["high_throughput"] = "Qdrant (optimized for scale)"
    else:
        recommendations["high_throughput"] = "Pinecone or Qdrant (not benchmarked, but recommended for high throughput)"
    
    return recommendations


def print_recommendations(recommendations: Dict[str, str]):
    """
    Print recommendations for different use cases.
    
    Args:
        recommendations: Dictionary of recommendations
    """
    print_header("RECOMMENDATIONS")
    print()
    
    use_cases = [
        ("Development", "development"),
        ("Small Projects", "small_projects"),
        ("Enterprise", "enterprise"),
        ("High Throughput", "high_throughput")
    ]
    
    for use_case_name, use_case_key in use_cases:
        recommendation = recommendations.get(use_case_key, "No recommendation available")
        print(f"{use_case_name}:")
        print(f"  → {recommendation}")
        print()


def generate_markdown_report(results: List[Dict[str, Any]], recommendations: Dict[str, str]) -> str:
    """
    Generate a markdown report of benchmark results.
    
    Args:
        results: List of benchmark result dictionaries
        recommendations: Dictionary of recommendations
        
    Returns:
        Markdown report string
    """
    report = []
    
    report.append("# Vector Store Adapter Benchmark Report")
    report.append("")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Summary
    report.append("## Summary")
    report.append("")
    report.append(f"Total adapters benchmarked: {len(results)}")
    report.append("")
    
    for result in results:
        report.append(f"- **{result['provider'].upper()}**: Benchmark completed successfully")
    report.append("")
    
    # Comparison Table
    report.append("## Comparison Table")
    report.append("")
    report.append("| Metric | " + " | ".join([r['provider'].upper() for r in results]) + " |")
    report.append("|" + "---|" * (len(results) + 1))
    
    metrics = [
        ("Connect Time (s)", "connect_time_seconds"),
        ("Memory Usage (MB)", "memory_usage_mb"),
        ("Upsert 100 Time (s)", "upsert_100_time_seconds"),
        ("Upsert 100 Throughput (v/s)", "upsert_100_throughput_vectors_per_second"),
        ("Upsert 1000 Time (s)", "upsert_1000_time_seconds"),
        ("Upsert 1000 Throughput (v/s)", "upsert_1000_throughput_vectors_per_second"),
        ("Query Latency Avg (s)", "query_latency_avg_seconds"),
        ("Query Latency Min (s)", "query_latency_min_seconds"),
        ("Query Latency Max (s)", "query_latency_max_seconds"),
        ("Delete 100 Time (s)", "delete_100_time_seconds"),
        ("Delete 100 Throughput (v/s)", "delete_100_throughput_vectors_per_second"),
        ("Final Memory (MB)", "final_memory_mb"),
    ]
    
    for metric_name, metric_key in metrics:
        row = f"| {metric_name} |"
        for result in results:
            value = result["metrics"].get(metric_key, "N/A")
            if isinstance(value, float):
                row += f" {value:.4f} |"
            else:
                row += f" {str(value)} |"
        report.append(row)
    
    report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    report.append("")
    
    use_cases = [
        ("Development", "development"),
        ("Small Projects", "small_projects"),
        ("Enterprise", "enterprise"),
        ("High Throughput", "high_throughput")
    ]
    
    for use_case_name, use_case_key in use_cases:
        recommendation = recommendations.get(use_case_key, "No recommendation available")
        report.append(f"### {use_case_name}")
        report.append("")
        report.append(f"{recommendation}")
        report.append("")
    
    # Detailed Results
    report.append("## Detailed Results")
    report.append("")
    
    for result in results:
        report.append(f"### {result['provider'].upper()}")
        report.append("")
        report.append(f"**Timestamp:** {result['timestamp']}")
        report.append("")
        report.append("#### Metrics")
        report.append("")
        for key, value in result["metrics"].items():
            if isinstance(value, float):
                report.append(f"- {key}: {value:.4f}")
            else:
                report.append(f"- {key}: {value}")
        report.append("")
    
    return "\n".join(report)


def save_results(results: List[Dict[str, Any]], recommendations: Dict[str, str]):
    """
    Save benchmark results to files.
    
    Args:
        results: List of benchmark result dictionaries
        recommendations: Dictionary of recommendations
    """
    # Create benchmarks directory if it doesn't exist
    benchmarks_dir = Path(__file__).parent
    benchmarks_dir.mkdir(exist_ok=True)
    
    # Save JSON results
    json_file = benchmarks_dir / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_file, 'w') as f:
        json.dump({
            "results": results,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    print_success(f"Results saved to: {json_file}")
    
    # Save markdown report
    markdown_file = benchmarks_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    markdown_report = generate_markdown_report(results, recommendations)
    
    with open(markdown_file, 'w') as f:
        f.write(markdown_report)
    
    print_success(f"Markdown report saved to: {markdown_file}")
    print()


def run_benchmarks():
    """
    Run benchmarks for all available vector store adapters.
    """
    print_header("VECTOR STORE ADAPTER BENCHMARKS")
    print()
    
    providers_to_benchmark = ['memory', 'pinecone', 'qdrant']
    results = []
    
    for provider in providers_to_benchmark:
        result = benchmark_adapter(provider)
        if result:
            results.append(result)
    
    # Print comparison table
    if results:
        print_comparison_table(results)
        
        # Generate and print recommendations
        recommendations = generate_recommendations(results)
        print_recommendations(recommendations)
        
        # Save results
        save_results(results, recommendations)
        
        # Final summary
        print_header("BENCHMARK SUMMARY")
        print()
        print_success(f"Benchmarks completed for {len(results)} adapters")
        print()
        print("🚀 Vector Store Benchmarking Complete")
        
        return True
    else:
        print_warning("No benchmarks completed")
        return False


if __name__ == "__main__":
    try:
        success = run_benchmarks()
        sys.exit(0 if success else 1)
    except Exception as e:
        print_failure(f"Benchmark execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
