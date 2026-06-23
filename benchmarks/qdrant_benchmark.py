"""
Qdrant Adapter Benchmark.

This module benchmarks the Qdrant adapter against the Memory adapter,
measuring search latency, insert latency, throughput, and memory usage.
"""

import time
import psutil
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass

from src.vector_store.qdrant import QdrantAdapter
from src.vector_store.memory_adapter import MemoryAdapter


@dataclass
class BenchmarkResult:
    """Result of a benchmark operation."""
    operation: str
    adapter: str
    latency_ms: float
    throughput_ops_per_sec: float
    memory_usage_mb: float
    success: bool
    error_message: str = ""


class QdrantBenchmark:
    """
    Benchmark suite for Qdrant adapter.
    
    This class compares Qdrant adapter performance against Memory adapter
    across various operations including insert, search, and filtering.
    """
    
    def __init__(
        self,
        vector_size: int = 1024,
        num_vectors: int = 1000,
        num_searches: int = 100
    ):
        """
        Initialize the benchmark suite.
        
        Args:
            vector_size: Dimension of vectors to benchmark
            num_vectors: Number of vectors to insert
            num_searches: Number of search operations to perform
        """
        self.vector_size = vector_size
        self.num_vectors = num_vectors
        self.num_searches = num_searches
        
        # Generate test data
        self.test_vectors = self._generate_vectors(num_vectors, vector_size)
        self.test_query_vectors = self._generate_vectors(num_searches, vector_size)
        self.test_payloads = self._generate_payloads(num_vectors)
        
        print(f"QdrantBenchmark initialized - Vectors: {num_vectors}, Searches: {num_searches}")
    
    def _generate_vectors(self, count: int, size: int) -> List[List[float]]:
        """Generate random test vectors."""
        return [np.random.rand(size).tolist() for _ in range(count)]
    
    def _generate_payloads(self, count: int) -> List[Dict[str, Any]]:
        """Generate test payloads."""
        payloads = []
        skills_list = ["Python", "SQL", "AWS", "Docker", "Kubernetes", "React", "Node.js", "Java"]
        locations = ["Bangalore", "Mumbai", "Delhi", "Hyderabad", "Remote"]
        roles = ["Software Engineer", "Data Scientist", "DevOps Engineer", "ML Engineer"]
        
        for i in range(count):
            payload = {
                "resume_id": f"resume_{i}",
                "candidate_name": f"Candidate {i}",
                "chunk_id": f"chunk_{i}",
                "section": "Skills",
                "skills": np.random.choice(skills_list, size=np.random.randint(1, 4), replace=False).tolist(),
                "experience": float(np.random.randint(0, 15)),
                "location": np.random.choice(locations),
                "education": np.random.choice(["Bachelor's", "Master's", "PhD"]),
                "role": np.random.choice(roles),
                "salary": float(np.random.randint(10, 50)),
                "notice_period": np.random.randint(0, 90),
                "metadata": {"test": True}
            }
            payloads.append(payload)
        
        return payloads
    
    def benchmark_insert(self, adapter, adapter_name: str) -> BenchmarkResult:
        """
        Benchmark vector insertion.
        
        Args:
            adapter: Vector store adapter instance
            adapter_name: Name of the adapter for reporting
        
        Returns:
            BenchmarkResult with insertion metrics
        """
        print(f"\nBenchmarking INSERT - {adapter_name}")
        
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        try:
            # Create collection if needed
            if hasattr(adapter, 'create_collection'):
                adapter.create_collection()
            
            # Benchmark insert
            start_time = time.time()
            
            if hasattr(adapter, 'batch_insert'):
                result = adapter.batch_insert(
                    self.test_vectors,
                    self.test_payloads,
                    batch_size=100
                )
            elif hasattr(adapter, 'upsert_vectors'):
                result = adapter.upsert_vectors(
                    self.test_vectors,
                    self.test_payloads
                )
            else:
                raise AttributeError("Adapter has no insert method")
            
            latency_ms = (time.time() - start_time) * 1000
            throughput = self.num_vectors / (latency_ms / 1000)
            
            # Get final memory
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_usage = final_memory - initial_memory
            
            benchmark_result = BenchmarkResult(
                operation="insert",
                adapter=adapter_name,
                latency_ms=latency_ms,
                throughput_ops_per_sec=throughput,
                memory_usage_mb=memory_usage,
                success=True
            )
            
            print(f"  ✓ Inserted {self.num_vectors} vectors")
            print(f"  ✓ Latency: {latency_ms:.2f}ms")
            print(f"  ✓ Throughput: {throughput:.2f} ops/sec")
            print(f"  ✓ Memory Usage: {memory_usage:.2f}MB")
            
            return benchmark_result
            
        except Exception as e:
            print(f"  ✗ Insert failed: {str(e)}")
            return BenchmarkResult(
                operation="insert",
                adapter=adapter_name,
                latency_ms=0,
                throughput_ops_per_sec=0,
                memory_usage_mb=0,
                success=False,
                error_message=str(e)
            )
    
    def benchmark_search(self, adapter, adapter_name: str) -> BenchmarkResult:
        """
        Benchmark vector search.
        
        Args:
            adapter: Vector store adapter instance
            adapter_name: Name of the adapter for reporting
        
        Returns:
            BenchmarkResult with search metrics
        """
        print(f"\nBenchmarking SEARCH - {adapter_name}")
        
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        try:
            # Benchmark search
            start_time = time.time()
            
            total_results = 0
            for query_vector in self.test_query_vectors:
                if hasattr(adapter, 'search'):
                    results = adapter.search(query_vector, top_k=10)
                    total_results += len(results)
                else:
                    raise AttributeError("Adapter has no search method")
            
            latency_ms = (time.time() - start_time) * 1000
            throughput = self.num_searches / (latency_ms / 1000)
            
            # Get final memory
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_usage = final_memory - initial_memory
            
            benchmark_result = BenchmarkResult(
                operation="search",
                adapter=adapter_name,
                latency_ms=latency_ms,
                throughput_ops_per_sec=throughput,
                memory_usage_mb=memory_usage,
                success=True
            )
            
            print(f"  ✓ Performed {self.num_searches} searches")
            print(f"  ✓ Total Results: {total_results}")
            print(f"  ✓ Latency: {latency_ms:.2f}ms")
            print(f"  ✓ Throughput: {throughput:.2f} ops/sec")
            print(f"  ✓ Memory Usage: {memory_usage:.2f}MB")
            
            return benchmark_result
            
        except Exception as e:
            print(f"  ✗ Search failed: {str(e)}")
            return BenchmarkResult(
                operation="search",
                adapter=adapter_name,
                latency_ms=0,
                throughput_ops_per_sec=0,
                memory_usage_mb=0,
                success=False,
                error_message=str(e)
            )
    
    def benchmark_search_with_filters(self, adapter, adapter_name: str) -> BenchmarkResult:
        """
        Benchmark vector search with filters.
        
        Args:
            adapter: Vector store adapter instance
            adapter_name: Name of the adapter for reporting
        
        Returns:
            BenchmarkResult with filtered search metrics
        """
        print(f"\nBenchmarking SEARCH_WITH_FILTERS - {adapter_name}")
        
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        try:
            from src.vector_store.qdrant.schema import QdrantFilter
            
            # Create filter
            filters = QdrantFilter(
                skills=["Python"],
                experience_min=3,
                location="Bangalore"
            )
            
            # Benchmark search with filters
            start_time = time.time()
            
            total_results = 0
            for query_vector in self.test_query_vectors[:10]:  # Use fewer queries for filtered search
                if hasattr(adapter, 'search_with_filters'):
                    results = adapter.search_with_filters(query_vector, filters, top_k=10)
                    total_results += len(results)
                else:
                    raise AttributeError("Adapter has no search_with_filters method")
            
            latency_ms = (time.time() - start_time) * 1000
            throughput = 10 / (latency_ms / 1000)
            
            # Get final memory
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_usage = final_memory - initial_memory
            
            benchmark_result = BenchmarkResult(
                operation="search_with_filters",
                adapter=adapter_name,
                latency_ms=latency_ms,
                throughput_ops_per_sec=throughput,
                memory_usage_mb=memory_usage,
                success=True
            )
            
            print(f"  ✓ Performed 10 filtered searches")
            print(f"  ✓ Total Results: {total_results}")
            print(f"  ✓ Latency: {latency_ms:.2f}ms")
            print(f"  ✓ Throughput: {throughput:.2f} ops/sec")
            print(f"  ✓ Memory Usage: {memory_usage:.2f}MB")
            
            return benchmark_result
            
        except Exception as e:
            print(f"  ✗ Search with filters failed: {str(e)}")
            return BenchmarkResult(
                operation="search_with_filters",
                adapter=adapter_name,
                latency_ms=0,
                throughput_ops_per_sec=0,
                memory_usage_mb=0,
                success=False,
                error_message=str(e)
            )
    
    def run_full_benchmark(self) -> Dict[str, List[BenchmarkResult]]:
        """
        Run full benchmark comparing Memory and Qdrant adapters.
        
        Returns:
            Dictionary mapping adapter names to benchmark results
        """
        print("\n" + "="*70)
        print("🚀 Qdrant Adapter Benchmark Suite")
        print("="*70)
        print(f"Vector Size: {self.vector_size}")
        print(f"Number of Vectors: {self.num_vectors}")
        print(f"Number of Searches: {self.num_searches}")
        print("="*70)
        
        results = {}
        
        # Benchmark Memory Adapter
        print("\n" + "-"*70)
        print("📦 Memory Adapter")
        print("-"*70)
        
        try:
            memory_adapter = MemoryAdapter(vector_size=self.vector_size)
            memory_results = []
            
            memory_results.append(self.benchmark_insert(memory_adapter, "Memory"))
            memory_results.append(self.benchmark_search(memory_adapter, "Memory"))
            
            results["Memory"] = memory_results
            
        except Exception as e:
            print(f"Memory adapter benchmark failed: {str(e)}")
            results["Memory"] = []
        
        # Benchmark Qdrant Adapter
        print("\n" + "-"*70)
        print("📦 Qdrant Adapter")
        print("-"*70)
        
        try:
            qdrant_adapter = QdrantAdapter(
                url="http://localhost:6333",
                collection_name="benchmark_collection",
                vector_size=self.vector_size
            )
            qdrant_results = []
            
            qdrant_results.append(self.benchmark_insert(qdrant_adapter, "Qdrant"))
            qdrant_results.append(self.benchmark_search(qdrant_adapter, "Qdrant"))
            qdrant_results.append(self.benchmark_search_with_filters(qdrant_adapter, "Qdrant"))
            
            # Cleanup
            qdrant_adapter.delete_collection()
            
            results["Qdrant"] = qdrant_results
            
        except Exception as e:
            print(f"Qdrant adapter benchmark failed: {str(e)}")
            print("Note: Make sure Qdrant is running at http://localhost:6333")
            results["Qdrant"] = []
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, List[BenchmarkResult]]) -> None:
        """Print benchmark summary."""
        print("\n" + "="*70)
        print("📊 Benchmark Summary")
        print("="*70)
        
        for adapter_name, benchmark_results in results.items():
            if not benchmark_results:
                continue
            
            print(f"\n{adapter_name}:")
            for result in benchmark_results:
                if result.success:
                    print(f"  {result.operation.upper()}:")
                    print(f"    Latency: {result.latency_ms:.2f}ms")
                    print(f"    Throughput: {result.throughput_ops_per_sec:.2f} ops/sec")
                    print(f"    Memory: {result.memory_usage_mb:.2f}MB")
                else:
                    print(f"  {result.operation.upper()}: FAILED")
                    print(f"    Error: {result.error_message}")
        
        print("\n" + "="*70 + "\n")


def main():
    """Main entry point for benchmark."""
    benchmark = QdrantBenchmark(
        vector_size=1024,
        num_vectors=1000,
        num_searches=100
    )
    
    results = benchmark.run_full_benchmark()
    
    return results


if __name__ == "__main__":
    main()
