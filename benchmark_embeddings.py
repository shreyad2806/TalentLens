import csv
import time
import psutil
import statistics
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

def load_resume_chunks(csv_path, num_chunks=100):
    """Load resume chunks from CSV."""
    chunks = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get('Resume_str', '')
            if text.strip():
                chunks.append(text.strip())
                if len(chunks) >= num_chunks:
                    break
    return chunks

def benchmark_model(model_name, chunks):
    """Benchmark an embedding model."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*60}")
    
    # Load model
    print(f"Loading model...")
    load_start = time.time()
    model = SentenceTransformer(model_name)
    load_time = time.time() - load_start
    
    # Get model info
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Model loaded in {load_time:.2f}s")
    print(f"Embedding dimension: {embedding_dim}")
    
    # Measure memory usage before embedding
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Benchmark embedding speed
    print(f"\nEmbedding {len(chunks)} chunks...")
    embed_start = time.time()
    embeddings = model.encode(chunks, show_progress_bar=False)
    embed_time = time.time() - embed_start
    
    # Measure memory usage after embedding
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    mem_used = mem_after - mem_before
    
    # Calculate metrics
    avg_time_per_chunk = embed_time / len(chunks)
    chunks_per_second = len(chunks) / embed_time
    
    # Calculate retrieval quality (cosine similarity)
    # Use first chunk as query, find most similar
    query_embedding = embeddings[0]
    similarities = np.dot(embeddings[1:], query_embedding) / (
        np.linalg.norm(embeddings[1:], axis=1) * np.linalg.norm(query_embedding)
    )
    avg_similarity = np.mean(similarities)
    max_similarity = np.max(similarities)
    
    results = {
        'model_name': model_name,
        'load_time': load_time,
        'embedding_dim': embedding_dim,
        'embed_time': embed_time,
        'avg_time_per_chunk': avg_time_per_chunk,
        'chunks_per_second': chunks_per_second,
        'mem_used_mb': mem_used,
        'avg_similarity': avg_similarity,
        'max_similarity': max_similarity,
        'embeddings': embeddings
    }
    
    print(f"\n=== RESULTS ===")
    print(f"Load time: {load_time:.2f}s")
    print(f"Embedding time: {embed_time:.2f}s")
    print(f"Avg time per chunk: {avg_time_per_chunk*1000:.2f}ms")
    print(f"Chunks per second: {chunks_per_second:.2f}")
    print(f"Memory used: {mem_used:.2f} MB")
    print(f"Avg similarity: {avg_similarity:.4f}")
    print(f"Max similarity: {max_similarity:.4f}")
    
    return results

def main():
    print("="*60)
    print("EMBEDDING MODEL BENCHMARK")
    print("="*60)
    
    # Load resume chunks
    csv_path = Path('Resume/Resume.csv')
    chunks = load_resume_chunks(csv_path, num_chunks=100)
    print(f"Loaded {len(chunks)} resume chunks")
    print(f"Average chunk length: {statistics.mean([len(c) for c in chunks]):.0f} chars")
    
    # Benchmark models
    models = [
        'BAAI/bge-small-en-v1.5',
        'all-MiniLM-L6-v2'
    ]
    
    results = []
    for model_name in models:
        result = benchmark_model(model_name, chunks)
        results.append(result)
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    
    print(f"\n{'Metric':<25} {'bge-small':<15} {'MiniLM-L6':<15} {'Winner'}")
    print("-"*70)
    
    # Speed (chunks per second)
    bge_speed = results[0]['chunks_per_second']
    minilm_speed = results[1]['chunks_per_second']
    speed_winner = 'bge-small' if bge_speed > minilm_speed else 'MiniLM-L6'
    print(f"{'Chunks/sec':<25} {bge_speed:<15.2f} {minilm_speed:<15.2f} {speed_winner}")
    
    # Memory usage
    bge_mem = results[0]['mem_used_mb']
    minilm_mem = results[1]['mem_used_mb']
    mem_winner = 'bge-small' if bge_mem < minilm_mem else 'MiniLM-L6'
    print(f"{'Memory (MB)':<25} {bge_mem:<15.2f} {minilm_mem:<15.2f} {mem_winner}")
    
    # Embedding dimension
    bge_dim = results[0]['embedding_dim']
    minilm_dim = results[1]['embedding_dim']
    dim_winner = 'bge-small' if bge_dim < minilm_dim else 'MiniLM-L6'
    print(f"{'Embedding dim':<25} {bge_dim:<15} {minilm_dim:<15} {dim_winner}")
    
    # Retrieval quality (avg similarity)
    bge_sim = results[0]['avg_similarity']
    minilm_sim = results[1]['avg_similarity']
    sim_winner = 'bge-small' if bge_sim > minilm_sim else 'MiniLM-L6'
    print(f"{'Avg similarity':<25} {bge_sim:<15.4f} {minilm_sim:<15.4f} {sim_winner}")
    
    # Recommendation
    print(f"\n{'='*60}")
    print("RECOMMENDATION")
    print(f"{'='*60}")
    
    # Calculate scores
    # Normalize metrics (higher is better for speed and similarity, lower is better for memory and dim)
    bge_speed_score = bge_speed / (bge_speed + minilm_speed)
    minilm_speed_score = minilm_speed / (bge_speed + minilm_speed)
    
    bge_mem_score = minilm_mem / (bge_mem + minilm_mem)  # Lower memory is better
    minilm_mem_score = bge_mem / (bge_mem + minilm_mem)
    
    bge_dim_score = minilm_dim / (bge_dim + minilm_dim)  # Lower dim is better
    minilm_dim_score = bge_dim / (bge_dim + minilm_dim)
    
    bge_sim_score = bge_sim / (bge_sim + minilm_sim)
    minilm_sim_score = minilm_sim / (bge_sim + minilm_sim)
    
    # Weighted score (speed: 30%, memory: 20%, dim: 20%, similarity: 30%)
    bge_total = (bge_speed_score * 0.3 + bge_mem_score * 0.2 + bge_dim_score * 0.2 + bge_sim_score * 0.3)
    minilm_total = (minilm_speed_score * 0.3 + minilm_mem_score * 0.2 + minilm_dim_score * 0.2 + minilm_sim_score * 0.3)
    
    print(f"\nbge-small score: {bge_total:.3f}")
    print(f"MiniLM-L6 score: {minilm_total:.3f}")
    
    if bge_total > minilm_total:
        print(f"\n✓ RECOMMENDED: BAAI/bge-small-en-v1.5")
        print(f"  - Better overall performance")
        print(f"  - Higher retrieval quality")
    else:
        print(f"\n✓ RECOMMENDED: all-MiniLM-L6-v2")
        print(f"  - Better overall performance")
        print(f"  - Lower memory usage")

if __name__ == "__main__":
    main()
