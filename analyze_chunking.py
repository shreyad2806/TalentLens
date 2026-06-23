import csv
import statistics
from pathlib import Path

lengths = []
with open('Resume/Resume.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        text = row.get('Resume_str', '')
        lengths.append(len(text))

print(f'Total resumes: {len(lengths)}')
print(f'Average length: {statistics.mean(lengths):.0f} chars')
print(f'Median length: {statistics.median(lengths):.0f} chars')
print(f'Min length: {min(lengths)} chars')
print(f'Max length: {max(lengths)} chars')
print(f'P25: {sorted(lengths)[int(len(lengths)*0.25)]:.0f} chars')
print(f'P75: {sorted(lengths)[int(len(lengths)*0.75)]:.0f} chars')
print(f'P90: {sorted(lengths)[int(len(lengths)*0.90)]:.0f} chars')

# Current chunking analysis
current_chunk_size = 1000
current_overlap = 100
current_chunks = 18569
current_resumes = 2484
avg_chunks_per_resume = current_chunks / current_resumes

print(f'\n=== CURRENT CHUNKING ===')
print(f'Chunk size: {current_chunk_size} chars')
print(f'Overlap: {current_overlap} chars')
print(f'Average chunks per resume: {avg_chunks_per_resume:.2f}')
print(f'Total chunks: {current_chunks}')

# Projected chunking for 1-3 chunks per resume
target_avg_chunks = 2  # Target 1-3 chunks per resume, use 2 as average
projected_chunks = current_resumes * target_avg_chunks
reduction_factor = current_chunks / projected_chunks
embedding_reduction = (current_chunks - projected_chunks) / current_chunks * 100

print(f'\n=== PROJECTED CHUNKING (Target: 1-3 chunks per resume) ===')
print(f'Target average chunks per resume: {target_avg_chunks}')
print(f'Projected total chunks: {projected_chunks}')
print(f'Chunk reduction: {current_chunks - projected_chunks} chunks')
print(f'Embedding reduction: {embedding_reduction:.1f}%')
print(f'Speedup factor: {reduction_factor:.1f}x')

# Recommended chunk size
avg_resume_length = statistics.mean(lengths)
recommended_chunk_size = int(avg_resume_length / target_avg_chunks)
print(f'\n=== RECOMMENDATIONS ===')
print(f'Average resume length: {avg_resume_length:.0f} chars')
print(f'Recommended chunk size: {recommended_chunk_size} chars')
print(f'Recommended overlap: {int(recommended_chunk_size * 0.1)} chars (10%)')
