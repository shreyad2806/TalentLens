"""
Verification script for the Production Indexing Service.

This script demonstrates and verifies that the indexing pipeline works correctly
by creating test resume files and running them through the complete indexing workflow.
"""

import tempfile
from pathlib import Path
from src.indexing.pipeline import IndexingPipeline
from src.config import EMBEDDING_DIM


def create_test_resume():
    """Create a simple test resume file."""
    resume_text = """
John Doe
Software Engineer
john.doe@email.com | (555) 123-4567 | San Francisco, CA

SUMMARY
Experienced software engineer with 5+ years of experience in Python, Java, and cloud technologies.
Passionate about building scalable systems and mentoring junior developers.

SKILLS
Programming: Python, Java, JavaScript, SQL
Cloud: AWS, GCP, Docker, Kubernetes
Tools: Git, Jenkins, Terraform

EXPERIENCE

Senior Software Engineer
TechCorp Inc. | San Francisco, CA | 2020 - Present
- Led development of microservices architecture serving 1M+ users
- Mentored team of 5 junior developers
- Improved system performance by 40% through optimization

Software Engineer
StartupXYZ | San Francisco, CA | 2018 - 2020
- Built RESTful APIs using Python and Flask
- Implemented CI/CD pipelines using Jenkins
- Collaborated with product team to deliver features on time

EDUCATION
Bachelor of Science in Computer Science
University of California, Berkeley | 2014 - 2018

CERTIFICATIONS
AWS Certified Solutions Architect
Google Cloud Professional Developer
"""
    return resume_text


def verify_indexing_pipeline():
    """Verify the indexing pipeline works correctly."""
    print("="*60)
    print("🧪 Verifying Production Indexing Service")
    print("="*60)
    
    # Create a temporary directory with test resume files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test resume files
        print("\n📝 Creating test resume files...")
        
        # Create 3 test resumes
        for i in range(1, 4):
            resume_file = temp_path / f"resume_{i}.txt"
            resume_file.write_text(create_test_resume())
            print(f"   Created: {resume_file.name}")
        
        # Initialize the indexing pipeline
        print("\n🚀 Initializing indexing pipeline...")
        pipeline = IndexingPipeline(embedding_dim=EMBEDDING_DIM)
        
        # Print initial status
        print("\n📊 Initial Index Status:")
        pipeline.print_startup_status()
        
        # Index the directory
        print("\n🔍 Running indexing pipeline...")
        result = pipeline.index_directory(temp_dir, verbose=True)
        
        # Verify results
        print("\n✅ Verification Results:")
        print("="*60)
        
        # Check ingestion
        ingestion = result['ingestion']
        print(f"Files Discovered: {ingestion.total_files}")
        print(f"Valid Resumes: {ingestion.valid_files}")
        print(f"Invalid Files: {ingestion.invalid_files}")
        
        # Check indexing
        indexing = result['indexing']
        if indexing:
            print(f"\nIndexing Results:")
            print(f"Total Files: {indexing['total_files']}")
            print(f"Successful: {indexing['successful']}")
            print(f"Failed: {indexing['failed']}")
            print(f"Total Chunks: {indexing['total_chunks']}")
            print(f"Total Embeddings: {indexing['total_embeddings']}")
        
        # Check statistics
        stats = result['statistics']
        print(f"\nFinal Statistics:")
        print(f"Indexed Documents: {stats['indexed_documents']}")
        print(f"Vector Count: {stats['vector_count']}")
        print(f"BM25 Count: {stats['bm25_count']}")
        
        # Verify counts are greater than 0
        print("\n🔍 Verification Checks:")
        print("="*60)
        
        docs_ok = stats['indexed_documents'] > 0
        vectors_ok = stats['vector_count'] > 0
        bm25_ok = stats['bm25_count'] > 0
        
        print(f"Documents > 0: {'✅ PASS' if docs_ok else '❌ FAIL'} ({stats['indexed_documents']})")
        print(f"Vectors > 0: {'✅ PASS' if vectors_ok else '❌ FAIL'} ({stats['vector_count']})")
        print(f"BM25 Docs > 0: {'✅ PASS' if bm25_ok else '❌ FAIL'} ({stats['bm25_count']})")
        
        # Final status
        print("\n" + "="*60)
        if docs_ok and vectors_ok and bm25_ok:
            print("🚀 Indexing Ready - All checks passed!")
        else:
            print("⚠️  Indexing incomplete - Some checks failed")
        print("="*60)
        
        return docs_ok and vectors_ok and bm25_ok


if __name__ == "__main__":
    success = verify_indexing_pipeline()
    exit(0 if success else 1)
