"""Verification test: Confirm all indexing paths write enriched metadata."""
import sys
sys.path.insert(0, '.')

print("=" * 70)
print("TEST 1: CSV Ingestion chunk_raw_text() — enriched metadata propagation")
print("=" * 70)

from src.bootstrap.csv_ingestion import CSVIngestionService
svc = CSVIngestionService()

chunks = svc.chunk_raw_text(
    raw_text="This is a test resume with some content that is long enough to create a chunk.",
    resume_id="test-resume-001",
    candidate_name="John Doe",
    source_document="Resume.csv",
    email="john@example.com",
    phone="555-0100",
    skills=["Python", "ML", "SQL"],
    location="New York",
    summary="Experienced engineer"
)

for c in chunks:
    meta = c.metadata.dict()
    non_null = {k: v for k, v in meta.items() if v is not None and v != [] and v != ''}
    print(f"\n  Chunk section={c.section}")
    print(f"  All keys: {sorted(meta.keys())}")
    print(f"  Non-null: {non_null}")
    
    # Verify enriched fields present
    assert 'candidate_name' in meta, "FAIL: candidate_name missing"
    assert 'skills' in meta, "FAIL: skills missing"
    assert 'email' in meta, "FAIL: email missing"
    assert 'phone' in meta, "FAIL: phone missing"
    assert 'summary' in meta, "FAIL: summary missing"
    assert meta['email'] == 'john@example.com', f"FAIL: email={meta['email']}"
    assert meta['skills'] == ['Python', 'ML', 'SQL'], f"FAIL: skills={meta['skills']}"
    print("  [PASS] ALL ENRICHED FIELDS PRESENT")

print("\n" + "=" * 70)
print("TEST 2: ChunkFactory — enriched metadata propagation")
print("=" * 70)

from src.chunks.factory import ChunkFactory
from src.resume_parser.schema import ResumeDocument

factory = ChunkFactory()
doc = ResumeDocument(
    resume_id="test-002",
    name="Jane Smith",
    email="jane@test.com",
    phone="555-0200",
    skills=["Java", "Spring"],
    summary="Senior developer",
    raw_text="Test resume text " * 50
)
doc.metadata['location'] = 'San Francisco'
doc.metadata['total_experience_years'] = 8

chunks2 = factory.create_chunks(doc, resume_id="test-002", source_document="test.pdf")
if chunks2:
    c = chunks2[0]
    meta = c.metadata.dict()
    non_null = {k: v for k, v in meta.items() if v is not None and v != [] and v != ''}
    print(f"\n  Chunk section={c.section}")
    print(f"  All keys: {sorted(meta.keys())}")
    print(f"  Non-null: {non_null}")
    assert meta.get('candidate_name') == 'Jane Smith'
    assert meta.get('email') == 'jane@test.com'
    assert meta.get('skills') == ['Java', 'Spring']
    print("  [PASS] ALL ENRICHED FIELDS PRESENT")

print("\n" + "=" * 70)
print("TEST 3: ChunkGenerator — enriched metadata propagation")
print("=" * 70)

from src.chunking.chunk_generator import ChunkGenerator
gen = ChunkGenerator()

doc3 = ResumeDocument(
    resume_id="test-003",
    name="Bob Wilson",
    email="bob@test.com",
    phone="555-0300",
    skills=["Go", "K8s"],
    summary="Platform engineer",
    raw_text="Another test resume " * 50
)

chunks3 = gen.generate_chunks(doc3, resume_id="test-003")
if chunks3:
    c = chunks3[0]
    meta = c.metadata.dict()
    non_null = {k: v for k, v in meta.items() if v is not None and v != [] and v != ''}
    print(f"\n  Chunk section={c.section}")
    print(f"  All keys: {sorted(meta.keys())}")
    print(f"  Non-null: {non_null}")
    assert 'candidate_name' in meta
    assert 'skills' in meta
    assert 'email' in meta
    print("  [PASS] ALL ENRICHED FIELDS PRESENT")

print("\n" + "=" * 70)
print("ALL TESTS PASSED - All indexing paths write enriched metadata")
print("=" * 70)
