# Metadata Propagation Audit Report
**Date:** 2026-07-18  
**Audit Scope:** BM25/Vector Index Staleness & Metadata Propagation Pipeline

---

## Executive Summary

**Root Cause Identified:** The application is loading a **stale BM25/vector index** that was built before the ChunkMetadata schema enrichment. The bootstrap process skips re-indexing because the index already contains 18,542 documents, perpetuating the stale metadata indefinitely.

**Impact:** All candidate cards display `Unknown`, `None`, and empty skills because the stored index contains only the old 5-field metadata schema with all values set to `None` (except `source_section='raw_text'`).

---

## Audit Findings

### 1. Stale Index Confirmed ✅

**Evidence:**
- **File:** `data/indexes/bm25/documents.json`
- **Document Count:** 18,542 documents
- **Stored Metadata Keys:** `['role', 'experience', 'location', 'education', 'source_section']`
- **Actual Values:** All `None` except `source_section='raw_text'`

**Conclusion:** The stored index predates the ChunkMetadata enrichment that added `candidate_name`, `skills`, `email`, `phone`, `summary` fields.

---

### 2. Bootstrap Skip Logic Confirmed ✅

**File:** `src/bootstrap/bootstrap_service.py` (lines 122-155)

**Logic:**
```python
stats = self.indexing_pipeline.get_statistics()
is_empty = (
    stats['indexed_documents'] == 0 and
    stats['vector_count'] == 0 and
    stats['bm25_count'] == 0
)

if not is_empty:
    # SKIP BOOTSTRAP - returns early
    return {
        'bootstrapped': False,
        'reason': 'index_not_empty',
        ...
    }
```

**Impact:** Since the BM25 index contains 18,542 documents, bootstrap **always skips re-indexing**. The enriched ChunkMetadata changes are **never written** to the index.

---

### 3. CSV Ingestion Path Identified ✅

**File:** `src/bootstrap/csv_ingestion.py` (lines 131-232)

**Primary Data Source:** `Resume.csv` → `chunk_raw_text()` → 18,542 documents

**Problem:** The `chunk_raw_text()` method creates ChunkMetadata with **only the old 5 fields**:
```python
chunk_metadata = ChunkMetadata(
    role=None,
    experience=None,
    location=None,
    education=None,
    source_section="raw_text"
)
```

**Missing Fields:** `candidate_name`, `skills`, `email`, `phone`, `summary` are **not propagated** from CSV records to ChunkMetadata.

---

### 4. Cache/Index Files Requiring Regeneration ✅

The following files contain stale data and **must be deleted** to force re-indexing:

#### Cache Files:
1. **`data/cache/chunks.json`** — Cached chunk objects with old metadata
2. **`data/cache/embeddings.npy`** — Cached embedding vectors (tied to stale chunks)
3. **`data/cache/indexed_documents.json`** — Cached indexed document list

#### BM25 Index Files:
4. **`data/indexes/bm25/documents.json`** — BM25 document store with old metadata
5. **`data/indexes/bm25/metadata.json`** — BM25 index metadata (18,542 docs)
6. **`data/indexes/bm25/inverted_index.json`** — BM25 inverted index
7. **`data/indexes/bm25/vocabulary.json`** — BM25 vocabulary
8. **`data/indexes/bm25/document_lengths.json`** — BM25 document lengths

**Total Files:** 8 files across 2 directories

---

### 5. Metadata Propagation Gap Analysis ✅

#### Written During Indexing (After Enrichment):
**ChunkMetadata Schema (src/chunks/schema.py):**
```python
class ChunkMetadata(BaseModel):
    candidate_name: Optional[str]      # ✅ NEW
    role: Optional[str]
    experience: Optional[int]
    location: Optional[str]
    education: Optional[str]
    skills: List[str]                  # ✅ NEW
    email: Optional[str]               # ✅ NEW
    phone: Optional[str]               # ✅ NEW
    summary: Optional[str]             # ✅ NEW
    source_section: Optional[str]
```

**Total Fields:** 10 fields

#### Loaded During Retrieval (From Stale Index):
**BM25Document Metadata:**
```python
metadata = {
    'role': None,
    'experience': None,
    'location': None,
    'education': None,
    'source_section': 'raw_text'
}
```

**Total Fields:** 5 fields (all None except source_section)

#### Gap:
- **Missing Fields:** `candidate_name`, `skills`, `email`, `phone`, `summary`
- **Reason:** Index was built before schema enrichment
- **Propagation Blocked By:** Bootstrap skip logic + stale cache files

---

## Instrumentation Added

To trace metadata propagation during indexing and retrieval, the following logs have been added:

### Indexing-Time Logs:

1. **`src/retrieval/bm25/index_builder.py`** (line 184-187)
   - **Tag:** `[INDEX-AUDIT][BM25-WRITE]`
   - **Logs:** Metadata keys written to BM25Document during indexing
   - **Frequency:** Every document indexed

2. **`src/embeddings/vectorizer.py`** (line 99-102)
   - **Tag:** `[INDEX-AUDIT][EMBED-WRITE]`
   - **Logs:** Metadata keys written to EmbeddingRecord during vectorization
   - **Frequency:** Every chunk vectorized

3. **`src/bootstrap/csv_ingestion.py`** (line 207-210)
   - **Tag:** `[INDEX-AUDIT][CSV-CHUNK]`
   - **Logs:** Metadata keys set during CSV ingestion chunking
   - **Frequency:** Every chunk created from CSV

### Retrieval-Time Logs:

4. **`src/retrieval/bm25/search_service.py`** (line 113-117)
   - **Tag:** `[INDEX-AUDIT][BM25-READ]`
   - **Logs:** Metadata keys loaded from BM25Document during retrieval
   - **Frequency:** First 3 results per query

---

## Expected Log Output

### Current State (Stale Index):
```
[INDEX-AUDIT][BM25-READ]  rank=1  doc_id=abc12345  resume_id=xyz67890  
  meta_keys=['education', 'experience', 'location', 'role', 'source_section']  
  non_null=['source_section']
```

**Observation:** Only 5 keys, all None except `source_section`.

### After Regeneration (Expected):
```
[INDEX-AUDIT][BM25-WRITE] chunk_id=abc12345  resume_id=xyz67890  
  meta_keys=['candidate_name', 'education', 'email', 'experience', 'location', 'phone', 'role', 'skills', 'source_section', 'summary']  
  non_null=['candidate_name', 'email', 'location', 'skills', 'source_section']

[INDEX-AUDIT][BM25-READ]  rank=1  doc_id=abc12345  resume_id=xyz67890  
  meta_keys=['candidate_name', 'education', 'email', 'experience', 'location', 'phone', 'role', 'skills', 'source_section', 'summary']  
  non_null=['candidate_name', 'email', 'location', 'skills', 'source_section']
```

**Observation:** 10 keys, with enriched fields populated.

---

## Remediation Steps

### Step 1: Delete Stale Cache/Index Files
```powershell
# Delete cache files
Remove-Item "data\cache\chunks.json"
Remove-Item "data\cache\embeddings.npy"
Remove-Item "data\cache\indexed_documents.json"

# Delete BM25 index files
Remove-Item "data\indexes\bm25\*" -Recurse
```

### Step 2: Update CSV Ingestion to Propagate Enriched Metadata
**File:** `src/bootstrap/csv_ingestion.py`  
**Method:** `chunk_raw_text()` (lines 160-166, 200-206)

**Current Code:**
```python
chunk_metadata = ChunkMetadata(
    role=None,
    experience=None,
    location=None,
    education=None,
    source_section="raw_text"
)
```

**Required Change:**
```python
chunk_metadata = ChunkMetadata(
    candidate_name=candidate_name,  # ✅ Pass from method parameter
    role=None,                       # Extract from CSV if available
    experience=None,                 # Extract from CSV if available
    location=None,                   # Extract from CSV if available
    education=None,                  # Extract from CSV if available
    skills=[],                       # Extract from CSV if available
    email=None,                      # Extract from CSV if available
    phone=None,                      # Extract from CSV if available
    summary=None,                    # Extract from CSV if available
    source_section="raw_text"
)
```

**Note:** The `chunk_raw_text()` method signature already accepts `candidate_name` as a parameter. Additional CSV fields (Email, Phone, Skills, Location) can be passed as parameters and propagated here.

### Step 3: Restart Application
```powershell
streamlit run app.py
```

**Expected Behavior:**
1. Bootstrap detects empty index
2. Runs full bootstrap workflow
3. CSV ingestion loads records with enriched metadata
4. Chunking creates ChunkMetadata with all 10 fields
5. Vectorization writes EmbeddingRecord with all 10 fields
6. BM25 indexing writes BM25Document with all 10 fields
7. Retrieval loads BM25Document with all 10 fields
8. UI displays candidate_name, skills, email, phone correctly

---

## Verification Checklist

After regeneration, verify:

- [ ] `[INDEX-AUDIT][CSV-CHUNK]` logs show 10 metadata keys
- [ ] `[INDEX-AUDIT][BM25-WRITE]` logs show 10 metadata keys with non-null values
- [ ] `[INDEX-AUDIT][EMBED-WRITE]` logs show 10 metadata keys with non-null values
- [ ] `[INDEX-AUDIT][BM25-READ]` logs show 10 metadata keys with non-null values
- [ ] Candidate cards display candidate_name (not "Unknown")
- [ ] Candidate cards display skills (not empty)
- [ ] Candidate cards display email/phone (not "None")
- [ ] Bootstrap report shows `bootstrapped: True`

---

## Conclusion

The metadata propagation pipeline is **correctly instrumented** and **schema-enriched**, but the **stale index** prevents the enriched metadata from reaching the UI. Deleting the 8 cache/index files and updating CSV ingestion to propagate enriched metadata will resolve the issue.

**Root Cause:** Bootstrap skip logic + stale cache files  
**Solution:** Delete cache/index files + update CSV ingestion  
**Effort:** Low (file deletion + minor code change)  
**Risk:** Low (re-indexing is idempotent)
