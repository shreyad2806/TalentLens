"""
Vectorizer module - Chunk to EmbeddingRecord conversion.

This module provides the Vectorizer class that converts Chunk objects into
EmbeddingRecord objects. Each Chunk is converted to exactly one EmbeddingRecord
with its corresponding embedding vector.

The vectorizer does NOT concatenate chunks - one Chunk becomes one Vector.
"""


from .cache import get_embedding_cache
from .model_loader import get_model_loader
from .schema import EmbeddingRecord


class Vectorizer:
    """
    Vectorizer for converting Chunk objects to EmbeddingRecord objects.
    
    This class handles the conversion of Chunk objects into EmbeddingRecord objects
    by generating embedding vectors for each chunk. The vectorizer uses the
    singleton model loader and embedding cache for efficient processing.
    
    The vectorizer follows the principle: One Chunk → One Vector.
    Chunks are never concatenated; each chunk gets its own embedding.
    """
    
    def __init__(self):
        """
        Initialize the vectorizer.
        """
        self.model_loader = get_model_loader()
        self.cache = get_embedding_cache()
    
    def _build_record(self, chunk, vector: list[float]) -> EmbeddingRecord:
        """Build an EmbeddingRecord from a chunk and its vector."""
        embedding_record = EmbeddingRecord(
            chunk_id=str(chunk.chunk_id),
            section=chunk.section,
            text=chunk.text,
            vector=vector,
            vector_dimension=len(vector),
            model_name=self.model_loader.get_model_name(),
            resume_metadata=chunk.resume_metadata
        )

        m = embedding_record.resume_metadata
        print(f"[META-WRITE][EmbeddingRecord] chunk_id={str(chunk.chunk_id)[:8]}  resume_id={m.resume_id[:8]}  candidate_name={m.candidate_name}  skills_count={len(m.skills)}  location={m.location}  experience={m.experience_years}  role={m.role}")

        return embedding_record

    def vectorize_chunk(self, chunk) -> EmbeddingRecord:
        """
        Convert a single Chunk object to an EmbeddingRecord.
        
        This method:
        1. Checks the cache for existing embedding
        2. If not cached, generates new embedding
        3. Creates EmbeddingRecord with all metadata
        4. Caches the result
        
        Args:
            chunk: Chunk object to vectorize
            
        Returns:
            EmbeddingRecord with the chunk's embedding vector
        """
        # Check cache first
        cached_embedding = self.cache.get(chunk.text)
        
        if cached_embedding is not None:
            # Use cached embedding
            vector = cached_embedding
        else:
            # Generate new embedding
            model = self.model_loader.get_model()
            vector = model.encode(chunk.text, show_progress_bar=False).tolist()
            
            # Cache the result
            self.cache.set(chunk.text, vector)
        
        return self._build_record(chunk, vector)
    
    def vectorize_chunks(self, chunks: list, batch_size: int = 32, use_cache: bool = True) -> list[EmbeddingRecord]:
        """
        Convert multiple Chunk objects to EmbeddingRecord objects efficiently.
        
        This method uses batched model inference and a persistent cache to avoid
        redundant embedding of unchanged chunks.
        
        Args:
            chunks: List of Chunk objects to vectorize
            batch_size: Number of chunks to encode in a single model forward pass
            use_cache: Whether to read/write the persistent embedding cache
            
        Returns:
            List of EmbeddingRecord objects
        """
        if not chunks:
            return []

        if not use_cache:
            # Fast path for transient queries: never touches the persistent cache
            model = self.model_loader.get_model()
            texts = [c.text for c in chunks]
            encoded = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            return [self._build_record(chunk, vector.tolist()) for chunk, vector in zip(chunks, encoded)]

        # Separate cached and missing chunks
        cached_entries: list[tuple] = []
        missing_chunks: list = []

        for chunk in chunks:
            cached = self.cache.get(chunk.text)
            if cached is not None:
                cached_entries.append((chunk, cached))
            else:
                missing_chunks.append(chunk)

        # Batch encode missing chunks
        if missing_chunks:
            model = self.model_loader.get_model()
            texts = [c.text for c in missing_chunks]
            encoded = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            for chunk, vector in zip(missing_chunks, encoded.tolist()):
                self.cache.set(chunk.text, vector)

        # Re-assemble records in input order
        records: list[EmbeddingRecord] = []
        for chunk in chunks:
            cached = self.cache.get(chunk.text)
            records.append(self._build_record(chunk, cached))

        return records

    def save_cache(self) -> None:
        """Persist the embedding cache to disk."""
        self.cache.save()

    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return self.cache.get_stats()
