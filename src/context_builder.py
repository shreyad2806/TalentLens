"""
Context Builder for the Recruiter QA Assistant.

Converts hybrid retrieval results into a cited, de-duplicated context string
that an LLM can use to answer recruiter questions.  Each evidence snippet is
annotated with Candidate Name, Resume ID and the matched resume section to
prevent hallucinations and support verifiable answers.
"""

import logging
from typing import Any

from src.retrieval.hybrid.schema import HybridSearchResult

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Build a cited, de-duplicated context from hybrid retrieval results."""

    def build_context(
        self,
        hybrid_results: list[HybridSearchResult],
        top_k: int = 5,
        max_snippet_length: int = 400
    ) -> dict[str, Any]:
        """
        Build a context payload from a list of HybridSearchResult objects.

        Steps:
            1. Flatten all MatchedChunk evidence across results.
            2. De-duplicate by chunk_id (keep the highest-scoring evidence).
            3. Sort by score and truncate to top_k.
            4. Format a context string with citations.

        Args:
            hybrid_results: Results from the hybrid retrieval pipeline.
            top_k: Maximum number of unique chunks to include in the context.
            max_snippet_length: Maximum characters for each evidence snippet.

        Returns:
            Dictionary with keys:
                - context: formatted string with citations
                - entries: list of dicts with candidate_name, resume_id,
                  chunk_id, section, matched_text, score, offset, retrieval_source
                - retrieved_chunks: total number of unique chunks before top-k
                - context_length: character length of the formatted context
                - top_k: effective top-k used
        """
        entries: list[dict[str, Any]] = []
        seen_chunk_ids: set = set()

        for result in hybrid_results:
            # Use MatchedChunk evidence when available.
            if result.matched_chunks:
                for chunk in result.matched_chunks:
                    if not chunk.chunk_id or chunk.chunk_id in seen_chunk_ids:
                        continue
                    seen_chunk_ids.add(chunk.chunk_id)
                    text = (chunk.matched_text or "").strip()
                    if not text and result.metadata:
                        text = self._fallback_text(result.metadata, max_snippet_length)
                    entries.append({
                        "candidate_name": result.candidate_name,
                        "resume_id": result.resume_id,
                        "chunk_id": chunk.chunk_id,
                        "section": chunk.section or result.section or "unknown",
                        "matched_text": text,
                        "score": chunk.score,
                        "offset": chunk.offset,
                        "retrieval_source": chunk.retrieval_source,
                    })
            else:
                # Fall back to the top-level result fields.
                chunk_id = result.chunk_id
                if not chunk_id or chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)
                text = self._fallback_text(result.metadata, max_snippet_length)
                entries.append({
                    "candidate_name": result.candidate_name,
                    "resume_id": result.resume_id,
                    "chunk_id": chunk_id,
                    "section": result.section or "unknown",
                    "matched_text": text,
                    "score": result.rrf_score,
                    "offset": 0,
                    "retrieval_source": "hybrid",
                })

        # Sort by score descending and keep only the best chunk per candidate.
        entries.sort(key=lambda x: x["score"], reverse=True)

        unique_candidate_entries: list[dict[str, Any]] = []
        seen_resume_ids: set = set()
        for entry in entries:
            resume_id = entry["resume_id"]
            if resume_id in seen_resume_ids:
                continue
            seen_resume_ids.add(resume_id)
            unique_candidate_entries.append(entry)

        # Top-k refers to the number of unique candidates now.
        top_entries = unique_candidate_entries[:top_k]

        context_parts: list[str] = []
        for i, entry in enumerate(top_entries, start=1):
            citation = (
                f"[{i}] Candidate: {entry['candidate_name']} | "
                f"Resume ID: {entry['resume_id']} | "
                f"Section: {entry['section']} | "
                f"Source: {entry['retrieval_source']}"
            )
            snippet = entry["matched_text"][:max_snippet_length]
            context_parts.append(f"{citation}\n{snippet}")

        context = "\n\n".join(context_parts)

        result = {
            "context": context,
            "entries": top_entries,
            "retrieved_chunks": len(unique_candidate_entries),
            "context_length": len(context),
            "top_k": top_k,
        }

        logger.info(
            "Retrieved Chunks: %d | Context Length: %d chars | Top-K: %d",
            result["retrieved_chunks"],
            result["context_length"],
            result["top_k"],
        )

        return result

    @staticmethod
    def _fallback_text(metadata: dict[str, Any], max_length: int) -> str:
        """Return a fallback text snippet from result metadata."""
        text = metadata.get("text") or metadata.get("text_preview") or metadata.get("source_text") or ""
        return str(text)[:max_length].strip()
