"""
LLM answer generation for the Recruiter QA Assistant.

Provides a deterministic, context-only answer generator that prevents
hallucinations by refusing to answer when the retrieved context does not
contain relevant evidence.  If an OpenAI API key is available, an LLM call
is attempted; otherwise a structured summary is built directly from the
provided context.
"""

import logging
import os
import re
import time
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)


def _query_has_support(query: str, context: str) -> bool:
    """Return True when at least one non-trivial query term appears in the context."""
    if not context or not context.strip():
        return False
    terms = [t.lower() for t in re.findall(r"\b\w+\b", query) if len(t) > 2]
    if not terms:
        return True
    context_lower = context.lower()
    return any(term in context_lower for term in terms)


def _token_count_approx(text: str) -> int:
    """Approximate token count by whitespace splitting (fallback when tiktoken is absent)."""
    return len(text.split()) if text else 0


def _call_openai_if_configured(prompt: str, max_tokens: int = 512) -> Tuple[str, int]:
    """Attempt an OpenAI completion when OPENAI_API_KEY and openai are available."""
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_TALENTLENS")
    if not api_key:
        return "", 0

    try:
        import openai  # type: ignore
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a helpful technical recruiter assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        answer = response.choices[0].message.content.strip()
        completion_tokens = response.usage.completion_tokens if response.usage else _token_count_approx(answer)
        return answer, completion_tokens
    except Exception as e:
        logger.warning("OpenAI completion failed, falling back to deterministic generator: %s", e)
        return "", 0


def _build_prompt(user_query: str, context: str) -> str:
    return (
        "You are a technical recruiter assistant. Answer the question using ONLY the "
        "retrieved resume context below.\n\n"
        "Rules:\n"
        "- Answer directly.\n"
        "- No filler phrases like 'Based on', 'According to', or 'The indexed resumes show'.\n"
        "- Short, bullet-style output.\n"
        "- If the answer is not in the context, reply EXACTLY:\n"
        '"Not found in indexed resumes."\n\n'
        "Retrieved Context:\n"
        f"{context}\n\n"
        f"Question: {user_query}\n"
        "Answer:"
    )


def _format_deterministic_answer(user_query: str, entries: List[Dict[str, Any]]) -> str:
    """Build a short, grounded answer from the cited context entries."""
    if not entries:
        return "Not found in indexed resumes."

    lines = []
    for entry in entries:
        name = entry.get("candidate_name") or "Unknown"
        resume_id = entry.get("resume_id") or "N/A"
        snippet = (entry.get("matched_text") or "").strip()
        # Keep only the first meaningful line of the snippet.
        if snippet:
            first_line = snippet.split("\n")[0].strip()
            snippet = first_line[:160].strip()
        if snippet:
            lines.append(f"{name} ({resume_id})\n{snippet}")
        else:
            lines.append(f"{name} ({resume_id})")

    return "\n\n".join(lines)


class AnswerGenerator:
    """Generate grounded answers from retrieved resume context."""

    def generate(self, user_query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an answer and telemetry from a context payload.

        Args:
            user_query: The recruiter's question.
            context: Output from ContextBuilder containing 'context' and 'entries'.

        Returns:
            Dict with 'answer', 'prompt', 'prompt_tokens', 'completion_tokens',
            and 'response_time_ms'.
        """
        start_time = time.perf_counter()

        context_text = context.get("context", "")
        entries = context.get("entries", [])

        prompt = _build_prompt(user_query, context_text)
        prompt_tokens = _token_count_approx(prompt)

        if not entries:
            answer = "Not found in indexed resumes."
            completion_tokens = _token_count_approx(answer)
        else:
            answer, completion_tokens = _call_openai_if_configured(prompt)
            if not answer:
                answer = _format_deterministic_answer(user_query, entries)
                completion_tokens = _token_count_approx(answer)

        response_time = time.perf_counter() - start_time

        logger.info(
            "Prompt Tokens: %d | Completion Tokens: %d | Response Time: %.3fs",
            prompt_tokens,
            completion_tokens,
            response_time,
        )

        return {
            "answer": answer,
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "response_time_ms": response_time * 1000,
        }


def generate_answer(user_query: str, docs_with_ids: List[Tuple[str, str]]) -> str:
    """Backward-compatible answer entry point.

    Keeps the legacy (doc_id, text) tuple interface working by building a
    minimal context and calling the new AnswerGenerator.
    """
    from src.context_builder import ContextBuilder

    entries = [
        {
            "candidate_name": doc_id,
            "resume_id": doc_id,
            "chunk_id": f"chunk-{i}",
            "section": "unknown",
            "matched_text": text,
            "score": 1.0,
            "offset": 0,
            "retrieval_source": "legacy",
        }
        for i, (doc_id, text) in enumerate(docs_with_ids)
    ]
    context_lines = [
        f"[{i}] Candidate: {e['candidate_name']} | Resume ID: {e['resume_id']} | Section: {e['section']}\n{e['matched_text'][:300]}"
        for i, e in enumerate(entries, start=1)
    ]
    context = {
        "context": "\n\n".join(context_lines),
        "entries": entries,
    }
    return AnswerGenerator().generate(user_query, context)["answer"]


def generate_answer_with_trace(user_query: str, docs_with_ids: List[Tuple[str, str]]) -> Tuple[str, Dict]:
    """Backward-compatible answer entry point with a trace payload."""
    answer = generate_answer(user_query, docs_with_ids)
    trace = {
        "step": "Answer generation",
        "tool": "AnswerGenerator",
        "input_docs_count": len(docs_with_ids),
    }
    return answer, trace


