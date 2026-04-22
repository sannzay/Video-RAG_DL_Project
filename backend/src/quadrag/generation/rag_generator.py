"""RAG generator — chat answers via OpenRouter.

Uses the OpenAI Python SDK pointed at OpenRouter's OpenAI-compatible
``/api/v1`` endpoint. The same ``openai`` package is already in our
requirements because Pixeltable uses it for Whisper + embeddings, so there's
no new dependency to manage.
"""

import re
import time
from typing import List, Optional, Tuple

from loguru import logger
from openai import OpenAI

from quadrag.config import get_settings
from quadrag.models import ChatResponse, IndexType, RetrievalResult

settings = get_settings()
logger = logger.bind(name="RAGGenerator")

# Matches bracketed/parenthesised timestamp citations in two forms:
#   * M:SS        ``[0:12]`` ``(2:30)`` with optional ``.f`` fractional seconds
#   * N seconds   ``[127s]`` ``[127.9s]`` ``(200.0s)``
#
# Both forms happen in practice: we tell the model to cite in [M:SS], but
# the retrieved-context block shows timestamps like "At 127.9s:", and the
# model often mirrors that style. We accept both to keep ``grounded=True``
# honest and the UI's timestamp chips consistent.
#
# We still require brackets/parens on purpose — bare "meet at 3:00" in
# conversational prose would be a false positive.
_TIMESTAMP_RE = re.compile(
    r"[\[\(]"
    r"(?:"
    r"(\d+):(\d{2})(?:\.\d+)?"          # group 1,2 — M:SS form
    r"|"
    r"(\d+(?:\.\d+)?)s"                  # group 3   — Ns / N.Ns form
    r")"
    r"[\]\)]"
)


def _extract_cited_timestamps(answer: str) -> List[float]:
    """Parse bracketed timestamp citations (``[M:SS]`` or ``[Ns]``) from an
    answer. Returns seconds as floats. Deliberately strict about the bracket
    form so conversational text like "meet at 3:00" doesn't match.
    """
    if not answer:
        return []
    seconds: List[float] = []
    for match in _TIMESTAMP_RE.finditer(answer):
        m_str, s_str, raw_seconds = match.group(1), match.group(2), match.group(3)
        if m_str and s_str:
            secs_part = int(s_str)
            if secs_part >= 60:
                continue  # malformed like "[1:99]" — skip, don't coerce
            seconds.append(float(int(m_str) * 60 + secs_part))
        elif raw_seconds is not None:
            seconds.append(float(raw_seconds))
    return seconds


def apply_citation_grounding(
    answer: str,
    retrieved: List[RetrievalResult],
    tolerance_sec: Optional[float] = None,
) -> Tuple[List[RetrievalResult], bool]:
    """Filter ``retrieved`` to only citations the answer actually referenced.

    Returns ``(citations, grounded)``:

    * Answer has no ``[M:SS]`` references → ``grounded=False``, return the
      unmodified retrieved list so the UI still has *something* to show.
    * Answer cites timestamps and at least one retrieved chunk lands within
      ``tolerance_sec`` of a cited value → ``grounded=True``, return only
      the matched chunks.
    * Answer cites timestamps but none match any retrieved chunk
      (hallucinated timestamps) → ``grounded=False``, return the original
      retrieved list rather than an empty citations array. Returning empty
      would falsely suggest the LLM anchored its answer; returning
      ``grounded=False`` with the raw retrieved set is the honest surface.
    """
    if tolerance_sec is None:
        tolerance_sec = settings.CITATION_TIMESTAMP_TOLERANCE_SEC

    cited = _extract_cited_timestamps(answer)
    if not cited:
        return list(retrieved), False

    matched: List[RetrievalResult] = []
    for result in retrieved:
        if any(abs(result.timestamp - ts) <= tolerance_sec for ts in cited):
            matched.append(result)

    if not matched:
        return list(retrieved), False
    return matched, True


class RAGGenerator:
    """Generates answers from retrieved multimodal context using OpenRouter.

    OpenRouter is OpenAI-SDK-compatible, so we instantiate ``openai.OpenAI``
    with a custom ``base_url``. Model choice is driven by ``settings.CHAT_MODEL``
    (default: ``meta-llama/llama-3.3-70b-instruct`` — same underlying model
    that Groq used to serve).
    """

    def __init__(self):
        """Initialize the RAG generator."""
        self.client = OpenAI(
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY,
        )
        self.model = settings.CHAT_MODEL
        logger.info(f"RAGGenerator initialized with OpenRouter model: {self.model}")

    def build_context_prompt(
        self,
        query: str,
        retrieved_results: List[RetrievalResult],
        domain_context: Optional[str] = None,
    ) -> str:
        """Build the context prompt from retrieved results.

        Args:
            query: User query
            retrieved_results: Retrieved results from indexes
            domain_context: Optional domain context

        Returns:
            Formatted prompt string
        """
        # Group results by source
        audio_results = [r for r in retrieved_results if r.source == IndexType.AUDIO]
        image_results = [r for r in retrieved_results if r.source == IndexType.IMAGE]
        description_results = [
            r for r in retrieved_results if r.source == IndexType.DESCRIPTION
        ]
        domain_results = [r for r in retrieved_results if r.source == IndexType.DOMAIN]

        # Build context sections
        context_parts = []

        # System instruction
        system_part = """You are a helpful AI assistant that answers questions about video content. 
You have access to information from multiple sources: audio transcripts, visual descriptions, 
and domain-specific observations from the video. Use this information to provide accurate, 
detailed answers with timestamp references when relevant."""

        if domain_context:
            system_part += f"\n\nDomain Context: {domain_context}"
            system_part += "\nPay special attention to the domain-specific observations when answering."

        context_parts.append(system_part)
        context_parts.append("\n\n=== VIDEO CONTENT INFORMATION ===\n")

        # Add audio context
        if audio_results:
            context_parts.append("\n--- Audio Transcripts ---")
            for i, result in enumerate(audio_results, 1):
                context_parts.append(
                    f"\n[{i}] At {result.timestamp:.1f}s: \"{result.content}\""
                )

        # Add visual context from descriptions
        if description_results:
            context_parts.append("\n\n--- Visual Descriptions ---")
            for i, result in enumerate(description_results, 1):
                context_parts.append(
                    f"\n[{i}] At {result.timestamp:.1f}s: {result.content}"
                )

        # Add domain-specific context
        if domain_results:
            context_parts.append("\n\n--- Domain-Specific Observations ---")
            for i, result in enumerate(domain_results, 1):
                context_parts.append(
                    f"\n[{i}] At {result.timestamp:.1f}s: {result.content}"
                )

        # Add image context (if any)
        if image_results:
            context_parts.append("\n\n--- Visual Matches ---")
            for i, result in enumerate(image_results, 1):
                context_parts.append(
                    f"\n[{i}] Similar visual content found at {result.timestamp:.1f}s"
                )

        context_parts.append(f"\n\n=== USER QUESTION ===\n{query}")
        context_parts.append(
            "\n\n=== INSTRUCTIONS ===\n"
            "Answer the user's question based on the provided video content.\n"
            "When you reference something that happened in the video, cite the "
            "timestamp in the form [M:SS] — for example [0:12] for 12 seconds "
            "in, or [2:30] for two minutes thirty seconds. Use the timestamps "
            "given above in the context blocks; don't invent new ones. If the "
            "information isn't in the provided context, say so clearly."
        )

        return "".join(context_parts)

    def generate_answer(
        self,
        query: str,
        retrieved_results: List[RetrievalResult],
        domain_context: Optional[str] = None,
    ) -> ChatResponse:
        """Generate an answer using Groq LLM.

        Args:
            query: User query
            retrieved_results: Retrieved results from indexes
            domain_context: Optional domain context

        Returns:
            ChatResponse with answer and citations
        """
        start_time = time.time()

        try:
            logger.info(f"Generating answer for query: '{query[:50]}...'")

            # Build the context prompt
            prompt = self.build_context_prompt(query, retrieved_results, domain_context)

            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=settings.GROQ_TEMPERATURE,
                max_tokens=settings.GROQ_MAX_TOKENS,
            )

            answer = response.choices[0].message.content
            processing_time = time.time() - start_time

            citations, grounded = apply_citation_grounding(answer, retrieved_results)
            logger.info(
                f"Generated answer in {processing_time:.2f}s "
                f"(grounded={grounded}, {len(citations)}/{len(retrieved_results)} citations)"
            )

            return ChatResponse(
                answer=answer,
                citations=citations,
                processing_time=processing_time,
                grounded=grounded,
            )

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            processing_time = time.time() - start_time

            return ChatResponse(
                answer=f"I apologize, but I encountered an error while generating the answer: {str(e)}",
                citations=retrieved_results,
                processing_time=processing_time,
                grounded=False,
            )

    def generate_streaming_answer(
        self,
        query: str,
        retrieved_results: List[RetrievalResult],
        domain_context: Optional[str] = None,
    ):
        """Generate a streaming answer using Groq LLM.

        Args:
            query: User query
            retrieved_results: Retrieved results from indexes
            domain_context: Optional domain context

        Yields:
            Chunks of the generated answer
        """
        try:
            logger.info(f"Generating streaming answer for query: '{query[:50]}...'")

            # Build the context prompt
            prompt = self.build_context_prompt(query, retrieved_results, domain_context)

            # Call Groq API with streaming
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=settings.GROQ_TEMPERATURE,
                max_tokens=settings.GROQ_MAX_TOKENS,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Error generating streaming answer: {e}")
            yield f"Error: {str(e)}"


# Global generator instance
_generator = None


def get_generator() -> RAGGenerator:
    """Get or create global generator instance.

    Returns:
        RAGGenerator instance
    """
    global _generator
    if _generator is None:
        _generator = RAGGenerator()
    return _generator


