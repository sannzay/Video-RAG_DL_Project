"""RAG generator using Groq LLM."""

import time
from typing import List, Optional

from groq import Groq
from loguru import logger

from quadrag.config import get_settings
from quadrag.models import ChatResponse, IndexType, RetrievalResult

settings = get_settings()
logger = logger.bind(name="RAGGenerator")


class RAGGenerator:
    """Generates answers using retrieved context and Groq LLM."""

    def __init__(self):
        """Initialize the RAG generator."""
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = settings.GROQ_MODEL
        logger.info(f"RAGGenerator initialized with model: {self.model}")

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
            "Please answer the user's question based on the provided video content. "
            "Reference specific timestamps when mentioning information. "
            "If the information is not in the provided context, say so clearly."
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
                temperature=0.7,
                max_tokens=1024,
            )

            answer = response.choices[0].message.content
            processing_time = time.time() - start_time

            logger.info(f"Generated answer in {processing_time:.2f}s")

            return ChatResponse(
                answer=answer,
                citations=retrieved_results,
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            processing_time = time.time() - start_time
            
            return ChatResponse(
                answer=f"I apologize, but I encountered an error while generating the answer: {str(e)}",
                citations=retrieved_results,
                processing_time=processing_time,
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
                temperature=0.7,
                max_tokens=1024,
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


