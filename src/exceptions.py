"""
exceptions.py — Custom exception hierarchy for the RAG Agentic AI system.
"""

from __future__ import annotations


class RAGError(Exception):
    """Base class for all RAG pipeline errors."""


class RetrievalError(RAGError):
    """Raised when Pinecone similarity search fails."""


class GenerationError(RAGError):
    """Raised when the LLM call fails (timeout, auth, quota, etc.)."""
