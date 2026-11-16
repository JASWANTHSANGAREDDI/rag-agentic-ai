"""
schemas.py — Data contracts for the entire RAG Agentic AI system.

PRODUCTION RULE:
    Never pass raw dicts between modules. Always use typed models.
    One source of truth — change here, it updates everywhere.

PYDANTIC V2 NOTES:
    - Use `model_config = ConfigDict(...)` instead of inner `class Config`.
    - Use `model_validator` / `field_validator` instead of `@validator`.
    - `List` from typing is replaced by the built-in `list` (Python 3.9+).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Model 1: RetrievedChunk
# ---------------------------------------------------------------------------
# WHAT: One piece of text returned from Pinecone after a similarity search.
# USED BY: retriever.py (produces), graph.py (consumes), app.py (serialises)
# ---------------------------------------------------------------------------

class RetrievedChunk(BaseModel):
    """A single chunk retrieved from the vector store."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "Agentic AI refers to systems that can autonomously...",
                "score": 0.92,
                "page": 12,
                "source": "Ebook-Agentic-AI.pdf",
            }
        }
    )

    content: str = Field(
        ...,
        min_length=1,
        description="The actual text content of the chunk from the PDF.",
    )
    score: float = Field(
        ...,
        ge=0.0,
        description="Similarity score from the vector store (higher = more relevant).",
    )
    page: int = Field(
        default=-1,
        ge=-1,  # -1 = page unknown
        description="Page number from the source PDF. -1 if unknown.",
    )
    source: str = Field(
        default="unknown",
        description="File name or path of the source document.",
    )


# ---------------------------------------------------------------------------
# Model 2: AgentState
# ---------------------------------------------------------------------------
# WHAT: The mutable state LangGraph passes between nodes.
# WHY TypedDict and NOT Pydantic:
#   LangGraph reads/writes state as plain dicts (state["key"]).
#   TypedDict gives static-analysis type safety without runtime overhead.
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """State envelope carried through the LangGraph agent pipeline."""

    question: str                    # The user's original question (immutable)
    context: list[RetrievedChunk]    # Chunks retrieved from Pinecone
    answer: str                      # The LLM-generated answer
    grade: str                       # "good" or "retry" — set by grade_node
    retry_count: int                 # How many times we've looped (guards infinite retry)
    top_k: int                       # Number of chunks to retrieve from vector store


# ---------------------------------------------------------------------------
# Model 3: ChatRequest
# ---------------------------------------------------------------------------
# WHAT: The JSON body the user POSTs to /chat.
# USED BY: app.py — FastAPI auto-validates + documents this.
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Incoming request from the user."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What is agentic AI?",
                "top_k": 5,
            }
        }
    )

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The user's question (1–1000 characters).",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve (1–20).",
    )


# ---------------------------------------------------------------------------
# Model 4: ChatResponse
# ---------------------------------------------------------------------------
# WHAT: The JSON body we return from POST /chat.
# USED BY: app.py — FastAPI auto-serialises this.
# ---------------------------------------------------------------------------

class ChatResponse(BaseModel):
    """Response returned to the caller."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": "Agentic AI refers to AI systems that act autonomously...",
                "sources": [
                    {
                        "content": "Agentic AI refers to systems that can autonomously...",
                        "score": 0.92,
                        "page": 12,
                        "source": "Ebook-Agentic-AI.pdf",
                    }
                ],
            }
        }
    )

    answer: str = Field(
        ...,
        description="The LLM-generated answer grounded in retrieved context.",
    )
    sources: list[RetrievedChunk] = Field(
        default_factory=list,
        description="The document chunks used to generate the answer.",
    )
