"""
schemas.py — Data contracts for the entire RAG Agentic AI system.

WHY THIS FILE EXISTS:
    Every module (retriever, graph, app) needs to agree on "what does the data look like."
    Without schemas, each module guesses — and guessing causes bugs.
    With schemas, there's ONE source of truth. Change it here, it updates everywhere.

PRODUCTION RULE:
    Never pass raw dicts between modules. Always use typed models.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Model 1: RetrievedChunk
# ---------------------------------------------------------------------------
# WHAT: One piece of text returned from Pinecone after a similarity search.
# USED BY: retriever.py (returns these), graph.py (reads these), app.py (sends these)
# ---------------------------------------------------------------------------

class RetrievedChunk(BaseModel):
    """A single chunk retrieved from the vector store."""

    content: str = Field(
        ...,                          # '...' means REQUIRED — cannot be empty
        description="The actual text content of the chunk from the PDF."
    )
    score: float = Field(
        ...,
        ge=0.0,                       # ge = greater than or equal to 0
        le=1.0,                       # le = less than or equal to 1
        description="Similarity score between 0.0 (no match) and 1.0 (perfect match)."
    )
    page: int = Field(
        default=-1,                   # default -1 means "page unknown"
        ge=-1,
        description="Page number from the source PDF. -1 if unknown."
    )
    source: str = Field(
        default="unknown",
        description="File name or path of the source document."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "content": "Agentic AI refers to systems that can autonomously...",
                "score": 0.92,
                "page": 12,
                "source": "Ebook-Agentic-AI.pdf"
            }
        }


# ---------------------------------------------------------------------------
# Model 2: AgentState
# ---------------------------------------------------------------------------
# WHAT: The internal state that LangGraph passes between nodes.
# WHY TypedDict and NOT Pydantic:
#     LangGraph requires dict-like access (state["question"]).
#     Pydantic uses attribute access (state.question).
#     TypedDict gives us type safety + dict behavior. Best of both worlds.
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """State carried through the LangGraph agent pipeline."""

    question: str                       # The user's original question
    context: List[RetrievedChunk]       # Chunks retrieved from Pinecone
    answer: str                         # The LLM's generated answer


# ---------------------------------------------------------------------------
# Model 3: ChatRequest
# ---------------------------------------------------------------------------
# WHAT: What the user sends to our POST /chat API endpoint.
# USED BY: app.py (FastAPI auto-validates this from the request body)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Incoming request from the user."""

    question: str = Field(
        ...,
        min_length=1,                  # reject empty strings
        max_length=1000,               # protect against abuse — no 10MB queries
        description="The user's question to ask about the Agentic AI ebook."
    )
    top_k: int = Field(
        default=5,                     # optional — defaults to 5 results
        ge=1,                          # minimum 1 chunk
        le=20,                         # maximum 20 chunks (protects against expensive queries)
        description="Number of document chunks to retrieve from the vector store."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is agentic AI?",
                "top_k": 5
            }
        }


# ---------------------------------------------------------------------------
# Model 4: ChatResponse
# ---------------------------------------------------------------------------
# WHAT: What our API sends back to the user.
# USED BY: app.py (FastAPI auto-serializes this to JSON)
# ---------------------------------------------------------------------------

class ChatResponse(BaseModel):
    """Response sent back to the user."""

    answer: str = Field(
        ...,
        description="The LLM-generated answer based on retrieved context."
    )
    sources: List[RetrievedChunk] = Field(
        default_factory=list,          # default is empty list, not None
        description="The document chunks used to generate the answer."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Agentic AI refers to AI systems that can act autonomously...",
                "sources": [
                    {
                        "content": "Agentic AI refers to systems that can autonomously...",
                        "score": 0.92,
                        "page": 12,
                        "source": "Ebook-Agentic-AI.pdf"
                    }
                ]
            }
        }
