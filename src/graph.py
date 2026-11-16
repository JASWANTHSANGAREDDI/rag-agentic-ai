"""
graph.py — LangGraph agentic RAG pipeline.

ARCHITECTURE:

    ┌─────────────┐     ┌──────────────┐     ┌────────────┐
    │ retrieve_node│────▶│ generate_node│────▶│ grade_node │
    └─────────────┘     └──────────────┘     └─────┬──────┘
           ▲                                        │
           │          low quality / hallucination   │ retry
           └────────────────────────────────────────┘
                                                    │ good  (or max retries hit)
                                                    ▼
                                                  [END]

PRODUCTION NOTES:
  - LLM client is a module-level singleton — created once, reused by all nodes.
    Previously it was re-created on every node call (up to 3× per request).
  - AgentState now carries `retry_count` so the router uses an exact counter
    instead of the fragile "count how many chunks we have" heuristic.
  - Grade is stored as a separate key `grade` in AgentState — not embedded
    as a string prefix inside `answer`. Cleaner, safer, easier to test.
  - All LLM calls are wrapped in try/except with specific exception types.
  - Logging uses structured %s formatting — compatible with log aggregators
    (Cloud Logging, Datadog, etc.) that parse format args separately.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import SecretStr

from src.config import settings
from src.exceptions import GenerationError, RetrievalError
from src.retriever import retrieve
from src.schemas import AgentState, RetrievedChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM singleton — ONE client shared by all nodes, all requests.
# Creating ChatOpenAI is cheap (no network call), but doing it 3× per
# request wastes object construction and makes the call chain harder to trace.
# ---------------------------------------------------------------------------

def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=SecretStr(settings.openrouter_api_key),
        model=settings.model_name,
        temperature=settings.temperature,
        timeout=30,          # never wait more than 30 s for a token
        max_retries=2,       # langchain built-in HTTP retry on 5xx/timeout
    )

_llm: ChatOpenAI = _build_llm()


# ---------------------------------------------------------------------------
# Query expansion — rewrite short/vague queries for better retrieval
# ---------------------------------------------------------------------------

_QUERY_EXPAND_PROMPT = """\
You are a search query optimizer for a document about Agentic AI.
Given the user's question, rewrite it as a richer search query that will
match relevant passages more effectively in a vector similarity search.

Rules:
- Expand abbreviations and add related terms and synonyms
- Include key phrases likely to appear near the answer in the document
- If the question references a section name, chapter title, or topic heading,
  include that name verbatim AND add related subtopics
- Keep it as a single paragraph (2-4 sentences max)
- Do NOT answer the question — only rewrite it for better document retrieval

User question: {question}

Rewritten search query:"""

_QUERY_REPHRASE_PROMPT = """\
You are a search query optimizer. The previous search for this question
returned poor or irrelevant results. Rephrase the question using different
keywords, synonyms, and angles to find better matching passages in a
document about Agentic AI.

Original question: {question}
This is retry attempt #{attempt}. Use substantially different phrasing.

Output ONLY the new search query (2-4 sentences), nothing else:"""


def _expand_query(question: str, retry_count: int) -> str:
    """Use the LLM to rewrite the user's question for better retrieval."""
    try:
        if retry_count == 0:
            prompt = _QUERY_EXPAND_PROMPT.format(question=question)
        else:
            prompt = _QUERY_REPHRASE_PROMPT.format(
                question=question, attempt=retry_count
            )

        response = _llm.invoke([HumanMessage(content=prompt)])
        expanded = response.content
        result = expanded.strip() if isinstance(expanded, str) else str(expanded).strip()

        if result and len(result) > 5:
            logger.info("_expand_query | original=%.60s | expanded=%.100s", question, result)
            return result

        logger.warning("_expand_query | expansion too short, using original query")
        return question
    except Exception as exc:
        # Query expansion is best-effort — fall back to the original query
        logger.warning("_expand_query failed: %s — using original query", exc)
        return question


# ---------------------------------------------------------------------------
# Node 1: retrieve_node
# ---------------------------------------------------------------------------

def retrieve_node(state: AgentState) -> AgentState:
    """
    Fetch relevant chunks from Pinecone for the current question.

    On first attempt: expands the query using the LLM for better embedding match.
    On retries: rephrases the query with different keywords to find new passages.

    Reads  : state["question"], state["top_k"], state["retry_count"]
    Writes : state["context"]
    """
    question = state["question"]
    retry_count = state.get("retry_count", 0)
    top_k = state.get("top_k", settings.top_k)

    logger.info("retrieve_node | question=%.80s | top_k=%d | attempt=%d",
                question, top_k, retry_count + 1)

    # Expand / rephrase the query for better vector similarity matching
    search_query = _expand_query(question, retry_count)

    try:
        chunks = retrieve(search_query, top_k=top_k)
    except RetrievalError as exc:
        # Surface the error — let the graph framework handle propagation.
        logger.error("retrieve_node failed: %s", exc)
        raise

    # ----- Fix: deduplicate chunks across retries -----
    # On retry, merge newly retrieved chunks with the previous attempt's
    # chunks so the LLM gets a richer, broader context instead of
    # discarding potentially useful earlier results.
    if retry_count > 0 and state.get("context"):
        existing_contents = {c.content for c in state["context"]}
        merged = list(state["context"])
        for chunk in chunks:
            if chunk.content not in existing_contents:
                merged.append(chunk)
                existing_contents.add(chunk.content)
        # Sort by relevance and keep the best (up to 2× top_k)
        merged.sort(key=lambda c: c.score, reverse=True)
        chunks = merged[: top_k * 2]
        logger.info("retrieve_node | merged with previous chunks → %d unique chunks", len(chunks))

    if not chunks:
        logger.warning("retrieve_node | no chunks found — will produce fallback answer")

    logger.info("retrieve_node | fetched %d chunks (top score: %.4f)",
                len(chunks), chunks[0].score if chunks else 0)
    return {
        "question": question,
        "context": chunks,
        "answer": "",
        "grade": "",
        "retry_count": retry_count,
        "top_k": top_k,
    }


# ---------------------------------------------------------------------------
# Node 2: generate_node
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert assistant on Agentic AI. Answer the user's question using ONLY
the context passages provided below. Be concise, accurate, and cite page numbers
when available (e.g. "According to page 12, ...").

If the context does not contain enough information to answer the question
confidently, respond with exactly:
  "I don't have enough context to answer that question."

Do NOT invent or infer any information that is not explicitly in the context.
"""


def generate_node(state: AgentState) -> AgentState:
    """
    Call the LLM with retrieved context and produce an answer.

    Reads  : state["question"],state["context"]
    Writes : state["answer"]
    """
    question = state["question"]
    chunks: list[RetrievedChunk] = state["context"]

    # Short-circuit: if no chunks were retrieved, return a clear fallback
    # instead of wasting an LLM call that will inevitably say the same thing.
    if not chunks:
        logger.warning("generate_node | 0 context chunks — returning fallback answer")
        return {
            "question": question,
            "context": chunks,
            "answer": "I don't have enough context to answer that question.",
            "grade": "retry",   # signal the grader to retry with rephrased query
            "retry_count": state.get("retry_count", 0),
            "top_k": state.get("top_k", settings.top_k),
        }

    context_text = "\n\n".join(
        f"[Page {c.page} | Score {c.score:.2f}]\n{c.content}"
        for c in chunks
    )

    logger.info("generate_node | context_chunks=%d", len(chunks))

    try:
        response = _llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context_text}\n\nQuestion: {question}"),
        ])
        content = response.content
        answer = content.strip() if isinstance(content, str) else str(content).strip()
    except Exception as exc:
        logger.error("generate_node LLM call failed: %s", exc, exc_info=True)
        raise GenerationError(f"LLM generation failed: {exc}") from exc

    logger.info("generate_node | answer_len=%d chars", len(answer))
    return {
        "question": question,
        "context": chunks,
        "answer": answer,
        "grade": "",
        "retry_count": state.get("retry_count", 0),
        "top_k": state.get("top_k", settings.top_k),
    }


# ---------------------------------------------------------------------------
# Node 3: grade_node — the AGENTIC decision point
# ---------------------------------------------------------------------------

_GRADE_PROMPT = """\
You are a strict answer-quality evaluator.

Given a QUESTION, CONTEXT passages, and an ANSWER, decide:

  "good"  — The answer is factually grounded in the context, responsive,
             and not vague or empty.
  "retry" — The answer says it doesn't have enough context, is off-topic,
             is empty, or clearly contains facts not found in the context.

Reply with ONLY the single word "good" or "retry". No punctuation. No explanation.
"""


def grade_node(state: AgentState) -> AgentState:
    """
    Self-evaluate the generated answer. Store the grade in state["grade"].

    Reads  : state["question"], state["context"], state["answer"]
    Writes : state["grade"]  ("good" | "retry")

    On grading failure: defaults to "good" (non-fatal — don't loop forever).
    """
    question = state["question"]
    chunks: list[RetrievedChunk] = state["context"]
    answer = state["answer"]

    # If generate_node already set a grade (e.g. "retry" on empty context),
    # skip the LLM grading call and just pass it through.
    if state.get("grade") in ("good", "retry"):
        logger.info("grade_node | grade already set to %r — skipping LLM evaluation", state["grade"])
        return {
            "question": question,
            "context": chunks,
            "answer": answer,
            "grade": state["grade"],
            "retry_count": state.get("retry_count", 0),
            "top_k": state.get("top_k", settings.top_k),
        }

    context_text = "\n\n".join(c.content for c in chunks)

    logger.info("grade_node | evaluating answer (retry_count=%d)", state.get("retry_count", 0))

    try:
        grade_response = _llm.invoke([
            SystemMessage(content=_GRADE_PROMPT),
            HumanMessage(
                content=(
                    f"QUESTION: {question}\n\n"
                    f"CONTEXT:\n{context_text}\n\n"
                    f"ANSWER: {answer}"
                )
            ),
        ])
        raw_grade = grade_response.content
        grade_str = raw_grade.strip().lower() if isinstance(raw_grade, str) else str(raw_grade).strip().lower()
        grade = grade_str if grade_str in ("good", "retry") else "good"
        if grade_str not in ("good", "retry"):
            logger.warning("grade_node received unexpected grade %r — defaulting to 'good'", grade_str)
    except Exception as exc:
        # Grading is advisory — if it fails, accept the answer rather than crashing.
        logger.error("grade_node LLM call failed: %s — defaulting to 'good'", exc)
        grade = "good"

    logger.info("grade_node | grade=%s", grade)
    return {
        "question": question,
        "context": chunks,
        "answer": answer,
        "grade": grade,
        "retry_count": state.get("retry_count", 0),
        "top_k": state.get("top_k", settings.top_k),
    }


# ---------------------------------------------------------------------------
# Conditional router
# ---------------------------------------------------------------------------

def route_after_grade(state: AgentState) -> Literal["retrieve_node", "__end__"]:
    """
    Decide whether to loop back for another retrieval or accept the answer.

    Uses `state["retry_count"]` as an exact counter — not a chunk-count
    heuristic. This is reliable regardless of how many chunks are returned.
    """
    grade = state.get("grade", "good")
    retry_count = state.get("retry_count", 0)

    if grade == "retry" and retry_count < settings.max_retries:
        logger.info("route_after_grade | RETRY (attempt %d/%d)", retry_count + 1, settings.max_retries)
        return "retrieve_node"

    if grade == "retry":
        logger.warning(
            "route_after_grade | max retries (%d) reached — accepting answer as-is",
            settings.max_retries,
        )

    return "__end__"


def increment_retry(state: AgentState) -> AgentState:
    """
    Increment the retry counter before looping back to retrieve_node.
    This is a micro-node inserted on the retry edge so the counter stays accurate.
    """
    return {**state, "retry_count": state.get("retry_count", 0) + 1}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    """
    Wire all nodes into a compiled LangGraph StateGraph.

    Returns a compiled runnable: call `.invoke({"question": ..., ...})`.
    """
    graph = StateGraph(AgentState)

    graph.add_node("retrieve_node",   retrieve_node)
    graph.add_node("generate_node",   generate_node)
    graph.add_node("grade_node",      grade_node)
    graph.add_node("increment_retry", increment_retry)

    graph.set_entry_point("retrieve_node")
    graph.add_edge("retrieve_node",   "generate_node")
    graph.add_edge("generate_node",   "grade_node")
    graph.add_edge("increment_retry", "retrieve_node")

    graph.add_conditional_edges(
        "grade_node",
        route_after_grade,
        {
            "retrieve_node": "increment_retry",   # bump counter, then retry
            "__end__": END,
        },
    )

    return graph.compile()


# ---------------------------------------------------------------------------
# Module-level compiled graph singleton — import and invoke in app.py
# ---------------------------------------------------------------------------

rag_graph = build_graph()


# ---------------------------------------------------------------------------
# Public helper — used in app.py to produce the final user-facing answer
# ---------------------------------------------------------------------------

def get_final_answer(state: AgentState) -> str:
    """
    Extract the clean answer from the final AgentState.

    With the new design, `state["answer"]` is already clean (no prefix).
    This function exists as the single extraction point so app.py doesn't
    reach into state directly — easier to change internals later.
    """
    return state.get("answer", "").strip()

