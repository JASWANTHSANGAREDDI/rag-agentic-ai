import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from src.config import settings
from src.schemas import RetrievedChunk


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Initialize embeddings using HuggingFace.
    
    @lru_cache(maxsize=1) caches the model in RAM so it loads only ONCE,
    not on every single query.
    """
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
    )


@lru_cache(maxsize=1)
def get_vector_store() -> PineconeVectorStore:
    """Connect to the Pinecone vector store using cached embeddings."""
    embeddings = get_embeddings()
    return PineconeVectorStore(
        index_name=settings.PINECONE_INDEX_NAME,
        embedding=embeddings,
        text_key="text",
    )


def retrieve(query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
    """Retrieve relevant documents from Pinecone and pack them into RetrievedChunk models.
    
    This connects Pinecone's raw data directly to our Pydantic schema contract.
    """
    k = top_k if top_k is not None else settings.TOP_K
    vector_store = get_vector_store()

    # 1. Fetch raw matches from Pinecone: (Document, score)
    raw_results = vector_store.similarity_search_with_score(query, k=k)

    # 2. Pack each raw result into our clean RetrievedChunk schema
    chunks: List[RetrievedChunk] = []
    for doc, score in raw_results:
        chunk = RetrievedChunk(
            content=doc.page_content,
            score=round(float(score), 4),
            page=int(doc.metadata.get("page", -1)),
            source=str(doc.metadata.get("source", "unknown")),
        )
        chunks.append(chunk)

    return chunks


def get_retriever(top_k: Optional[int] = None):
    """Return a standard LangChain retriever interface if needed by LangGraph."""
    k = top_k if top_k is not None else settings.TOP_K
    vector_store = get_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": k})
