"""
ingestion.py — PDF → Chunks → Embeddings → Pinecone.

Run this ONCE to populate the vector store:
    python src/ingestion.py

This is NOT called at runtime — it's a one-time data pipeline.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from src.config import settings


def main():
    # 1. Load PDF
    pdf_path = PROJECT_ROOT / "data" / "Ebook-Agentic-AI.pdf"
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    print(f"  Loaded {len(documents)} pages")

    # 2. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    all_splits = text_splitter.split_documents(documents)
    print(f"  Created {len(all_splits)} chunks (size={settings.chunk_size}, overlap={settings.chunk_overlap})")

    # 3. Load embedding model
    print(f"Loading embedding model: {settings.embedding_model}")
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    # 4. Generate embeddings and store in Pinecone
    print(f"Generating embeddings and storing in Pinecone index: {settings.pinecone_index_name}...")
    vector_store = PineconeVectorStore.from_documents(
        documents=all_splits,
        embedding=embeddings,
        index_name=settings.pinecone_index_name,
        pinecone_api_key=settings.pinecone_api_key,
        text_key="text",
    )
    print(f"✅ Done! {len(all_splits)} chunks stored in '{settings.pinecone_index_name}'")


if __name__ == "__main__":
    main()
