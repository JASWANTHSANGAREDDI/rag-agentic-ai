import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """All config lives here. One place to check, one place to change."""

    # OpenRouter API (OpenAI-compatible)
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    # LLM Model (OpenRouter supports OpenAI, Claude, Llama, DeepSeek)
    MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

    # Pinecone
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")

    # Embeddings via OpenRouter
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

    # Chunking
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Retrieval
    TOP_K = int(os.getenv("TOP_K", "5"))

    # Server
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))


settings = Settings()
