import sys
from pathlib import Path

# Add project root so imports work whether run from root or src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings 
from src.config import settings 

# 1. Load PDF
pdf_path = PROJECT_ROOT / "data" / "Ebook-Agentic-AI.pdf"
loader = PyPDFLoader(str(pdf_path))
documents = loader.load()

# 2. Text Splitting using settings
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP
) 
all_splits = text_splitter.split_documents(documents)

print("Loaded documents:", len(documents))
print("Created chunks:", len(all_splits)) 

# 3. Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name=settings.EMBEDDING_MODEL
) 

# 4. Quick Sanity Check
sample_vector = embeddings.embed_query("What is agentic AI?")
print(f"Model: {settings.EMBEDDING_MODEL}")
print(f"Generated Vector Dimension: {len(sample_vector)}")

# 5. Generate embeddings for every document chunk
chunk_texts = [doc.page_content for doc in all_splits]

print("Generating embeddings for document chunks...")

document_vectors = embeddings.embed_documents(chunk_texts)

print("Chunks embedded:", len(document_vectors))
print("Vector dimensions:", len(document_vectors[0]))

# 6. Store vectors in Pinecone
from langchain_pinecone import PineconeVectorStore

print("Storing vectors in Pinecone index:", settings.PINECONE_INDEX_NAME, "...")
vector_store = PineconeVectorStore.from_documents(
    documents=all_splits,
    embedding=embeddings,
    index_name=settings.PINECONE_INDEX_NAME,
    text_key="text",
)
print("Vectors stored successfully in Pinecone index:", settings.PINECONE_INDEX_NAME)
