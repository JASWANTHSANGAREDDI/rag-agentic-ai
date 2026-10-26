from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders.base_o365 import CHUNK_SIZE
import chunk

pdf_parth  = "data/Ebook-Agentic-AI.pdf" 


loader  =  PyPDFLoader(pdf_parth)
documents  = loader.load()
text_splitter  =  RecursiveCharacterTextSplitter(

    chunk_size =  1000,
    chunk_overlap =  200
) 

all_splits  =  text_splitter.split_documents(documents)


print("Loaded documents:", len(documents))
print("Created chunks:", len(all_splits))