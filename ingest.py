from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

def ingest_pdf(file_path: str):
    print(f"Loading {file_path}...")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages")

    for doc in pages:
        doc.metadata["source_file"] = os.path.basename(file_path)

    print("Chunking...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(pages)
    print(f"Created {len(chunks)} chunks")

    print("Embedding and storing in ChromaDB...")
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="data/chroma_db"
    )

    print(f"Done. Stored {len(chunks)} chunks from {os.path.basename(file_path)}")
    return vectorstore

if __name__ == "__main__":
    ingest_pdf("data/test.pdf")