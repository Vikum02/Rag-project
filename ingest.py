from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("data/test.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(pages)

print(f"Total pages: {len(pages)}")
print(f"Total chunks: {len(chunks)}")
print("\nFirst chunk:")
print(chunks[0].page_content)
print("\nSecond chunk:")
print(chunks[1].page_content)