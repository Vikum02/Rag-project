from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

def load_vectorstore():
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = Chroma(
        persist_directory="data/chroma_db",
        embedding_function=embeddings
    )
    return vectorstore

def retrieve_chunks(question: str, k: int = 3):
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search(question, k=k)
    return results

def build_prompt(question: str, chunks: list) -> str:
    context = "\n\n".join([
        f"[Source: page {doc.metadata.get('page', 'unknown')}]\n{doc.page_content}"
        for doc in chunks
    ])

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information to answer that."
Always mention which page your answer came from.

Context:
{context}

Question: {question}

Answer:"""

    return prompt

def ask(question: str) -> dict:
    print(f"\nQuestion: {question}")
    print("Retrieving relevant chunks...")
    
    chunks = retrieve_chunks(question, k=3)
    prompt = build_prompt(question, chunks)
    
    print("Sending to GPT...")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    answer = response.choices[0].message.content
    
    sources = list(set([
        f"Page {doc.metadata.get('page', 'unknown')}" 
        for doc in chunks
    ]))
    
    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }

if __name__ == "__main__":
    result = ask("what is this document about")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources: {', '.join(result['sources'])}")