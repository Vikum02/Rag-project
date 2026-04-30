from ragas import evaluate
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision
from ragas.llms import llm_factory
from openai import OpenAI
from datasets import Dataset
from query import retrieve_chunks, ask
from dotenv import load_dotenv
import os

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
llm = llm_factory("gpt-3.5-turbo", client=openai_client)

questions = [
    "what is this document about",
    "what are the main topics covered",
    "what conclusions does the document make",
    "who is the intended audience",
    "what are the key findings",
]

print("Running RAG pipeline for each question...")
data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

for q in questions:
    chunks = retrieve_chunks(q, k=3)
    result = ask(q)

    data["question"].append(q)
    data["answer"].append(result["answer"])
    data["contexts"].append([c.page_content for c in chunks])
    data["ground_truth"].append("")

dataset = Dataset.from_dict(data)

print("\nEvaluating with RAGAs...")
results = evaluate(
    dataset,
    metrics=[
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm),
        ContextPrecision(llm=llm)
    ]
)

print("\n=== RAGAs Evaluation Results ===")
print(f"Faithfulness     : {results['faithfulness']:.3f}")
print(f"Answer relevancy : {results['answer_relevancy']:.3f}")
print(f"Context precision: {results['context_precision']:.3f}")
print("\nScores are between 0 and 1. Higher is better.")
print("Screenshot these results — they go in your CV and README.")