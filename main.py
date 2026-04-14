import os
import sys
import time

# Fix working directory to always be the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_corpus
from graph_builder import build_app

def main():
    documents = load_corpus("data/corpus.txt")
    app, triples = build_app(documents)

    print("=== Extracted Triples ===")
    for triple in triples:
        print(triple)

    print("\n=== Ask a Question ===")
    question = input("Question: ").strip()
    start_time = time.time()
    result = app.invoke({
        "question": question,
        "current_query": question,
        "route": "vector",
        "retrieved_docs": [],
        "extracted_triples": triples,
        "graph_facts": [],
        "grade": "bad",
        "retry_count": 0,
        "final_answer": ""
    })
    end_time = time.time()
    latency = round(end_time-start_time, 2)
    print("\n=== Latency ===")
    print(latency, )

    print("\n=== Final Answer ===")
    print(result["final_answer"])

    print("\n=== Route Used ===")
    print(result["route"])

    print("\n=== Retrieved Docs ===")
    for doc in result["retrieved_docs"]:
        print("-", doc.page_content)

    print("\n=== Graph Facts ===")
    for fact in result["graph_facts"]:
        print("-", fact)

if __name__ == "__main__":
    main()