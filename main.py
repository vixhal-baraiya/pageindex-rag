import os
from pageindex.parser import parse_document
from pageindex.indexer import build_summaries
from pageindex.retriever import retrieve
from pageindex import storage
import openai

client = openai.OpenAI()
INDEX_PATH = "index.json"


def build_index(doc_path: str):
    print("Parsing document...")
    text = open(doc_path).read()
    tree = parse_document(text)

    print("Building summaries (this makes LLM calls)...")
    build_summaries(tree)

    print(f"Saving index to {INDEX_PATH}")
    storage.save(tree, INDEX_PATH)
    return tree


def ask(query: str) -> str:
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("Index not found. Run build_index() first.")

    tree = storage.load(INDEX_PATH)
    context = retrieve(query, tree)

    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=[{
            "role": "user",
            "content": f"Answer using only the context below.\n\nContext:\n{context}\n\nQuestion: {query}"
        }],
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    # First time: build the index
    build_index("document.md")

    # Then query it
    print(ask("Your Question"))
