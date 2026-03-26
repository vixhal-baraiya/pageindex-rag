import openai
from .node import PageNode

client = openai.OpenAI()


def _pick_child(query: str, node: PageNode) -> PageNode:
    options = "\n".join(
        f"{i + 1}. [{c.title}]: {c.summary}"
        for i, c in enumerate(node.children)
    )
    prompt = f"""You are navigating a document tree to find the answer to a question.

Current section: "{node.title}"
Question: {query}

Children of this section:
{options}

Which child section most likely contains the answer? Reply with only the number."""

    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
    )
    try:
        index = int(response.choices[0].message.content.strip()) - 1
        return node.children[index]
    except (ValueError, IndexError):
        return node.children[0]


def retrieve(query: str, root: PageNode) -> str:
    node = root
    while not node.is_leaf():
        if not node.children:
            break
        node = _pick_child(query, node)
    return node.content
