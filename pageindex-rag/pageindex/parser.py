import json
import openai
from .node import PageNode

client = openai.OpenAI()

SUBSECTION_THRESHOLD = 300  # words


def _segment(text: str) -> list:
    prompt = f"""Split the following text into logical sections.
Return a JSON object with a "sections" key. Each item has:
- "title": short title (5 words or less)
- "content": the text belonging to this section

Text:
{text[:8000]}"""

    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3000,
        response_format={"type": "json_object"},
    )
    parsed = json.loads(response.choices[0].message.content)
    return parsed.get("sections", [])


def parse_document(text: str) -> PageNode:
    root = PageNode(title="root", content="", summary="", depth=0)

    for item in _segment(text):
        title = item.get("title", "Section")
        content = item.get("content", "")

        node = PageNode(title=title, content="", summary="", depth=1)
        node.parent = root

        word_count = len(content.split())
        if word_count > SUBSECTION_THRESHOLD:
            subsections = _segment(content)
            if len(subsections) > 1:
                for sub in subsections:
                    child = PageNode(
                        title=sub.get("title", "Subsection"),
                        content=sub.get("content", ""),
                        summary="",
                        depth=2,
                    )
                    child.parent = node
                    node.children.append(child)
            else:
                node.content = content  # splitting gave nothing useful, keep as leaf
        else:
            node.content = content  # short enough to stay as a leaf

        root.children.append(node)

    return root
