import json
import os
import re
from typing import List, Dict, Any
from langchain_core.documents import Document


def load_corpus(file_path: str) -> List[Document]:
    documents = []

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    for i, line in enumerate(lines, start=1):
        documents.append(
            Document(
                page_content=line,
                metadata={"source": f"doc_{i}"}
            )
        )

    return documents


def safe_json_loads(text: str, default: Any) -> Any:
    """
    Safely parse JSON from LLM output.
    Handles markdown code fences like ```json ... ``` that Gemini often returns.
    """
    if not isinstance(text, str):
        return default

    # Strip markdown fences: ```json ... ``` or ``` ... ```
    cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except Exception:
        # Last resort: try to find a JSON object or array inside the text
        match = re.search(r"(\[.*\]|\{.*\})", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
        return default


def triples_to_text(triples: List[Dict[str, str]]) -> List[str]:
    results = []
    for triple in triples:
        subject = triple.get("subject", "")
        relation = triple.get("relation", "")
        obj = triple.get("object", "")
        results.append(f"{subject} --{relation}--> {obj}")
    return results


def save_triples(file_path: str, triples: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(triples, f, indent=2, ensure_ascii=False)


def load_triples(file_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(file_path):
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)