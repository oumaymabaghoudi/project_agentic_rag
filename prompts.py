from langchain_core.prompts import ChatPromptTemplate

router_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a routing classifier.

Choose:
- "vector" if the question can likely be answered by direct semantic retrieval
- "graph" if the question likely needs relational or multi-hop reasoning

Return only one word:
vector
or
graph
"""
    ),
    ("human", "Question: {question}")
])

triple_extraction_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Extract factual triples from the text.

Return ONLY valid JSON in this format:
[
  {{"subject": "...", "relation": "...", "object": "..."}}
]

Rules:
- Use short normalized relation names
- Keep only explicit facts
- If no factual triple exists, return []
"""
    ),
    ("human", "Text: {text}")
])

graph_query_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
From the question, extract:
1. important entities
2. relation intent

Return ONLY valid JSON in this format:
{{
  "entities": ["..."],
  "relation_hint": "..."
}}

If you are unsure, return:
{{
  "entities": [],
  "relation_hint": "unknown"
}}
"""
    ),
    ("human", "Question: {question}")
])

grader_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are grading whether the evidence is sufficient to answer the user's question.

Return only one word:
good
or
bad
"""
    ),
    (
        "human",
        """
Question:
{question}

Evidence:
{evidence}
"""
    )
])

rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Rewrite the user's question into a better retrieval query.

Rules:
- Keep the meaning
- Make it short
- Make it better for search
- Return only the rewritten query
"""
    ),
    (
        "human",
        """
Original question: {question}
Current query: {current_query}
"""
    )
])

answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Answer the user's question using ONLY the provided evidence.
If the evidence is insufficient, say that clearly.
Be concise and factual.
"""
    ),
    (
        "human",
        """
Question:
{question}

Evidence:
{evidence}
"""
    )
])