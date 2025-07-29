def build_prompt(question, similar_messages):
    retrieved_chunk = "\n".join([msg for msg, _ in similar_messages])
    prompt = f"""
Context information is below.
---------------------
{retrieved_chunk}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {question}
Answer:
"""
    return prompt
