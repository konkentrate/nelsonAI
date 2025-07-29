from duckduckgo_search import DDGS

def duckduckgo_search(query, max_results=5):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(r.get("body", "") or r.get("title", ""))
    return "\n".join(results) if results else "No relevant results found."

