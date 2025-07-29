import httpx
from config import MISTRAL_API_KEY

async def call_mistral(prompt):
    """Simplified function to call Mistral API with a single user message"""
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }

    # Send everything in a single user message
    messages = [
        {"role": "user", "content": prompt}
    ]

    payload = {
        "model": "mistral-medium",
        "messages": messages,
        "temperature": 0.7
    }

    print(f"[DEBUG] Sending request to Mistral API")
    print(f"[DEBUG] PROMPT TO MISTRAL:\n{prompt}...")

    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(
            "https://api.mistral.ai/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=15
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return content
