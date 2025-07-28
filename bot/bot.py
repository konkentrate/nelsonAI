import os
import discord
import httpx
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
ALLOWED_CHANNEL_ID = int(os.getenv("ALLOWED_CHANNEL_ID"))

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

async def call_mistral(prompt):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "mistral-small",  # Or "mistral-small"/"mistral-medium"
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(
            "https://api.mistral.ai/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=15
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

@client.event
async def on_ready():
    print(f"[READY] Logged in as {client.user}")
    channel = client.get_channel(ALLOWED_CHANNEL_ID)
    if channel:
        await channel.send("Hello! I am online.")

@client.event
async def on_message(message):
    if message.author.bot:
        return

    if message.channel.id != ALLOWED_CHANNEL_ID:
        return

    async with message.channel.typing():
        try:
            reply = await call_mistral(message.content)
            await message.channel.send(reply)
        except Exception as e:
            await message.channel.send("⚠️ Error communicating with Mistral.")
            print(f"Error: {e}")

client.run(DISCORD_TOKEN)
