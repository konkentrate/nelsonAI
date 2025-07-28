import os
import json
import discord
from discord.ext import commands
import httpx
from dotenv import load_dotenv
from memory import ConversationMemory

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
ALLOWED_CHANNEL_ID = int(os.getenv("ALLOWED_CHANNEL_ID"))

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="/", intents=intents)

# Initialize memory system once
memory = ConversationMemory()
print("[DEBUG] Bot memory system initialized")

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
        "model": "mistral-small",
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

@bot.event
async def on_ready():
    print(f"[READY] Logged in as {bot.user}")
    channel = bot.get_channel(ALLOWED_CHANNEL_ID)
    if channel:
        await channel.send("Hello! I am online.")

@bot.event
async def on_message(message):
    if message.author.bot or message.channel.id != ALLOWED_CHANNEL_ID:
        return
    await bot.process_commands(message)

@bot.command(name="nai")
async def nai_command(ctx, *, prompt: str = None):
    if ctx.channel.id != ALLOWED_CHANNEL_ID:
        return

    if not prompt:
        await ctx.send("⚠️ Please provide a prompt after /nai")
        return

    print(f"[DEBUG] Processing /nai command from {ctx.author}: {prompt[:50]}...")

    try:
        # Get similar messages for context
        similar_messages = memory.search_similar_messages(prompt, ignore_bot=False)
        print(f"[DEBUG] Found {len(similar_messages)} similar messages")

        # Get recent messages for additional context
        recent_messages = memory.get_recent_messages(10)
        print(f"[DEBUG] Retrieved {len(recent_messages)} recent messages")

        # Build a single optimized prompt with instructions and context
        full_prompt = f"""You're a Discord AI assistant. Answer ONLY the current question below.

[INSTRUCTIONS]
- Answer only the current question
- Use past messages for context only
- Be concise and helpful
- No prefixes like "Assistant:" in response

[RELEVANT MESSAGES]
"""
        # Add similar messages
        if similar_messages:
            for msg, similarity in similar_messages:
                full_prompt += f"{msg}\n"

        # Add recent messages
        if recent_messages:
            full_prompt += "\n[RECENT MESSAGES]\n"
            for msg in recent_messages:
                full_prompt += f"{msg['author']} ({msg['role']}): {msg['content']}\n"

        # Add current question
        full_prompt += f"\n[CURRENT QUESTION]\n{ctx.author}: {prompt}\n"

        reply = None
        async with ctx.channel.typing():
            try:
                reply = await call_mistral(full_prompt)
                if not reply.strip():
                    raise ValueError("Empty response from Mistral")
                print("[DEBUG] Got response from Mistral")
                await ctx.send(reply)
            except Exception as e:
                await ctx.send("⚠️ Error communicating with Mistral.")
                print(f"[DEBUG] Error calling Mistral: {str(e)}")

        # Store messages in memory after sending the response
        memory.store_message(prompt, str(ctx.author), role="user")
        print("[DEBUG] User message stored in memory")

        if reply:
            memory.store_message(reply, str(ctx.me), role="bot")
            print("[DEBUG] Bot reply stored in memory")

    except Exception as e:
        print(f"[DEBUG] Error in nai_command: {str(e)}")
        await ctx.send("⚠️ An error occurred while processing the command.")

    print("---")

bot.run(DISCORD_TOKEN)
