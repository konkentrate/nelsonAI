import os
import json
import discord
from discord.ext import commands
import httpx
import time
from dotenv import load_dotenv
from memory import ConversationMemory
import sqlite3

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
ALLOWED_CHANNEL_ID = int(os.getenv("ALLOWED_CHANNEL_ID"))

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="/", intents=intents)

# Dictionary to store conversation history for each user
user_conversations = {}

# Initialize memory system for context search - still needed for stats
memory = ConversationMemory()
print("[DEBUG] Memory system initialized but RAG functionality commented out")

async def call_mistral(messages):
    """Call Mistral API with conversation messages"""
    print(f"[DEBUG] Calling Mistral API with {len(messages)} messages")

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistral-small",
        "messages": messages,
        "temperature": 0.7
    }

    print(f"[DEBUG] PROMPT TO MISTRAL:\n{json.dumps(messages, indent=2)[:300]}...")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.mistral.ai/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=30
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
        await channel.send("Hello! I am online with conversation history only (RAG disabled).")

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

    user_id = str(ctx.author.id)
    start_time = time.perf_counter()

    try:
        # Initialize conversation for this user if it doesn't exist
        if user_id not in user_conversations:
            user_conversations[user_id] = []

            # Add system message to the conversation
            user_conversations[user_id].append({
                "role": "system",
                "content": """You're a Discord AI assistant. 
- Answer ONLY the current question
- Be concise and helpful
- No prefixes like "Assistant:" in response"""
            })

        # Get conversation history for this user
        conversation = user_conversations[user_id].copy()

        # RAG functionality commented out
        """
        # Search for similar messages in memory to provide context
        similar_messages = memory.search_similar_messages(prompt, k=4, ignore_bot=False)
        print(f"[DEBUG] Found {len(similar_messages)} similar messages")

        # Get recent messages for additional context
        recent_messages = memory.get_recent_messages(5)
        print(f"[DEBUG] Retrieved {len(recent_messages)} recent messages")

        # Add context to the conversation
        if similar_messages or recent_messages:
            context_text = ""

            if similar_messages:
                context_text += "[RELEVANT CONTEXT]\n"
                for msg, similarity in similar_messages:
                    context_text += f"{msg}\n"

            if recent_messages:
                context_text += "\n[RECENT MESSAGES]\n"
                for msg in recent_messages:
                    context_text += f"{msg['author']} ({msg['role']}): {msg['content']}\n"

            # Add context as a system message
            conversation.append({
                "role": "system",
                "content": f"Context for your response:\n{context_text}"
            })
        """
        print("[DEBUG] RAG functionality disabled - using only conversation history")

        # Add user message to conversation
        conversation.append({
            "role": "user",
            "content": f"{ctx.author}: {prompt}"
        })

        # Send typing indicator while processing
        async with ctx.channel.typing():
            # Get response from Mistral
            reply = await call_mistral(conversation)

            if not reply.strip():
                raise ValueError("Empty response from Mistral")

            # Calculate elapsed time
            elapsed_time = (time.perf_counter() - start_time) * 1000
            print(f"[DEBUG] Got response from Mistral (took {elapsed_time:.2f} ms)")

            # Send the response
            await ctx.send(reply)

            # Add assistant response to conversation history
            user_conversations[user_id].append({
                "role": "user",
                "content": f"{ctx.author}: {prompt}"
            })
            user_conversations[user_id].append({
                "role": "assistant",
                "content": reply
            })

            # Keep conversation history to a reasonable size (last 10 messages)
            if len(user_conversations[user_id]) > 12:  # system + 10 messages (5 exchanges)
                # Keep the system message and the last 10 messages
                user_conversations[user_id] = [user_conversations[user_id][0]] + user_conversations[user_id][-10:]

            # Still store messages in memory for potential future use and stats
            memory.store_message(prompt, str(ctx.author), role="user")
            memory.store_message(reply, str(ctx.me), role="bot")
            print("[DEBUG] Messages stored in memory (but not used for context)")

    except Exception as e:
        print(f"[DEBUG] Error in nai_command: {str(e)}")
        await ctx.send("⚠️ An error occurred while processing the command.")

    print("---")

# Add a command to create a new conversation
@bot.command(name="new")
async def new_conversation(ctx):
    """Start a new conversation, forgetting previous context"""
    user_id = str(ctx.author.id)

    # Reset conversation for this user
    user_conversations[user_id] = [{
        "role": "system",
        "content": """You're a Discord AI assistant. 
- Answer ONLY the current question
- Be concise and helpful
- No prefixes like "Assistant:" in response"""
    }]

    await ctx.send("Started a new conversation! Your previous conversation history has been cleared.")

@bot.command(name="stats")
async def stats_command(ctx):
    """Show conversation statistics"""
    user_id = str(ctx.author.id)

    stats_text = "**Conversation Stats:**\n"

    # Get memory statistics
    with sqlite3.connect(memory.db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM messages")
        total_messages = cursor.fetchone()[0]

        cursor = conn.execute("SELECT COUNT(*) FROM messages WHERE role='user'")
        user_messages = cursor.fetchone()[0]

        cursor = conn.execute("SELECT COUNT(*) FROM messages WHERE role='bot'")
        bot_messages = cursor.fetchone()[0]

    stats_text += f"📊 Total messages in memory: {total_messages}\n"
    stats_text += f"👤 User messages: {user_messages}\n"
    stats_text += f"🤖 Bot messages: {bot_messages}\n"

    # Add conversation history stats
    if user_id in user_conversations:
        conv_len = len(user_conversations[user_id]) - 1  # Don't count system message
        stats_text += f"💬 Your current conversation length: {conv_len} messages\n"
    else:
        stats_text += "💬 No active conversation\n"

    stats_text += "\n*Note: RAG functionality is currently disabled*"

    await ctx.send(stats_text)

bot.run(DISCORD_TOKEN)
