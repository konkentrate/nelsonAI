import os
import json
import discord
from discord.ext import commands
import httpx

from memory import ConversationMemory
from config import DISCORD_TOKEN, ALLOWED_CHANNEL_ID
from mistral import call_mistral
from prompt import build_prompt  # <-- import the prompt builder

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="/", intents=intents)

# Initialize memory system once
memory = ConversationMemory()
print("[DEBUG] Bot memory system initialized")

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
    print(f"[DEBUG] Processing /nai command from {ctx.author}: {prompt}")
    try:
        # Get similar messages for context
        similar_messages = memory.search_similar_messages(prompt, ignore_bot=False)
        print(f"[DEBUG] Found {len(similar_messages)} similar messages")
        # Build prompt using prompt builder (no recent messages)
        full_prompt = build_prompt(prompt, similar_messages)
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
