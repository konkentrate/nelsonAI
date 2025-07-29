import os
import discord
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from config import DISCORD_TOKEN, ALLOWED_CHANNEL_ID, MISTRAL_API_KEY
from memory import MessageMemory
from prompt import prompt_template
from search import duckduckgo_search

# Setup Discord
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# Define the chat model
model = init_chat_model("mistral-tiny", model_provider="mistralai", mistral_api_key=MISTRAL_API_KEY, temperature=0.7)
memory = MessageMemory()

@client.event
async def on_ready():
    print(f"[READY] Logged in as {client.user}")

@client.event
async def on_message(message):
    if message.author.bot or message.channel.id != ALLOWED_CHANNEL_ID:
        return

    try:
        async with message.channel.typing():
            user_id = str(message.author.id)
            short_term = memory.get_short_term_history()
            long_term = memory.get_relevant_long_term_history(
                message.content,
                user_id=user_id,
                distance_threshold=0.7,
                user_weight=0.5  # boost for same user
            )

            short_term_str = "\n".join(
                [f"{'User' if isinstance(m, HumanMessage) else 'Bot'}: {m.content}" for m in short_term]
            )
            long_term_str = "\n".join(
                [f"{'User' if isinstance(m, HumanMessage) else 'Bot'}: {m.content}" for m in long_term]
            )

            # DuckDuckGo search trigger
            internet_context = ""
            if "search:" in message.content.lower() or "look up" in message.content.lower():
                search_query = message.content.split("search:", 1)[-1].strip() if "search:" in message.content.lower() else message.content.split("look up", 1)[-1].strip()
                print(f"[DEBUG] Performing DuckDuckGo search for: {search_query}")
                search_results = duckduckgo_search(search_query)
                internet_context = f"\nInternet Search Results (DuckDuckGo):\n{search_results}\n"

            prompt = prompt_template.format(
                long_term=long_term_str + internet_context,
                short_term=short_term_str,
                query=message.content
            )

            print("[DEBUG] Prompt sent to AI:\n", prompt)

            response = model.invoke([HumanMessage(content=prompt)])
            # Now save the current user message and bot response
            memory.save_message(str(message.author.id), message.content, is_bot=False)
            memory.save_message(str(client.user.id), response.content, is_bot=True)
            await message.channel.send(response.content)

    except Exception as e:
        print("❌ Error:", e)
        await message.channel.send("⚠️ An error occurred while processing your message.")

client.run(DISCORD_TOKEN)
