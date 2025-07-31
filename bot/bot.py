import os
import discord
from discord.ext import commands
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from config import DISCORD_TOKEN, ALLOWED_CHANNEL_ID, MISTRAL_API_KEY #, DEEPSEEK_API_KEY
from memory import MessageMemory
from prompt import prompt_template
from search import duckduckgo_search

# Setup Discord with commands
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Model configuration
MODEL_CONFIGS = {
    "mistral": {
        "name": "mistral-tiny",
        "provider": "mistralai",
        "api_key": MISTRAL_API_KEY
    },
    # "deepseek": {
    #     "name": "deepseek-chat",
    #     "provider": "deepseek",
    #     "api_key": DEEPSEEK_API_KEY
    # }
}

def get_model(model_name, api_key):
    if model_name == "mistral-tiny":
        return init_chat_model(
            model_name,
            model_provider="mistralai",
            api_key=api_key,
            temperature=0.7
        )
    # elif model_name == "deepseek-chat":
    #     return init_chat_model(
    #         model_name,
    #         model_provider="deepseek",
    #         api_key=api_key,
    #         temperature=0.7
    #     )
    else:
        raise ValueError("Unknown model name")

current_model_config = MODEL_CONFIGS["mistral"]
model = get_model(current_model_config["name"], current_model_config["api_key"])
memory = MessageMemory()

@bot.event
async def on_ready():
    print(f"[READY] Logged in as {bot.user}")

@bot.command(name="switch")
async def switch_model(ctx, model_name: str):
    """Switch between AI models. Usage: !switch mistral"""
    global model, current_model_config

    if model_name.lower() not in MODEL_CONFIGS:
        await ctx.send("Invalid model name. Use 'mistral'") # or 'deepseek'")
        return

    new_config = MODEL_CONFIGS[model_name.lower()]
    if new_config["api_key"] is None:
        await ctx.send(f"No API key found for {model_name}. Please set the appropriate environment variable.")
        return

    try:
        model = get_model(new_config["name"], new_config["api_key"])
        current_model_config = new_config
        await ctx.send(f"Successfully switched to {model_name} model!")
        print(f"[DEBUG] Switched to {model_name} model")
    except Exception as e:
        await ctx.send(f"Error switching model: {str(e)}")
        print(f"[ERROR] Failed to switch model: {e}")

@bot.event
async def on_message(message):
    await bot.process_commands(message)
    if message.author.bot or message.channel.id != ALLOWED_CHANNEL_ID:
        return
    if message.content.startswith('!'):
        return

    try:
        async with message.channel.typing():
            user_id = str(message.author.id)
            short_term = memory.get_short_term_history()
            long_term = memory.get_relevant_long_term_history(
                message.content,
                user_id=user_id,
                distance_threshold=0.7,
                user_weight=0.5
            )

            short_term_str = "\n".join(
                [f"{'User' if isinstance(m, HumanMessage) else 'Bot'}: {m.content}" for m in short_term]
            )
            long_term_str = "\n".join(
                [f"{'User' if isinstance(m, HumanMessage) else 'Bot'}: {m.content}" for m in long_term]
            )

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

            print(f"[DEBUG] Using model: {current_model_config['name']}")
            print("[DEBUG] Prompt sent to AI:\n", prompt)

            response = model.invoke([HumanMessage(content=prompt)])
            memory.save_message(str(message.author.id), message.content, is_bot=False, model=model)
            memory.save_message(str(bot.user.id), response.content, is_bot=True, model=model)
            await message.channel.send(response.content)

    except Exception as e:
        print("❌ Error:", e)
        await message.channel.send("⚠️ An error occurred while processing your message.")

bot.run(DISCORD_TOKEN)
