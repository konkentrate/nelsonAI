import os
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
ALLOWED_CHANNEL_ID = int(os.getenv("ALLOWED_CHANNEL_ID"))