# nelsonAI

A smart, memory-augmented **Discord chatbot** built using LangChain, SQLite, and the Mistral API.  
nelsonAI supports **long-term and short-term memory** per user, allowing for fluid, contextual conversations tailored to individual preferences.

> Project is a **work in progress** – expect frequent updates as it becomes smarter and more flexible!

---

## Features

- **Retrieval-Augmented Generation (RAG)** via LangChain
- **Memory support**:
  - Long-term memory (persisted in SQLite)
  - Short-term memory for ongoing chat sessions
  - User-specific preference memory
- Powered by **Mistral LLM** for generating high-quality responses
- Easy integration with **Discord** using the Discord API
- Lightweight and minimal dependencies – quick setup

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/konkentrate/nelsonAI.git
   cd nelsonAI
   ```

2. **Install dependencies**:
   Make sure you have Python 3.11 installed. (there are some unupdated libraries, so YMMV)
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:  
   Set up your `.env` file (or however you handle your secrets) with:
   ```
   DISCORD_TOKEN=your_discord_bot_token
   MISTRAL_API_KEY=your_mistral_api_key
   ```

4. **Run the bot**:
   ```bash
   python bot.py
   ```

> A `start.sh` script is coming soon for easier startup.

---

## Usage

Once running, invite the bot to your Discord server. It will respond in channels or DMs based on context and user memory. You can **train** it with specific messages or documents, and it will **remember your preferences**.

---

## Tech Stack

- **Python 3.10+**
- [LangChain](https://github.com/langchain-ai/langchain) – for chaining LLM calls with memory + RAG
- [Discord API](https://discordpy.readthedocs.io/en/stable/)
- [Mistral API](https://mistral.ai/) – fast, efficient open-weight LLMs
- **SQLite** – lightweight local storage for persistent memory

---

## Roadmap

- [ ] Add `start.sh` script
- [ ] Improve context handling across multi-user channels
- [ ] Add web UI for training & logs
- [ ] Pluggable memory backends (e.g., Postgres, Redis)
- [ ] Plugin system for custom commands
- [ ] Smarter persona modeling per user

---

## Contributing

PRs and feedback are welcome! If you'd like to contribute:
1. Fork the repo
2. Create a branch
3. Submit a pull request

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Author

Built by [@konkentrate](https://github.com/konkentrate)

---

## _Outdated_

This is a simple chat bot for Discord that uses Mistral AI (local LLMs in the future) for chit-chatting purposes.

### Scope

- LLM implementation using configurable Mistral AI (DeepSeek in the future?)
- Contextual chat history using SQLite database
- Ability to analyze chat history and provide insights
- Trainable personality

### Technical Notes

- Use SQLite for chat history storage locally
- Use FAISS for vector storage and search of text embeddings
- Implement a simple filtering / moderation command to trim database

### TODO
- Add user-specific memory for personalized requests and general memory / context
- Other than relevant context messages, also give the last 5-10 messages before prompt, for fluent chatting

