# Chatbot for Discord using Mistral AI
This is a simple chat bot for Discord that uses Mistral AI (local LLMs in the future) for chit-chatting purposes.

### Scope

- LLM implementation using configurable Mistral AI (DeepSeek in the future?)
- Contextual chat history using SQLite database (or other)
- Ability to analyze chat history and provide insights
- Trainable personality

### Technical Notes

- Use SQLite for chat history storage locally
- Use FAISS for vector storage and search of text embeddings
- Implement a simple filtering / moderation command to trim database
- 