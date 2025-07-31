### Ignore deprecation warnings from torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.modules.module")
###

import sqlite3
import os
import numpy as np
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from datetime import datetime
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import time

class MessageMemory:

    def __init__(self, db_path="data/message_history.db", index_path="data/faiss_index"):
        self.db_path = db_path
        self.index_path = index_path
        # Explicitly set model_name to avoid deprecation warning
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.dimension = 384  # Default dimension for this model

        self._init_db()
        self._init_faiss()

        # Short-term memory: last k messages
        self.short_term_memory = ConversationBufferWindowMemory(
            k=7,
            return_messages=True,
            memory_key="chat_history"
        )

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                message TEXT,
                summary TEXT,
                is_bot BOOLEAN,
                timestamp TEXT,
                embedding_id INTEGER
            )
            ''')
            conn.commit()

    def _init_faiss(self):
        if os.path.exists(f"{self.index_path}.index"):
            self.index = faiss.read_index(f"{self.index_path}.index")
        else:
            self.index = faiss.IndexFlatL2(self.dimension)

    # def summarise_message(self, message, model):
    #     """Use AI to create a concise summary of a message for quick context understanding."""
    #     try:
    #         prompt = (
    #             "Create an extremely concise summary of this message. "
    #             "Include only the key information another AI would need to understand the context. "
    #             "Be as brief as possible while preserving the essential meaning:\n\n"
    #             f"{message}"
    #         )
    #         summary = model.invoke([HumanMessage(content=prompt)]).content
    #         print(f"[DEBUG] Generated summary: {summary[:80]}")
    #         return summary
    #     except Exception as e:
    #         print(f"[DEBUG] AI summarisation failed: {e}")
    #         # Fallback to simple truncation if AI summarisation fails
    #         return message[:200]

    # def _fallback_summarise(self, message):
    #     """Simple fallback summarisation when AI model is not available."""
    #     words = message.split()
    #     if len(words) <= 30:  # If message is already short, use as is
    #         return message
    #     return " ".join(words[:30]) + "..."

    def save_message(self, user_id, message, is_bot=False, model=None):
        print(f"[DEBUG] Saving message to DB | user_id: {user_id} | is_bot: {is_bot} | message: {message[:80]}")
        embedding = self.embeddings.embed_query(message)
        embedding_id = self.index.ntotal

        # Deactivated summarisation: just use the whole message as summary
        summary = message

        self.index.add(np.array([embedding], dtype='float32'))
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO messages (user_id, message, summary, is_bot, timestamp, embedding_id) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, message, summary, is_bot, datetime.now().isoformat(), embedding_id)
            )
            conn.commit()
        print(f"[DEBUG] Message saved with embedding_id: {embedding_id} | summary: {summary[:80]}")
        faiss.write_index(self.index, f"{self.index_path}.index")

        # Add to short-term memory
        if is_bot:
            self.short_term_memory.chat_memory.add_ai_message(message)
        else:
            self.short_term_memory.chat_memory.add_user_message(message)

    def get_short_term_history(self):
        # Returns last k messages as LangChain messages
        return self.short_term_memory.load_memory_variables({})["chat_history"]

    def get_relevant_long_term_history(self, query, user_id=None, k=6, distance_threshold=0.7, user_weight=0.5, recency_weight=0.2, similarity_cutoff=0.92):
        """
        Semantic search for relevant historical messages, weighted by semantic similarity and user match.
        Filters out messages that are too similar to each other.
        user_weight: additional weight (lower score) for messages from the same user.
        similarity_cutoff: if two messages have cosine similarity above this, only one is kept.
        """
        query_vector = self.embeddings.embed_query(query)
        D, I = self.index.search(np.array([query_vector], dtype='float32'), k*2)  # get more candidates for filtering
        candidates = []
        candidate_embeddings = []
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            for idx, dist in zip(I[0], D[0]):
                if idx == -1:
                    continue
                result = conn.execute(
                    "SELECT message, is_bot, user_id, timestamp FROM messages WHERE embedding_id = ?",
                    (int(idx),)
                ).fetchone()
                if result:
                    msg_user_id = result[2]
                    msg_time = datetime.fromisoformat(result[3]).timestamp()
                    recency_score = recency_weight * (1 - min((now - msg_time) / (60*60*24), 1))  # boost for messages within 24h
                    effective_dist = dist - user_weight if user_id and msg_user_id == user_id else dist
                    effective_dist -= recency_score
                    print(f"[DEBUG] idx: {idx}, raw_dist: {dist}, user_id: {msg_user_id}, effective_dist: {effective_dist}, recency_score: {recency_score}")
                    if effective_dist < distance_threshold:
                        candidates.append((
                            AIMessage(content=result[0]) if result[1] else HumanMessage(content=result[0]),
                            self.embeddings.embed_query(result[0])
                        ))
                        candidate_embeddings.append(self.embeddings.embed_query(result[0]))

        # Filter out messages that are too similar to each other
        # Diversify context using clustering if enough candidates
        if len(candidate_embeddings) > k:
            try:
                n_clusters = min(k, len(candidate_embeddings))
                kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
                labels = kmeans.fit_predict(candidate_embeddings)
                selected = []
                selected_embeddings = []
                for cluster_id in range(n_clusters):
                    idxs = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
                    # Pick the candidate closest to the cluster center
                    center = kmeans.cluster_centers_[cluster_id]
                    best_idx = min(idxs, key=lambda i: np.linalg.norm(candidate_embeddings[i] - center))
                    msg, emb = candidates[best_idx]
                    selected.append(msg)
                    selected_embeddings.append(emb)
                # If less than k clusters, fill up with remaining diverse candidates
                if len(selected) < k:
                    for i, (msg, emb) in enumerate(candidates):
                        if msg not in selected and len(selected) < k:
                            sims = cosine_similarity([emb], selected_embeddings)[0]
                            if all(sim < similarity_cutoff for sim in sims):
                                selected.append(msg)
                                selected_embeddings.append(emb)
            except Exception as e:
                print(f"[DEBUG] Clustering failed: {e}")
                # fallback to previous selection
                selected = []
                selected_embeddings = []
                for msg, emb in candidates:
                    if len(selected) == 0:
                        selected.append(msg)
                        selected_embeddings.append(emb)
                    else:
                        sims = cosine_similarity([emb], selected_embeddings)[0]
                        if all(sim < similarity_cutoff for sim in sims):
                            selected.append(msg)
                            selected_embeddings.append(emb)
                    if len(selected) >= k:
                        break
        else:
            selected = []
            selected_embeddings = []
            for msg, emb in candidates:
                if len(selected) == 0:
                    selected.append(msg)
                    selected_embeddings.append(emb)
                else:
                    sims = cosine_similarity([emb], selected_embeddings)[0]
                    if all(sim < similarity_cutoff for sim in sims):
                        selected.append(msg)
                        selected_embeddings.append(emb)
                if len(selected) >= k:
                    break

        return selected
