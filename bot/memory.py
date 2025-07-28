import sqlite3
import faiss
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import time


class ConversationMemory:
    def __init__(self, db_path: str = "conversations.db"):
        print(f"[DEBUG] Initializing ConversationMemory with database: {db_path}")
        self.db_path = db_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.initialize_db()
        self.load_faiss_index()

    def initialize_db(self):
        print("[DEBUG] Initializing SQLite database...")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 content TEXT NOT NULL,
                 author TEXT NOT NULL,
                 role TEXT NOT NULL DEFAULT 'user',
                 timestamp DATETIME NOT NULL,
                 embedding BLOB)
            """)
            conn.commit()

            # Print schema for confirmation
            cursor = conn.execute("PRAGMA table_info(messages)")
            columns = [row[1] for row in cursor.fetchall()]
            print(f"[DEBUG] messages table columns: {columns}")

        print("[DEBUG] Database initialization complete")

    def store_message(self, content: str, author: str, role: str = "user"):
        print(f"[DEBUG] Storing message from {author} (role={role}): {content[:50]}...")
        embedding = self.model.encode([content])[0]

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO messages (content, author, role, timestamp, embedding) VALUES (?, ?, ?, ?, ?)",
                (content, author, role, datetime.now(), embedding.tobytes())
            )
            message_id = cursor.lastrowid
            conn.commit()
            print(f"[DEBUG] Inserted message with ID: {message_id}")

            cursor = conn.execute("SELECT COUNT(*) FROM messages")
            count = cursor.fetchone()[0]
            print(f"[DEBUG] Total messages in database: {count}")

        if self.index is None:
            print("[DEBUG] Creating new FAISS index")
            self.index = faiss.IndexFlatL2(embedding.shape[0])
        self.index.add(embedding.reshape(1, -1))
        print(f"[DEBUG] FAISS index size: {self.index.ntotal}")

    def search_similar_messages(self, query: str, k: int = 4, ignore_bot: bool = False) -> List[Tuple[str, float]]:
        print(f"[DEBUG] Searching for messages similar to: {query[:50]}... (ignore_bot={ignore_bot})")
        query_embedding = self.model.encode([query])[0]

        if self.index is None or self.index.ntotal == 0:
            print("[DEBUG] No messages in index to search")
            return []

        # Start timing the search
        start_time = time.perf_counter()  # Start timing

        # Get k+1 results since the query might be in the database
        distances, indices = self.index.search(query_embedding.reshape(1, -1), min(k+1, self.index.ntotal))

        elapsed = (time.perf_counter() - start_time) * 1000  # ms
        print(f"[DEBUG] FAISS search took {elapsed:.2f} ms")
        print(f"[DEBUG] FAISS found {len(indices[0])} results with indices: {indices[0]}")

        results = []
        with sqlite3.connect(self.db_path) as conn:
            for idx, distance in zip(indices[0], distances[0]):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue

                # Fetch message and role
                cursor = conn.execute(
                    """
                    SELECT content, author, role, timestamp 
                    FROM messages 
                    LIMIT 1 OFFSET ?
                    """,
                    (int(idx),)
                )
                row = cursor.fetchone()
                if row:
                    content, author, role, timestamp = row
                    if ignore_bot and role == "bot":
                        continue
                    similarity = 1.0 / (1.0 + float(distance))  # Convert distance to similarity score
                    results.append((f"{author} ({role}, {timestamp}): {content}", similarity))
                    print(f"[DEBUG] Found message at index {idx} with similarity {similarity:.3f} (role={role})")
                else:
                    print(f"[DEBUG] No message found at index {idx}")

        # Ensure we return only up to `k` results
        results = results[:k]
        print(f"[DEBUG] Returning {len(results)} results")
        return results

    def load_faiss_index(self):
        print("[DEBUG] Loading FAISS index from database...")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM messages")
            count = cursor.fetchone()[0]
            print(f"[DEBUG] Found {count} messages in database")

            cursor = conn.execute("SELECT embedding FROM messages")
            embeddings = cursor.fetchall()

            if embeddings:
                embedding_size = len(np.frombuffer(embeddings[0][0], dtype=np.float32))
                self.index = faiss.IndexFlatL2(embedding_size)
                print(f"[DEBUG] Created FAISS index with dimension {embedding_size}")

                all_embeddings = np.vstack([
                    np.frombuffer(emb[0], dtype=np.float32) for emb in embeddings
                ])
                self.index.add(all_embeddings)
                print(f"[DEBUG] Loaded {len(embeddings)} embeddings into FAISS index")
            else:
                print("[DEBUG] No existing embeddings found in database")

    def get_recent_messages(self, limit: int = 10) -> List[dict]:
        """Retrieve the most recent messages from the database."""
        print(f"[DEBUG] Retrieving {limit} most recent messages")

        results = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT content, author, role, timestamp 
                FROM messages 
                ORDER BY id DESC 
                LIMIT ?
                """,
                (limit,)
            )

            rows = cursor.fetchall()
            for row in rows:
                content, author, role, timestamp = row
                results.append({
                    "author": author,
                    "role": role,
                    "content": content,
                    "timestamp": timestamp
                })

            print(f"[DEBUG] Retrieved {len(results)} recent messages from database")

        # Return in chronological order (oldest first)
        return list(reversed(results))
