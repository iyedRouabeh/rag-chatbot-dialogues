import os
import psycopg
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# load env
load_dotenv("src/.env")

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

DATA_DIR = "data"

# embedding model (384 dims)
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def main():
    conn = psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

    with conn.cursor() as cur:
        for filename in os.listdir(DATA_DIR):
            if not filename.endswith(".txt"):
                continue

            path = os.path.join(DATA_DIR, filename)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()

            if not content:
                continue

            embedding = model.encode(content)
            embedding = np.array(embedding, dtype=np.float32).tolist()

            cur.execute(
                """
                INSERT INTO dialogues (filename, content, embedding)
                VALUES (%s, %s, %s)
                """,
                (filename, content, embedding)
            )

        conn.commit()

    conn.close()
    print("✅ Ingestion finished")

if __name__ == "__main__":
    main()
