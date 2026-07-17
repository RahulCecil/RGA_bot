import os
import psycopg2
from psycopg2.extras import RealDictCursor


def get_db_connection():
    conn_string = os.getenv(
        "DB_CONN_STRING",
        "postgresql://admin:secret_password@localhost:5433/ai_act_db",
    )
    return psycopg2.connect(conn_string)


def ensure_schema():
    embedding_dim = int(os.getenv("PRIVATEMODE_EMBEDDING_DIM", "2560"))
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    document_name TEXT NOT NULL,
                    article TEXT NOT NULL,          -- e.g., "Article 5" or "Recital 12"
                    paragraph_number INT,           -- e.g., 1, 2, 3 (NULL if it's general text)
                    page_number INT NOT NULL,
                    content TEXT NOT NULL,          -- The precise text of that paragraph
                    embedding VECTOR({embedding_dim})
                )
                """
            )
            conn.commit()
    finally:
        conn.close()
