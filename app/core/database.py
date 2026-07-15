import os
import psycopg2
from psycopg2.extras import RealDictCursor


def get_db_connection():
    conn_string = os.getenv(
        "DB_CONN_STRING",
        "postgresql://admin:secret_password@localhost:5432/ai_act_db",
    )
    return psycopg2.connect(conn_string)


def ensure_schema():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    document_name TEXT NOT NULL,
                    article TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    embedding VECTOR(1024)
                )
                """
            )
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
    finally:
        conn.close()
