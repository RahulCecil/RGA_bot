import os
import psycopg2
from openai import OpenAI
from pgvector.psycopg2 import register_vector


class PGVectorRetriever:
    def __init__(self, connection_string=None):
        self.connection_string = connection_string or os.getenv(
            "DB_CONN_STRING",
            "postgresql://admin:secret_password@localhost:5432/ai_act_db",
        )
        self.client = OpenAI(
            base_url=os.getenv("PRIVATEMODE_PROXY_URL", "http://localhost:8080/v1"),
            api_key=os.getenv("PRIVATEMODE_API_KEY", "placeholder"),
        )
        self.embedding_model = os.getenv("PRIVATEMODE_EMBEDDING_MODEL", "qwen3-embedding-4b")

    def _embed_text(self, text: str) -> list:
        response = self.client.embeddings.create(input=[text], model=self.embedding_model)
        return response.data[0].embedding

    def search(self, query: str, limit: int = 4) -> list:
        query_vector = self._embed_text(query)
        conn = psycopg2.connect(self.connection_string)
        try:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT content, document_name, article, page_number
                    FROM document_chunks
                    ORDER BY embedding <-> %s
                    LIMIT %s
                    """,
                    (query_vector, limit),
                )
                rows = cur.fetchall()
        finally:
            conn.close()

        return [
            {
                "text": row[0],
                "document_name": row[1],
                "article": row[2],
                "page": row[3],
                "paragraph": "",
            }
            for row in rows
        ]
