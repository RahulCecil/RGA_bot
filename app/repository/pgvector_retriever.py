import os
import re
import json
import psycopg2
from pypdf import PdfReader
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

    def _load_local_fallback(self, limit: int = 4) -> list:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        pdf_path = os.path.join(repo_root, "docs", "EU_AI_Act_EN_TXT.pdf")
        if not os.path.exists(pdf_path):
            return []

        reader = PdfReader(pdf_path)
        chunks = []
        for page_idx, page in enumerate(reader.pages[:3]):
            text = page.extract_text() or ""
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                chunks.append({
                    "text": text[:1800],
                    "document_name": os.path.basename(pdf_path),
                    "article": "PDF fallback",
                    "page": page_idx + 1,
                    "paragraph": "n/a",
                })
        return chunks[:limit]

    def search(self, query: str, limit: int = 4) -> list:
        try:
            query_vector = self._embed_text(query)
        except Exception:
            query_vector = None

        if query_vector is not None:
            try:
                conn = psycopg2.connect(self.connection_string)
            except Exception:
                conn = None

            if conn is not None:
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
                    if rows:
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
                except Exception:
                    pass
                finally:
                    conn.close()

        return self._load_local_fallback(limit=limit)
