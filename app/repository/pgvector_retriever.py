import os
import re
import json
import logging
import psycopg2
from pypdf import PdfReader
from openai import OpenAI
from pgvector.psycopg2 import register_vector


logger = logging.getLogger(__name__)


class PGVectorRetriever:
    def __init__(self, connection_string=None):
        self.connection_string = connection_string or os.getenv(
            "DB_CONN_STRING",
            "postgresql://admin:secret_password@localhost:5433/ai_act_db",
        )
        self.client = OpenAI(
            base_url=os.getenv("PRIVATEMODE_PROXY_URL", "http://localhost:8080/v1"),
            api_key=os.getenv("PRIVATEMODE_API_KEY", "placeholder"),
        )
        self.embedding_model = os.getenv("PRIVATEMODE_EMBEDDING_MODEL", "qwen3-embedding-4b")

    def _embed_text(self, text: str) -> list:
        response = self.client.embeddings.create(input=[text], model=self.embedding_model)
        return response.data[0].embedding

    def _extract_article_number(self, query: str):
        match = re.search(r"\b(?:article|art\.?|a\s*r\s*t\s*i\s*c\s*l\s*e)\s*(\d+)\b", query, flags=re.IGNORECASE)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

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
        article_number = self._extract_article_number(query)

        try:
            query_vector = self._embed_text(query)
        except Exception as exc:
            logger.warning("Embedding request failed; using local fallback documents: %s", exc)
            query_vector = None

        if query_vector is not None:
            try:
                conn = psycopg2.connect(self.connection_string)
            except Exception as exc:
                logger.warning("Database connection failed; using local fallback documents: %s", exc)
                conn = None

            if conn is not None:
                try:
                    register_vector(conn)
                    with conn.cursor() as cur:
                        # If the user asks for a specific article, prioritize exact article chunks first.
                        if article_number is not None:
                            vector_literal = "[" + ",".join(str(float(x)) for x in query_vector) + "]"

                            # 1) Parent-level article chunks (paragraph_number IS NULL) provide broad context.
                            cur.execute(
                                """
                                SELECT content, document_name, article, paragraph_number, page_number
                                FROM document_chunks
                                WHERE article ILIKE %s
                                  AND paragraph_number IS NULL
                                ORDER BY page_number, paragraph_number NULLS LAST
                                LIMIT 1
                                """,
                                (f"Article {article_number}%",),
                            )
                            parent_rows = cur.fetchall()

                            # 2) Child paragraph chunks within that article are ranked by vector similarity.
                            child_limit = max(limit - len(parent_rows), 1)
                            cur.execute(
                                """
                                SELECT content, document_name, article, paragraph_number, page_number
                                FROM document_chunks
                                WHERE article ILIKE %s
                                  AND paragraph_number IS NOT NULL
                                ORDER BY embedding <-> %s::vector
                                LIMIT %s
                                """,
                                (f"Article {article_number}%", vector_literal, child_limit),
                            )
                            child_rows = cur.fetchall()

                            combined = parent_rows + child_rows
                            if combined:
                                return [
                                    {
                                        "text": row[0],
                                        "document_name": row[1],
                                        "article": row[2],
                                        "paragraph": str(row[3]) if row[3] is not None else "",
                                        "page": row[4],
                                    }
                                    for row in combined[:limit]
                                ]

                        # Send a proper vector literal so the distance operator sees vector <-> vector.
                        vector_literal = "[" + ",".join(str(float(x)) for x in query_vector) + "]"
                        cur.execute(
                            """
                            SELECT content, document_name, article, paragraph_number, page_number
                            FROM document_chunks
                            ORDER BY embedding <-> %s::vector
                            LIMIT %s
                            """,
                            (vector_literal, limit),
                        )
                        rows = cur.fetchall()
                    if rows:
                        return [
                            {
                                "text": row[0],
                                "document_name": row[1],
                                "article": row[2],
                                "paragraph": str(row[3]) if row[3] is not None else "",
                                "page": row[4],
                            }
                            for row in rows
                        ]
                except Exception:
                    logger.warning("PGVector query failed; using local fallback documents", exc_info=True)
                finally:
                    conn.close()

        return self._load_local_fallback(limit=limit)
