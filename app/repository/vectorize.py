import os
import re
import sys
from pypdf import PdfReader
from openai import OpenAI
import psycopg2
from pgvector.psycopg2 import register_vector

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.core.database import ensure_schema

# 1. Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(REPO_ROOT, "docs", "EU_AI_Act_EN_TXT.pdf")

DB_CONN_STRING = os.getenv("DB_CONN_STRING", "postgresql://admin:secret_password@localhost:5432/ai_act_db")
PRIVATEMODE_PROXY_URL = os.getenv("PRIVATEMODE_PROXY_URL", "http://localhost:8080/v1")
PRIVATEMODE_API_KEY = os.getenv("PRIVATEMODE_API_KEY", "placeholder")
PRIVATEMODE_EMBEDDING_MODEL = os.getenv("PRIVATEMODE_EMBEDDING_MODEL", "qwen3-embedding-4b")
PRIVATEMODE_EMBEDDING_DIM = int(os.getenv("PRIVATEMODE_EMBEDDING_DIM", "2560"))

# Initialize OpenAI Client pointing to Privatemode AI Proxy
client = OpenAI(base_url=PRIVATEMODE_PROXY_URL, api_key=PRIVATEMODE_API_KEY)

# 2. Advanced Multi-Pass Extractor & Normalizer
def extract_and_chunk_by_article(pdf_path):
    reader = PdfReader(pdf_path)
    
    print("Extracting and normalizing raw text from PDF pages...")
    full_text_list = []
    page_map = [] 
    
    for page_idx, page in enumerate(reader.pages):
        page_text = page.extract_text()
        
        # Repair words split by hyphens at line breaks (e.g., Ar- \nticle -> Article)
        page_text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', page_text)
        
        full_text_list.append(page_text)
        page_map.append((len("".join(full_text_list[:-1])), len("".join(full_text_list)), page_idx + 1))
        
    full_text = "\n".join(full_text_list)
    
    # Standardize whitespace and hidden characters (like non-breaking spaces) to clean up formatting
    full_text = re.sub(r'[ \t\xa0]+', ' ', full_text)
    
    print("Slicing document by Article structures...")
    # Ultra-flexible lookahead: matches 'Article' or 'ticle' (in case 'Ar' was separated) followed by a number
    article_pattern = re.compile(r'(?=\b(?:Article|ticle)\s+\d+)', re.IGNORECASE)
    raw_chunks = article_pattern.split(full_text)
    
    chunks = []
    MAX_CHAR_LENGTH = 5000  # Safety threshold (~1200 words)
    
    for raw_chunk in raw_chunks:
        clean_chunk = raw_chunk.strip()
        if len(clean_chunk) < 50:
            continue
            
        # Extract the structural Article name from the front of the chunk
        title_match = re.match(r'^(?:Article|ticle)\s+(\d+)', clean_chunk, re.IGNORECASE)
        
        if title_match:
            article_name = f"Article {title_match.group(1)}"
        else:
            article_name = "Official Recitals / Preamble"
        
        # Resolve physical page location
        chunk_start_idx = full_text.find(clean_chunk[:30])
        page_number = 1
        for start, end, pg in page_map:
            if start <= chunk_start_idx <= end:
                page_number = pg
                break
                
        # Safety sub-chunking fallback for very long articles or the initial preamble block
        if len(clean_chunk) > MAX_CHAR_LENGTH:
            for sub_idx, i in enumerate(range(0, len(clean_chunk), MAX_CHAR_LENGTH)):
                sub_text = clean_chunk[i:i + MAX_CHAR_LENGTH]
                chunks.append({
                    "content": sub_text,
                    "document_name": os.path.basename(pdf_path),
                    "article": f"{article_name} (Part {sub_idx + 1})",
                    "page_number": page_number
                })
        else:
            chunks.append({
                "content": clean_chunk,
                "document_name": os.path.basename(pdf_path),
                "article": article_name,
                "page_number": page_number
            })
            
    return chunks

# 3. Generate vectors from the configured embedding model.
def get_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model=PRIVATEMODE_EMBEDDING_MODEL,
    )
    return response.data[0].embedding

# 4. Storage Matrix Execution
def store_in_db(chunks):
    ensure_schema()
    conn = psycopg2.connect(DB_CONN_STRING)
    conn.autocommit = True

    cursor = conn.cursor()
    register_vector(conn)
    
    print(f"\nUploading {len(chunks)} normalized segments into PGvector...")
    
    for i, chunk in enumerate(chunks):
        try:
            vector = get_embedding(chunk["content"])
            
            cursor.execute(
                """
                INSERT INTO document_chunks (content, document_name, article, page_number, embedding)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (chunk["content"], chunk["document_name"], chunk["article"], chunk["page_number"], vector)
            )
            
            if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                print(f"Successfully vectorized {i + 1}/{len(chunks)} chunks...")
                
        except Exception as e:
            print(f"❌ Error inserting chunk [{chunk['article']}]: {e}")
            continue
            
    cursor.close()
    conn.close()
    print("\n🎉 Data ingestion successfully resolved! Check DBeaver for structural verification.")

if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: Please check your filename. '{PDF_PATH}' was not found.")
    else:
        print("Starting advanced structural legal text parsing...")
        document_chunks = extract_and_chunk_by_article(PDF_PATH)
        store_in_db(document_chunks)