import os
import re
from pypdf import PdfReader
from openai import OpenAI
import psycopg2
from pgvector.psycopg2 import register_vector

# 1. Configuration
PDF_PATH = "EU_AI_Act_EN_TXT.pdf"  # Place your PDF in this folder
DB_CONN_STRING = "postgresql://admin:secret_password@localhost:5432/ai_act_db" # Port 5432 to avoid conflicts
PRIVATEMODE_PROXY_URL = "http://localhost:8080/v1"
PRIVATEMODE_API_KEY = "placeholder"

# Initialize OpenAI Client (Privatemode AI proxy)
client = OpenAI(base_url=PRIVATEMODE_PROXY_URL, api_key=PRIVATEMODE_API_KEY)

# 2. Optimized Structural Extractor & Chunker
def extract_and_chunk_by_article(pdf_path):
    reader = PdfReader(pdf_path)
    
    # Step A: Extract all text into a single continuous string to eliminate page-break splits
    print("Extracting raw text from all pages...")
    full_text_list = []
    page_map = [] # Track page numbers for citations
    
    for page_idx, page in enumerate(reader.pages):
        page_text = page.extract_text()
        full_text_list.append(page_text)
        # Store page index boundaries to map characters back to physical pages later
        page_map.append((len("".join(full_text_list[:-1])), len("".join(full_text_list)), page_idx + 1))
        
    full_text = "\n".join(full_text_list)
    
    # Step B: Find all Article boundaries using regex.
    # This matches patterns like "Article 13" at the start of a line or after a paragraph break.
    # It uses a lookahead (?=Article\s+\d+) to split the text while keeping the "Article X" header in the chunk.
    print("Identifying Article boundaries...")
    article_pattern = re.compile(r'(?=\n\s*Article\s+\d+)', re.IGNORECASE)
    raw_chunks = article_pattern.split(full_text)
    
    chunks = []
    
    for raw_chunk in raw_chunks:
        clean_chunk = raw_chunk.strip()
        if len(clean_chunk) < 100: # Skip irrelevant trailing text, table of contents lines, or headers
            continue
            
        # Extract the Article title/number from the beginning of this chunk
        title_match = re.match(r'^(Article\s+\d+)', clean_chunk, re.IGNORECASE)
        article_name = title_match.group(1) if title_match else "General Context / Preamble"
        
        # Determine which physical PDF page this chunk primarily starts on
        chunk_start_idx = full_text.find(clean_chunk[:50])
        page_number = 1
        for start, end, pg in page_map:
            if start <= chunk_start_idx <= end:
                page_number = pg
                break
                
        chunks.append({
            "content": clean_chunk,
            "document_name": os.path.basename(pdf_path),
            "article": article_name,
            "page_number": page_number
        })
        
    return chunks

# 3. Generate High-Quality Vectors
def get_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# 4. Store Chunks & Metadata in PostgreSQL
def store_in_db(chunks):
    conn = psycopg2.connect(DB_CONN_STRING)
    cursor = conn.cursor()
    register_vector(conn)
    
    print(f"\nUploading {len(chunks)} parsed articles/preambles to PGvector...")
    
    for i, chunk in enumerate(chunks):
        try:
            # Vectorize the complete, self-contained Article context
            vector = get_embedding(chunk["content"])
            
            cursor.execute(
                """
                INSERT INTO document_chunks (content, document_name, article, page_number, embedding)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (chunk["content"], chunk["document_name"], chunk["article"], chunk["page_number"], vector)
            )
            
            if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                print(f"Successfully processed {i + 1}/{len(chunks)} articles...")
                
        except Exception as e:
            print(f"Failed parsing chunk starting with '{chunk['content'][:30]}': {e}")
            continue
            
    conn.commit()
    cursor.close()
    conn.close()
    print("\n🎉 Database vectorization complete! Your data is structured, embedded, and ready.")

if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: Please place your '{PDF_PATH}' file in this directory before running.")
    else:
        print("Starting structural legal vectorization...")
        document_chunks = extract_and_chunk_by_article(PDF_PATH)
        store_in_db(document_chunks)