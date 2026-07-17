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


# 2. Text Cleaner Helper
def clean_page_text(text):
    """Strips recurrent EU Official Journal metadata noise from raw text."""
    # Strip ELI URIs
    text = re.sub(r'ELI:\s+http://data\.europa\.eu/eli/reg/2024/1689/oj', '', text, flags=re.IGNORECASE)
    # Strip OJ publication metadata
    text = re.sub(r'\bOJ\s+L,\s+\d{1,2}\.\d{1,2}\.\d{4}\b', '', text, flags=re.IGNORECASE)
    # Strip page indicators (e.g., 1/144, 44/144)
    text = re.sub(r'\b\d{1,3}/144\b', '', text)
    # Remove stand-alone EN language markers
    text = re.sub(r'^\s*EN\s*$', '', text, flags=re.MULTILINE)
    return text


# 3. Advanced Multi-Pass Extractor & Normalizer
def extract_and_chunk_by_article(pdf_path):
    reader = PdfReader(pdf_path)
    
    print("Extracting and normalizing raw text from PDF pages...")
    full_text_list = []
    page_map = [] 
    
    current_offset = 0
    for page_idx, page in enumerate(reader.pages):
        page_text = page.extract_text()
        
        # Repair words split by hyphens at line breaks (e.g., Ar- \nticle -> Article)
        page_text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', page_text)
        # Strip recurring Official Journal metadata
        page_text = clean_page_text(page_text)
        
        full_text_list.append(page_text)
        
        # Exact index mapping to trace where pages begin/end globally (eliminates search bugs)
        text_len = len(page_text)
        page_map.append((current_offset, current_offset + text_len, page_idx + 1))
        current_offset += text_len + 1  # +1 accounts for join separator newline
        
    full_text = "\n".join(full_text_list)
    
    # Standardize whitespace and hidden characters (like non-breaking spaces)
    full_text = re.sub(r'[ \t\xa0]+', ' ', full_text)
    
    # Locate where preambles end and the enacting articles begin.
    marker_match = re.search(r'\bHAVE\s+ADOPTED\s+THIS\s+REGULATION:', full_text, flags=re.IGNORECASE)
    if marker_match:
        preamble_text = full_text[:marker_match.start()]
        enacted_text = full_text[marker_match.end():]
    else:
        # Fallback for OCR/text extraction variants where the exact marker is missing.
        first_article_match = re.search(r'\b(?:Article|ticle)\s+\d+\b', full_text, flags=re.IGNORECASE)
        if first_article_match:
            preamble_text = full_text[:first_article_match.start()]
            enacted_text = full_text[first_article_match.start():]
        else:
            preamble_text = full_text
            enacted_text = ""
    
    chunks = []
    MAX_CHAR_LENGTH = 4500  # Fallback safety character limit
    OVERLAP_CHAR_LENGTH = 500  # 11% character overlap for fallback splits

    # Helper function to find the physical page number based on character offset
    def get_page_number(char_idx):
        for start, end, pg in page_map:
            if start <= char_idx <= end:
                return pg
        return 1

    # =========================================================================
    # SECTION A: Parse the Recitals/Preamble (FIXED with safety sub-splits)
    # =========================================================================
    print("Parsing preambles and recitals...")
    
    # Split by numbered recitals: e.g. \n(1), \n(12)
    recital_pattern = re.compile(r'(?=\n\s*\(\d+\)\s+)')
    raw_recitals = recital_pattern.split(preamble_text)
    
    # We track our progress offset so we don't have to search from start of doc
    current_search_offset = 0
    
    for item in raw_recitals:
        clean_rec = item.strip()
        if len(clean_rec) < 40:
            continue
            
        rec_match = re.match(r'^\((\d+)\)', clean_rec)
        recital_id = int(rec_match.group(1)) if rec_match else None
        meta_name = f"Recital {recital_id}" if recital_id else "Official Preamble"
        
        # Find index starting only from our last processed offset to guarantee correctness
        global_idx = full_text.find(clean_rec[:40], current_search_offset)
        if global_idx == -1:
            global_idx = current_search_offset  # Fallback
        else:
            current_search_offset = global_idx
            
        pg_num = get_page_number(global_idx)
        
        # Safety sub-split for extremely long or failed regex split recitals
        if len(clean_rec) > MAX_CHAR_LENGTH:
            step = MAX_CHAR_LENGTH - OVERLAP_CHAR_LENGTH
            for sub_idx, i in enumerate(range(0, len(clean_rec), step)):
                chunks.append({
                    "content": clean_rec[i:i + MAX_CHAR_LENGTH],
                    "document_name": os.path.basename(pdf_path),
                    "article": f"{meta_name} (Part {sub_idx + 1})",
                    "paragraph_number": recital_id,  # Use recital number as structural position
                    "page_number": pg_num
                })
        else:
            chunks.append({
                "content": clean_rec,
                "document_name": os.path.basename(pdf_path),
                "article": meta_name,
                "paragraph_number": recital_id,  # Use recital number as structural position
                "page_number": pg_num
            })

    # =========================================================================
    # SECTION B: Parse the Enacted Articles and Paragraphs
    # =========================================================================
    if enacted_text.strip():
        print("Parsing structural articles and paragraph numbers...")
        
        # Split only when "Article" starts on a fresh line to bypass inline citations
        article_pattern = re.compile(r'(?=\n\s*(?:Article|ticle)\s+\d+)', re.IGNORECASE)
        raw_articles = article_pattern.split(enacted_text)
        
        # Set search cursor to start of enacted text
        current_search_offset = len(preamble_text)
        
        for raw_art in raw_articles:
            clean_art = raw_art.strip()
            if len(clean_art) < 50:
                continue
                
            title_match = re.match(r'^(?:Article|ticle)\s+(\d+)', clean_art, re.IGNORECASE)
            article_name = f"Article {title_match.group(1)}" if title_match else "Regulation Provision"
            
            # Anchor our search pointer to where this specific article starts
            article_global_idx = full_text.find(clean_art[:40], current_search_offset)
            if article_global_idx == -1:
                article_global_idx = current_search_offset
            else:
                current_search_offset = article_global_idx
                
            # Split this specific article into its physical paragraphs (e.g. "1.   ", "2.   " at start of lines)
            paragraph_pattern = re.compile(r'(?=\n\s*\d+\.\s+)')
            raw_paragraphs = paragraph_pattern.split(clean_art)
            
            # If the article has no numbered paragraphs, process it as a single block
            if len(raw_paragraphs) <= 1:
                pg_num = get_page_number(article_global_idx)
                if len(clean_art) > MAX_CHAR_LENGTH:
                    step = MAX_CHAR_LENGTH - OVERLAP_CHAR_LENGTH
                    for sub_idx, i in enumerate(range(0, len(clean_art), step)):
                        chunks.append({
                            "content": clean_art[i:i + MAX_CHAR_LENGTH],
                            "document_name": os.path.basename(pdf_path),
                            "article": f"{article_name} (Part {sub_idx + 1})",
                            "paragraph_number": None,
                            "page_number": pg_num
                        })
                else:
                    chunks.append({
                        "content": clean_art,
                        "document_name": os.path.basename(pdf_path),
                        "article": article_name,
                        "paragraph_number": None,
                        "page_number": pg_num
                    })
            else:
                # Process each numbered paragraph sequentially
                paragraph_search_offset = article_global_idx
                for raw_para in raw_paragraphs:
                    clean_para = raw_para.strip()
                    if len(clean_para) < 20:
                        continue
                    
                    # Track coordinates of this specific paragraph
                    para_global_idx = full_text.find(clean_para[:30], paragraph_search_offset)
                    if para_global_idx == -1:
                        para_global_idx = paragraph_search_offset
                    else:
                        paragraph_search_offset = para_global_idx
                        
                    pg_num = get_page_number(para_global_idx)
                    
                    # Extract paragraph number from the front (e.g. "1. High-risk...")
                    para_match = re.match(r'^(\d+)\.', clean_para)
                    para_num = int(para_match.group(1)) if para_match else None
                    
                    # Safety sub-split for extremely long paragraphs
                    if len(clean_para) > MAX_CHAR_LENGTH:
                        step = MAX_CHAR_LENGTH - OVERLAP_CHAR_LENGTH
                        for sub_idx, i in enumerate(range(0, len(clean_para), step)):
                            chunks.append({
                                "content": clean_para[i:i + MAX_CHAR_LENGTH],
                                "document_name": os.path.basename(pdf_path),
                                "article": article_name,
                                "paragraph_number": para_num,
                                "page_number": pg_num
                            })
                    else:
                        chunks.append({
                            "content": clean_para,
                            "document_name": os.path.basename(pdf_path),
                            "article": article_name,
                            "paragraph_number": para_num,
                            "page_number": pg_num
                        })
                
    return chunks


# 4. Generate vectors from the configured embedding model
def get_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model=PRIVATEMODE_EMBEDDING_MODEL,
    )
    return response.data[0].embedding


# 5. Storage Matrix Execution
def store_in_db(chunks):
    ensure_schema()
    conn = psycopg2.connect(DB_CONN_STRING)
    conn.autocommit = True

    cursor = conn.cursor()
    register_vector(conn)
    
    print(f"\nUploading {len(chunks)} paragraph-level segments into PGvector...")
    
    for i, chunk in enumerate(chunks):
        try:
            vector = get_embedding(chunk["content"])
            
            cursor.execute(
                """
                INSERT INTO document_chunks (content, document_name, article, paragraph_number, page_number, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    chunk["content"], 
                    chunk["document_name"], 
                    chunk["article"], 
                    chunk["paragraph_number"], 
                    chunk["page_number"], 
                    vector
                )
            )
            
            if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                print(f"Successfully vectorized {i + 1}/{len(chunks)} chunks...")
                
        except Exception as e:
            print(f"❌ Error inserting chunk [{chunk['article']} - Para {chunk['paragraph_number']}]: {e}")
            continue
            
    cursor.close()
    conn.close()
    print("\n🎉 Data ingestion successfully completed! Every structural item has been mapped.")


if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: Please check your filename. '{PDF_PATH}' was not found.")
    else:
        print("Starting advanced structural legal text parsing...")
        document_chunks = extract_and_chunk_by_article(PDF_PATH)
        store_in_db(document_chunks)