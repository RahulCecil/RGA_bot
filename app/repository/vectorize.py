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

DB_CONN_STRING = os.getenv("DB_CONN_STRING", "postgresql://admin:secret_password@localhost:5433/ai_act_db")
PRIVATEMODE_PROXY_URL = os.getenv("PRIVATEMODE_PROXY_URL", "http://localhost:8080/v1")
PRIVATEMODE_API_KEY = os.getenv("PRIVATEMODE_API_KEY", "placeholder")
PRIVATEMODE_EMBEDDING_MODEL = os.getenv("PRIVATEMODE_EMBEDDING_MODEL", "qwen3-embedding-4b")
PRIVATEMODE_EMBEDDING_DIM = int(os.getenv("PRIVATEMODE_EMBEDDING_DIM", "2560"))

# Initialize OpenAI Client pointing to Privatemode AI Proxy
client = OpenAI(base_url=PRIVATEMODE_PROXY_URL, api_key=PRIVATEMODE_API_KEY)

# STRUCTURAL REGEX: Matches "Article X" only when it begins on a line.
# Bypasses leading page number artifacts (up to 4 digits/spaces)[cite: 1] and uses a negative lookahead
# to ignore inline citation matches such as "Article X of...", "Article X TFEU", "Article X paragraph..." etc.[cite: 3]
ARTICLE_HEADING_RE = re.compile(
    r"(?im)^[\s\d]{0,4}\b(?:Article|A\s*r\s*t\s*i\s*c\s*l\s*e)\s+(\d+)\b(?!\s+(?:of|TFEU|TEU|Directive|Regulation|paragraph|points|point|and))"
)


# 2. Text Cleaner Helper
def clean_page_text(text):
    """Strips recurrent EU Official Journal metadata noise from raw text."""
    # Strip ELI URIs[cite: 1]
    text = re.sub(r'ELI:\s+http://data\.europa\.eu/eli/reg/2024/1689/oj', '', text, flags=re.IGNORECASE)
    # Strip OJ publication metadata[cite: 1]
    text = re.sub(r'\bOJ\s+L,\s+\s?\d{1,2}\.\d{1,2}\.\d{4}\b', '', text, flags=re.IGNORECASE)
    # Strip page indicators (e.g., 1/144, 44/144)[cite: 1]
    text = re.sub(r'\b\d{1,3}/144\b', '', text)
    # Remove stand-alone EN language markers[cite: 1]
    text = re.sub(r'^\s*EN\s*$', '', text, flags=re.MULTILINE)
    # Standardize spaced-out Article characters[cite: 1]
    text = re.sub(r'(?im)\bA\s*r\s*t\s*i\s*c\s*l\s*e\s+(\d+)\b', r'Article \1', text)
    return text


# 3. Advanced Multi-Pass Extractor & Normalizer
def extract_and_chunk_by_article(pdf_path):
    reader = PdfReader(pdf_path)
    filename = os.path.basename(pdf_path)
    
    print("Extracting and normalizing raw text from PDF pages...")
    full_text_list = []
    page_map = [] 
    
    current_offset = 0
    for page_idx, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        
        # Repair words split by hyphens at line breaks (e.g., Ar- \nticle -> Article)[cite: 1]
        page_text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', page_text)
        # Strip recurring Official Journal metadata[cite: 1]
        page_text = clean_page_text(page_text)
        
        full_text_list.append(page_text)
        
        # Exact index mapping to trace where pages begin/end globally (eliminates search bugs)[cite: 1]
        text_len = len(page_text)
        page_map.append((current_offset, current_offset + text_len, page_idx + 1))
        current_offset += text_len + 1  # +1 accounts for join separator newline
        
    full_text = "\n".join(full_text_list)
    
    # Standardize whitespace and hidden characters (like non-breaking spaces)[cite: 1]
    full_text = re.sub(r'[ \t\xa0]+', ' ', full_text)
    
    # Locate where preambles end and the enacting articles begin.[cite: 1]
    marker_match = re.search(r'\bHAVE\s+ADOPTED\s+THIS\s+REGULATION\s*:', full_text, flags=re.IGNORECASE)
    if marker_match:
        print("Found official transition marker 'HAVE ADOPTED THIS REGULATION:'")
        preamble_text = full_text[:marker_match.start()]
        enacted_text = full_text[marker_match.end():]

    else:
        # Fallback using "Article 1" explicitly to avoid false positives[cite: 1]
        first_article_match = re.search(r'(?im)^[\s\d]{0,4}\bArticle\s+1\b', full_text)
        if first_article_match:
            print("Transition marker not found. Splitting precisely at Article 1...")
            preamble_text = full_text[:first_article_match.start()]
            enacted_text = full_text[first_article_match.start():]
        else:
            print("⚠️ Boundary transition missing. Splitting document by estimation (1/3).")
            midpoint = len(full_text) // 3
            preamble_text = full_text[:midpoint]
            enacted_text = full_text[midpoint:]
    
    chunks = []
    MAX_CHAR_LENGTH = 4500  # Fallback safety character limit[cite: 1]
    OVERLAP_CHAR_LENGTH = 500  # 11% character overlap for fallback splits[cite: 1]

    # Helper function to find the physical page number based on character offset[cite: 1]
    def get_page_number(char_idx):
        for start, end, pg in page_map:
            if start <= char_idx <= end:
                return pg
        return 1

    # =========================================================================
    # SECTION A: Parse the Recitals/Preamble
    # =========================================================================
    print("Parsing preambles and recitals...")
    
    # Split by numbered recitals: e.g. \n(1), \n(12)[cite: 1]
    recital_pattern = re.compile(r'(?=\n\s*\(\d+\)\s+)')
    raw_recitals = recital_pattern.split(preamble_text)
    
    current_search_offset = 0
    
    for item in raw_recitals:
        clean_rec = item.strip()
        if len(clean_rec) < 40:
            continue
            
        rec_match = re.match(r'^\((\d+)\)', clean_rec)
        recital_id = int(rec_match.group(1)) if rec_match else None
        meta_name = f"Recital {recital_id}" if recital_id else "Official Preamble"
        
        # Find index starting only from our last processed offset to guarantee correctness[cite: 1]
        global_idx = full_text.find(clean_rec[:40], current_search_offset)
        if global_idx == -1:
            global_idx = current_search_offset  # Fallback
        else:
            current_search_offset = global_idx
            
        pg_num = get_page_number(global_idx)
        
        context_header = f"[Document: {filename} | {meta_name}]\n\n"
        
        # Safety sub-split for extremely long or failed regex split recitals[cite: 1]
        if len(clean_rec) > MAX_CHAR_LENGTH:
            step = MAX_CHAR_LENGTH - OVERLAP_CHAR_LENGTH
            for sub_idx, i in enumerate(range(0, len(clean_rec), step)):
                sub_content = clean_rec[i:i + MAX_CHAR_LENGTH]
                chunks.append({
                    "content": f"[Document: {filename} | {meta_name} (Part {sub_idx + 1})]\n\n{sub_content}",
                    "document_name": filename,
                    "article": f"{meta_name} (Part {sub_idx + 1})",
                    "paragraph_number": recital_id,
                    "page_number": pg_num
                })
        else:
            chunks.append({
                "content": context_header + clean_rec,
                "document_name": filename,
                "article": meta_name,
                "paragraph_number": recital_id,
                "page_number": pg_num
            })

    # =========================================================================
    # SECTION B: Parse the Enacted Articles and Paragraphs
    # =========================================================================
    if enacted_text.strip():
        print("Parsing structural articles and paragraph numbers...")
        
        article_matches = list(ARTICLE_HEADING_RE.finditer(enacted_text))
        print(f"Located {len(article_matches)} distinct Articles in the enactment section.")

        for idx, match in enumerate(article_matches):
            article_no = int(match.group(1))
            article_name = f"Article {article_no}"

            start_idx = match.start()
            end_idx = article_matches[idx + 1].start() if idx + 1 < len(article_matches) else len(enacted_text)
            article_block = enacted_text[start_idx:end_idx].strip()
            if len(article_block) < 30:
                continue

            article_global_idx = full_text.find(article_block[:40])
            if article_global_idx == -1:
                article_global_idx = len(preamble_text) + start_idx
            article_page = get_page_number(article_global_idx)

            # --- PARENT CHUNK (Article-Level Overview Context) ---
            parent_context_header = f"[Document: {filename} | {article_name} (Overview)]\n\n"
            
            if len(article_block) > MAX_CHAR_LENGTH:
                step = MAX_CHAR_LENGTH - OVERLAP_CHAR_LENGTH
                for sub_idx, i in enumerate(range(0, len(article_block), step)):
                    sub_content = article_block[i:i + MAX_CHAR_LENGTH]
                    chunks.append({
                        "content": f"[Document: {filename} | {article_name} (Part {sub_idx + 1})]\n\n{sub_content}",
                        "document_name": filename,
                        "article": f"{article_name} (Part {sub_idx + 1})",
                        "paragraph_number": None,
                        "page_number": article_page
                    })
            else:
                chunks.append({
                    "content": parent_context_header + article_block,
                    "document_name": filename,
                    "article": article_name,
                    "paragraph_number": None,
                    "page_number": article_page
                })

            # --- CHILD CHUNKS (Paragraph-Level Slices) ---
            paragraph_pattern = re.compile(r'(?=\n\s*\d+\.\s+)')
            raw_paragraphs = paragraph_pattern.split(article_block)
            paragraph_search_offset = article_global_idx

            for raw_para in raw_paragraphs:
                clean_para = raw_para.strip()
                if len(clean_para) < 20:
                    continue

                para_match = re.match(r'^(\d+)\.', clean_para)
                if not para_match:
                    # Ignore article titles and unnumbered header text in child layer.[cite: 1]
                    continue
                para_num = int(para_match.group(1))

                para_global_idx = full_text.find(clean_para[:30], paragraph_search_offset)
                if para_global_idx == -1:
                    para_global_idx = paragraph_search_offset
                else:
                    paragraph_search_offset = para_global_idx
                para_page = get_page_number(para_global_idx)

                # Small-to-Large context injection header[cite: 1]
                child_context_header = f"[Document: {filename} | {article_name} | Paragraph {para_num}]\n\n"

                if len(clean_para) > MAX_CHAR_LENGTH:
                    step = MAX_CHAR_LENGTH - OVERLAP_CHAR_LENGTH
                    for sub_idx, i in enumerate(range(0, len(clean_para), step)):
                        sub_content = clean_para[i:i + MAX_CHAR_LENGTH]
                        chunks.append({
                            "content": f"[Document: {filename} | {article_name} | Paragraph {para_num} (Part {sub_idx + 1})]\n\n{sub_content}",
                            "document_name": filename,
                            "article": article_name,
                            "paragraph_number": para_num,
                            "page_number": para_page
                        })
                else:
                    chunks.append({
                        "content": child_context_header + clean_para,
                        "document_name": filename,
                        "article": article_name,
                        "paragraph_number": para_num,
                        "page_number": para_page
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
    
    print(f"\nUploading {len(chunks)} contextualized segments into PGvector...")
    
    for i, chunk in enumerate(chunks):
        try:
            # Send the context-injected text directly to the embedding model
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
    print("\n🎉 Ingestion complete! Articles are cleanly mapped and context-injected.")


if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: Please check your filename. '{PDF_PATH}' was not found.")
    else:
        print("Starting robust structural legal text parsing with context injection...")
        document_chunks = extract_and_chunk_by_article(PDF_PATH)
        store_in_db(document_chunks)