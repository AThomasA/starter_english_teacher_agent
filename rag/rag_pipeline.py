from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import os

# ============================================================
# CONFIG
# ============================================================

CHROMA_PATH = "rag/chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

client = chromadb.PersistentClient(path=CHROMA_PATH)

# ============================================================
# OCR CONFIG (IMPORTANTE)
# ============================================================

# Ajusta se necessário (Windows geralmente precisa)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ============================================================
# EXTRAÇÃO DE TEXTO (OCR + TEXTO NORMAL)
# ============================================================

def extract_text_from_pdf(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""

    for page_num, page in enumerate(doc):
        text = page.get_text()

        # Se não encontrou texto → usa OCR
        if not text.strip():
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            ocr_text = pytesseract.image_to_string(img)
            full_text += ocr_text + "\n"

        else:
            full_text += text + "\n"

    return full_text


# ============================================================
# CHUNKING (SEM LANGCHAIN)
# ============================================================

def chunk_text(text: str, chunk_size=500, overlap=80):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

# ============================================================
# DETECTAR TIPO DE LIVRO
# ============================================================

def detect_book_type(filename: str) -> str:
    name = filename.lower()

    if "student" in name:
        return "student_book"
    elif "workbook" in name:
        return "workbook"
    elif "teacher" in name:
        return "teacher_book"
    else:
        return "general"

# ============================================================
# INGESTÃO
# ============================================================

def run_ingestion(level: str):
    base_path = Path("rag/data/pdfs") / level

    if not base_path.exists():
        raise ValueError(f"Pasta não encontrada: {base_path}")

    pdf_files = list(base_path.glob("*.pdf"))

    print("Arquivos encontrados:")
    for f in pdf_files:
        print(f" - {f.name}")

    if not pdf_files:
        raise ValueError(f"Nenhum PDF encontrado em {base_path}")

    collection_name = f"sos_{level}"

    try:
        client.delete_collection(collection_name)
    except:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )

    all_chunks = []
    metadatas = []
    ids = []

    chunk_id = 0

    for pdf_file in pdf_files:
        print(f"📄 Processando: {pdf_file.name}")

        text = extract_text_from_pdf(pdf_file)

        if not text.strip():
            print(f"⚠️ Nenhum texto extraído de {pdf_file.name}")
            continue

        chunks = chunk_text(text)
        book_type = detect_book_type(pdf_file.name)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)

            metadatas.append({
                "level": level,
                "book": book_type,
                "source": pdf_file.name,
                "chunk_index": i
            })

            ids.append(f"{level}_{book_type}_{chunk_id}")
            chunk_id += 1

    if not all_chunks:
        raise ValueError("Nenhum chunk foi gerado.")

    collection.add(
        documents=all_chunks,
        metadatas=metadatas,
        ids=ids
    )

    print(f"✅ {len(all_chunks)} chunks salvos para {level}")

# ============================================================
# RETRIEVE
# ============================================================

def retrieve_context(query: str, level: str, k: int = 4):
    collection_name = f"sos_{level}"
    collection = client.get_collection(name=collection_name)

    results = collection.query(
        query_texts=[query],
        n_results=k,
        where={"level": level}
    )

    docs = results.get("documents", [[]])[0]

    return "\n\n".join(docs)