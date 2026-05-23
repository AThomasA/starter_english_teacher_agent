"""

# rag/rag_pipeline.py
# ============================================================
# PIPELINE RAG — Ingestão e Recuperação
# ============================================================

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
EMBED_MODEL  = "all-MiniLM-L6-v2"

# Descomente se necessário no Windows:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

client = chromadb.PersistentClient(path=CHROMA_PATH)


# ============================================================
# EXTRAÇÃO DE TEXTO (OCR + TEXTO NORMAL)
# ============================================================

def extract_text_from_pdf(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""

    for page in doc:
        text = page.get_text()

        if not text.strip():
            # Página sem texto → OCR
            pix = page.get_pixmap(dpi=200)  # 200 DPI suficiente para OCR
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(img, lang="eng")
            full_text += ocr_text + "\n"
        else:
            full_text += text + "\n"

    return full_text


# ============================================================
# CHUNKING
# ============================================================
# FASE A: chunk_size=800 chars ≈ 200 tokens (menor = mais preciso na busca)
# overlap=80 chars para não perder contexto entre chunks
# ============================================================

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 80) -> list[str]:
    """
    Divide texto em chunks com overlap.
    800 chars ≈ 200 tokens por chunk.
    """
    # Remove linhas em branco excessivas
    text = "\n".join(line for line in text.splitlines() if line.strip())

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
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

def run_ingestion(level: str) -> int:
    """
    Processa todos os PDFs de uma pasta de nível e indexa no ChromaDB.
    Retorna o total de chunks gerados.
    """
    base_path = Path("rag/data/pdfs") / level

    if not base_path.exists():
        raise ValueError(f"Pasta não encontrada: {base_path}")

    pdf_files = list(base_path.glob("*.pdf"))

    print("Arquivos encontrados:")
    for f in pdf_files:
        print(f"  - {f.name}")

    if not pdf_files:
        raise ValueError(f"Nenhum PDF encontrado em {base_path}")

    collection_name = f"sos_{level}"

    # Recria a collection (limpa dados anteriores)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_function,
    )

    all_chunks: list[str] = []
    metadatas:  list[dict] = []
    ids:        list[str]  = []
    chunk_id = 0

    for pdf_file in pdf_files:
        print(f"📄 Processando: {pdf_file.name}")

        text = extract_text_from_pdf(pdf_file)

        if not text.strip():
            print(f"  ⚠️  Nenhum texto extraído de {pdf_file.name}")
            continue

        chunks    = chunk_text(text)
        book_type = detect_book_type(pdf_file.name)
        print(f"  ✅ {len(chunks)} chunks gerados")

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadatas.append({
                "level":       level,
                "book":        book_type,
                "source":      pdf_file.name,
                "chunk_index": i,
            })
            ids.append(f"{level}_{book_type}_{chunk_id}")
            chunk_id += 1

    if not all_chunks:
        raise ValueError("Nenhum chunk foi gerado. Verifique os PDFs.")

    # Insere em lotes de 100 para evitar problemas de memória
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        collection.add(
            documents=all_chunks[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
            ids=ids[i : i + batch_size],
        )

    print(f"\n✅ Total: {len(all_chunks)} chunks salvos para nível '{level}'")
    return len(all_chunks)


# ============================================================
# RETRIEVE
# ============================================================
# FASE A:
# k=2 por padrão (era 4) → ~400 tokens de contexto RAG por mensagem
# threshold: descarta chunks com distância > 1.2 (baixa relevância)
# ============================================================

def retrieve_context(query: str, level: str, k: int = 2) -> str:
    """
    Recupera os chunks mais relevantes para a query.
    k=2 por padrão → ~400 tokens de contexto RAG por mensagem.
    Chunks com distância > threshold são descartados.
    """
    collection_name = f"sos_{level}"

    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_function,
        )
    except Exception:
        return ""

    results = collection.query(
        query_texts=[query],
        n_results=k,
        where={"level": level},
    )

    docs      = results.get("documents", [[]])[0]
    distances = results.get("distances",  [[]])[0]

    # Filtra por relevância — distância menor = mais relevante
    # < 1.0: boa relevância | > 1.2: provavelmente irrelevante
    threshold = float(os.getenv("RAG_DISTANCE_THRESHOLD", "1.2"))

    filtered = [
        doc for doc, dist in zip(docs, distances)
        if dist < threshold
    ]

    if not filtered:
        return ""

    return "\n\n---\n\n".join(filtered)


# ============================================================
# UTILITÁRIO — INFO DA COLLECTION
# ============================================================

def get_collection_info(level: str) -> dict:
    """Retorna informações sobre a collection de um nível."""
    collection_name = f"sos_{level}"
    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_function,
        )
        return {"exists": True, "count": collection.count(), "name": collection_name}
    except Exception:
        return {"exists": False, "count": 0, "name": collection_name}


"""