from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import pytesseract
from pdf2image import convert_from_path
import os

load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "sos_english")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\poppler-25.12.0\Library\bin"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def load_pdfs(data_dir: Path) -> list:
    documents = []
    pdf_files = list(data_dir.glob("*.pdf"))

    if not pdf_files:
        print("Nenhum PDF encontrado na pasta data/")
        return documents

    for pdf_path in pdf_files:
        print(f"Carregando: {pdf_path.name}")
        pages = convert_from_path(str(pdf_path), poppler_path=POPPLER_PATH)
        count = 0
        for page_num, page_image in enumerate(pages):
            text = pytesseract.image_to_string(page_image, lang="eng")
            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={"source": pdf_path.name, "page": page_num + 1}
                ))
                count += 1
        print(f"{count} paginas extraidas de {pdf_path.name}")

    return documents


def split_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", "."],
    )
    chunks = splitter.split_documents(documents)
    print(f"Total de chunks criados: {len(chunks)}")
    return chunks


def create_vectorstore(chunks: list):
    print(f"Carregando modelo de embeddings: {EMBEDDING_MODEL}")
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    print("Criando banco vetorial...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print(f"Banco vetorial criado em: {CHROMA_DIR}")
    return vectorstore


def main():
    print("Iniciando ingestao de dados...\n")

    documents = load_pdfs(DATA_DIR)
    if not documents:
        return

    print(f"Total de documentos: {len(documents)}")
    print(f"Amostra do primeiro documento:")
    print(repr(documents[0].page_content[:300]))

    chunks = split_documents(documents)

    if not chunks:
        print("Nenhum chunk criado.")
        return

    create_vectorstore(chunks)
    print("Ingestao concluida com sucesso!")


if __name__ == "__main__":
    main()