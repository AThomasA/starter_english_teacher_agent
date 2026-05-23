# rag/md_indexer.py
# ============================================================
# INDEXADOR HIERÁRQUICO — lê .md do vault, indexa com LlamaIndex
# ============================================================
# Substitui llama_pipeline.py e rag_pipeline.py.
# Fonte de verdade: obsidian-vault/02-books/
#
# Hierarquia detectada nos .md:
#   # Título          → nível 1 (livro/nível)
#   ## Seção          → nível 2 (unidade, chapter, grammar...)
#   ### Subseção      → nível 3 (exercício, exemplo, vocabulário...)
#   texto puro        → chunks folha
#
# SETUP:
#   uv add llama-index
#   uv add llama-index-embeddings-huggingface
#   uv add llama-index-vector-stores-chroma
#   uv add llama-index-readers-file
# ============================================================

from __future__ import annotations

import json
import os
from pathlib import Path

import chromadb
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# ============================================================
# CONFIG
# ============================================================

VAULT_PATH   = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./obsidian-vault"))
BOOKS_PATH   = VAULT_PATH / "02-books"
CHROMA_PATH  = "rag/chroma_db"
GRAPH_PATH   = Path("rag/graph_data")
MODEL_CACHE  = Path("rag/model_cache")
EMBED_MODEL  = "all-MiniLM-L6-v2"

# Chunk sizes por nível hierárquico (em tokens estimados)
# chunk_sizes=[2048, 512, 128]:
#   2048 → nó pai (contexto amplo, não buscado diretamente)
#    512 → nó intermediário (seção)
#    128 → nó folha (o que vai pro LLM — ~100 tokens reais)
CHUNK_SIZES = [2048, 512, 128]

_embed_initialized = False
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)


def _init_embed_model() -> None:
    """Carrega o modelo de embeddings só quando for indexar (evita travar a UI)."""
    global _embed_initialized
    if _embed_initialized:
        return
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        cache_folder=str(MODEL_CACHE.resolve()),
    )
    Settings.llm = None
    _embed_initialized = True

# UI usa starter, level-1…; vault pode usar nivel-starter, nivel-1…
LEVEL_FOLDER_ALIASES: dict[str, str] = {
    "starter": "nivel-starter",
    "level-1": "nivel-1",
    "level-2": "nivel-2",
    "level-3": "nivel-3",
    "level-4": "nivel-4",
}


def _resolve_books_folder(level: str) -> Path:
    """Resolve pasta de livros no vault para o nível informado."""
    candidates = [level]
    if level in LEVEL_FOLDER_ALIASES:
        candidates.append(LEVEL_FOLDER_ALIASES[level])
    for name in candidates:
        path = BOOKS_PATH / name
        if path.exists():
            return path
    return BOOKS_PATH / level


# ============================================================
# LEITURA DOS .MD DO VAULT
# ============================================================

def _detect_book_type(filename: str) -> str:
    name = filename.lower()
    if "student" in name:
        return "student_book"
    if "workbook" in name or "work_book" in name:
        return "workbook"
    if "teacher" in name:
        return "teacher_book"
    return "general"


def load_md_documents(level: str) -> list[Document]:
    """
    Lê todos os .md de obsidian-vault/02-books/<level>/
    e retorna lista de Document do LlamaIndex com metadata rica.
    """
    level_path = _resolve_books_folder(level)

    if not level_path.exists():
        raise FileNotFoundError(
            f"Pasta não encontrada: {level_path}\n"
            f"Verifique se os .md foram colocados em obsidian-vault/02-books/{level}/"
        )

    md_files = sorted(level_path.rglob("*.md"))

    if not md_files:
        raise ValueError(f"Nenhum .md encontrado em {level_path}")

    documents: list[Document] = []

    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8").strip()
        if not text:
            continue

        # Metadata extraída do caminho do arquivo
        # Ex: 02-books/starter/student_book/unit-01.md
        relative = md_file.relative_to(BOOKS_PATH)
        parts    = relative.parts  # ('starter', 'student_book', 'unit-01.md')

        book_type = _detect_book_type(md_file.parent.name if len(parts) > 2 else md_file.name)
        unit_hint = md_file.stem  # nome do arquivo sem extensão

        doc = Document(
            text=text,
            metadata={
                "level":     level,
                "book":      book_type,
                "unit":      unit_hint,
                "source":    str(md_file.relative_to(VAULT_PATH)),
                "file_name": md_file.name,
            },
            excluded_embed_metadata_keys=["source", "file_name"],
            excluded_llm_metadata_keys=["source"],
        )
        documents.append(doc)

    print(f"  [{len(documents)}] arquivos .md carregados para nivel '{level}'")
    return documents


# ============================================================
# PARSER HIERÁRQUICO
# ============================================================

def parse_into_nodes(documents: list[Document]) -> tuple[list, list]:
    """
    Aplica HierarchicalNodeParser nos documentos.
    Retorna (all_nodes, leaf_nodes).

    HierarchicalNodeParser cria 3 camadas:
    - Nós pai (chunk_size=2048): contexto geral do documento
    - Nós intermediários (512): seções/unidades
    - Nós folha (128): chunks pequenos que são indexados e buscados
    """
    parser = HierarchicalNodeParser.from_defaults(chunk_sizes=CHUNK_SIZES)

    all_nodes  = parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(all_nodes)

    print(f"  [{len(leaf_nodes)}] nos folha | {len(all_nodes)} nos totais")
    return all_nodes, leaf_nodes


# ============================================================
# INDEXAÇÃO NO CHROMADB
# ============================================================

def index_nodes(
    all_nodes: list,
    leaf_nodes: list,
    level: str,
) -> VectorStoreIndex:
    """
    Indexa os nós folha no ChromaDB via LlamaIndex.
    Os nós pai ficam no storage para o AutoMergingRetriever usar depois.
    """
    _init_embed_model()
    collection_name = f"sos_{level}"

    # Limpa collection anterior
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    chroma_col   = chroma_client.create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_col)
    storage_ctx  = StorageContext.from_defaults(vector_store=vector_store)

    # Adiciona TODOS os nós ao storage (pai + folha)
    # O vector store indexa só os folha; os pai ficam disponíveis para merge
    storage_ctx.docstore.add_documents(all_nodes)

    # Cria índice a partir dos nós folha
    index = VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_ctx,
    )

    print(f"  OK: {len(leaf_nodes)} nos folha indexados em '{collection_name}'")
    return index


# ============================================================
# SALVA METADATA DO ÍNDICE (para o retriever carregar depois)
# ============================================================

def _save_index_metadata(level: str, stats: dict):
    GRAPH_PATH.mkdir(parents=True, exist_ok=True)
    meta_file = GRAPH_PATH / f"{level}_index_meta.json"
    meta_file.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_index_metadata(level: str) -> dict | None:
    meta_file = GRAPH_PATH / f"{level}_index_meta.json"
    if not meta_file.exists():
        return None
    return json.loads(meta_file.read_text(encoding="utf-8"))


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def run_indexing(level: str) -> dict:
    """
    Pipeline completo de indexação para um nível:
    1. Lê .md do vault
    2. Parseia hierarquicamente
    3. Indexa no ChromaDB
    4. Salva metadata

    Retorna stats da indexação.
    """
    print(f"\nIniciando indexacao para nivel '{level}'...")

    documents          = load_md_documents(level)
    all_nodes, leafs   = parse_into_nodes(documents)
    index              = index_nodes(all_nodes, leafs, level)

    stats = {
        "level":       level,
        "documents":   len(documents),
        "total_nodes": len(all_nodes),
        "leaf_nodes":  len(leafs),
        "chunk_sizes": CHUNK_SIZES,
        "collection":  f"sos_{level}",
        "embed_model": EMBED_MODEL,
    }

    _save_index_metadata(level, stats)

    print(f"\nIndexacao concluida para '{level}':")
    print(f"   Documentos .md:  {stats['documents']}")
    print(f"   Nós totais:      {stats['total_nodes']}")
    print(f"   Nós folha:       {stats['leaf_nodes']} (indexados)")

    return stats


# ============================================================
# UTILITÁRIO — info da collection
# ============================================================

def get_collection_info(level: str) -> dict:
    collection_name = f"sos_{level}"
    try:
        col = chroma_client.get_collection(collection_name)
        meta = get_index_metadata(level) or {}
        return {
            "exists":      True,
            "count":       col.count(),
            "name":        collection_name,
            "documents":   meta.get("documents", "?"),
            "leaf_nodes":  meta.get("leaf_nodes", "?"),
        }
    except Exception:
        return {"exists": False, "count": 0, "name": collection_name}