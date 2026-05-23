# rag/md_retriever.py
# ============================================================
# RETRIEVER HIERÁRQUICO — lê do ChromaDB indexado pelo md_indexer
# ============================================================
# Usa AutoMergingRetriever do LlamaIndex:
#   - Busca nos nós folha (128 tokens cada)
#   - Se vários folhos do mesmo pai baterem → promove para o pai (512)
#   - Resultado: contexto mais coeso e ainda compacto
#
# Mesma assinatura de retrieve_context(query, level, k) —
# o agent.py não precisa de nenhuma mudança além do import.
# ============================================================

from __future__ import annotations

import os
from pathlib import Path

import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore

from rag.md_indexer import CHROMA_PATH, _init_embed_model

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# Cache em memória dos retrievers já construídos (evita recriar a cada query)
_retriever_cache: dict[str, AutoMergingRetriever] = {}


# ============================================================
# CONSTRÓI O RETRIEVER
# ============================================================

def _build_retriever(level: str, k: int) -> AutoMergingRetriever:
    """
    Reconstrói o VectorStoreIndex + AutoMergingRetriever a partir
    do ChromaDB já indexado. Cacheia em memória.
    """
    _init_embed_model()
    collection_name = f"sos_{level}"

    try:
        chroma_col = chroma_client.get_collection(collection_name)
    except Exception:
        raise ValueError(
            f"Collection '{collection_name}' não encontrada.\n"
            f"Execute run_indexing('{level}') primeiro no Streamlit."
        )

    vector_store = ChromaVectorStore(chroma_collection=chroma_col)

    # Reconstrói o storage context com docstore em memória
    # (os nós pai foram adicionados pelo md_indexer via storage_ctx.docstore)
    # Nota: SimpleDocumentStore em memória é recriado — os nós pai
    # precisam ser recarregados do disco se quiser persistência total.
    # Para Fase B isso é suficiente: os nós pai ficam em memória por sessão.
    storage_ctx = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=SimpleDocumentStore(),
    )

    # Reconstrói índice a partir do vector store existente
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_ctx,
    )

    # Retriever base: busca os k nós folha mais similares
    base_retriever = index.as_retriever(similarity_top_k=k)

    # AutoMergingRetriever: se ≥50% dos filhos de um nó pai baterem,
    # substitui pelos filhos e retorna o pai (mais contexto, menos ruído)
    auto_retriever = AutoMergingRetriever(
        base_retriever,
        storage_context=storage_ctx,
        verbose=False,
    )

    return auto_retriever


def _get_retriever(level: str, k: int) -> AutoMergingRetriever:
    """Retorna retriever cacheado ou cria um novo."""
    cache_key = f"{level}_{k}"
    if cache_key not in _retriever_cache:
        _retriever_cache[cache_key] = _build_retriever(level, k)
    return _retriever_cache[cache_key]


def invalidate_cache(level: str | None = None):
    """Invalida cache após reindexação."""
    global _retriever_cache
    if level:
        keys = [k for k in _retriever_cache if k.startswith(level)]
        for k in keys:
            del _retriever_cache[k]
    else:
        _retriever_cache = {}


# ============================================================
# FUNÇÃO PRINCIPAL — mesma assinatura do rag_pipeline
# ============================================================

def retrieve_context(query: str, level: str, k: int = 2) -> str:
    """
    Recupera contexto relevante para a query.
    Mesma assinatura de rag_pipeline.retrieve_context() —
    agent.py só precisa trocar o import.

    Retorna string formatada com os trechos mais relevantes,
    com header indicando a origem (livro, seção).
    """
    try:
        retriever = _get_retriever(level, k)
    except ValueError:
        return ""
    except Exception:
        return ""

    try:
        nodes = retriever.retrieve(query)
    except Exception:
        return ""

    if not nodes:
        return ""

    # Filtra por score mínimo de relevância
    threshold = float(os.getenv("RAG_SCORE_THRESHOLD", "0.3"))
    filtered  = [n for n in nodes if (n.score or 0) >= threshold]

    if not filtered:
        filtered = nodes[:1]  # garante pelo menos 1 resultado

    # Formata o contexto com header de origem
    parts: list[str] = []
    for node in filtered:
        meta = node.metadata or {}
        header = (
            f"[{meta.get('book', '').replace('_', ' ').title()} | "
            f"{meta.get('unit', '').replace('-', ' ').title()} | "
            f"Level: {meta.get('level', '')}]"
        )
        parts.append(f"{header}\n{node.text}")

    return "\n\n---\n\n".join(parts)


# ============================================================
# UTILITÁRIO — debug de uma query (uso no terminal)
# ============================================================

def debug_query(query: str, level: str, k: int = 3) -> None:
    """
    Imprime os nós retornados para uma query com scores.
    Útil para entender o que o retriever está buscando.
    """
    try:
        retriever = _get_retriever(level, k)
        nodes = retriever.retrieve(query)
    except Exception as e:
        print(f"Erro: {e}")
        return

    print(f"\n🔍 Query: '{query}' | Level: {level}")
    print(f"{'─'*60}")

    for i, node in enumerate(nodes, 1):
        meta  = node.metadata or {}
        score = node.score or 0
        print(f"\n[{i}] Score: {score:.3f}")
        print(f"     Livro: {meta.get('book', '?')} | Unidade: {meta.get('unit', '?')}")
        print(f"     Texto: {node.text[:200]}...")

    print(f"\n{'─'*60}")
    print(f"Total: {len(nodes)} nós retornados")