"""
retriever.py
------------
Handles document retrieval from ChromaDB.
Supports 3 retrieval modes — all pluggable into chain.py:

    1. Semantic Search     — pure vector similarity (default)
    2. Hybrid Search       — BM25 keyword + semantic, combined via EnsembleRetriever
    3. MMR Reranking       — Maximal Marginal Relevance (reduces redundant chunks)

Usage in chain.py:
    from retriever import get_retriever

    retriever = get_retriever(vectorstore, mode="hybrid", k=4)
"""

from typing import List, Literal

from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from loguru import logger


# ─── Types ────────────────────────────────────────────────────────────────────
RetrievalMode = Literal["semantic", "hybrid", "mmr"]


# ─── Step 1: Semantic Retriever ───────────────────────────────────────────────
def get_semantic_retriever(
    vectorstore: Chroma,
    k: int = 4,
) -> VectorStoreRetriever:
    """
    Pure vector similarity search retriever.
    Finds the top-k chunks whose embeddings are closest to the query embedding.
    Fast and works well for conceptual / meaning-based questions.

    Args:
        vectorstore: Populated Chroma vectorstore from embeddings.py.
        k:           Number of chunks to retrieve (default 4).

    Returns:
        LangChain VectorStoreRetriever.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    print(f"[INFO] Semantic retriever ready (top-k={k})")
    return retriever


# ─── Step 2: BM25 Retriever ───────────────────────────────────────────────────
def get_bm25_retriever(
    chunks: List[Document],
    k: int = 4,
) -> BM25Retriever:
    """
    BM25 keyword-based retriever (like classic search engines).
    Works well for exact keyword matches, names, IDs, technical terms.
    Does NOT understand meaning — complementary to semantic search.

    Args:
        chunks: The same list of Document chunks used during ingestion.
                BM25 indexes them in memory (no vector DB needed).
        k:      Number of chunks to retrieve (default 4).

    Returns:
        BM25Retriever instance.

    Note:
        BM25 works on raw text so it needs the original chunks, not the
        vectorstore. Always pass the same chunks from ingestion.py.
    """
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = k
    print(f"[INFO] BM25 retriever ready (top-k={k}, docs={len(chunks)})")
    return retriever


# ─── Step 3: Hybrid Retriever (BM25 + Semantic) ───────────────────────────────
def get_hybrid_retriever(
    vectorstore: Chroma,
    chunks: List[Document],
    k: int = 4,
    semantic_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> EnsembleRetriever:
    """
    Hybrid retriever combining BM25 keyword search + semantic vector search.
    Uses LangChain's EnsembleRetriever to merge and rerank results from both.

    Why hybrid?
        - Semantic alone misses exact keyword matches (names, codes, IDs)
        - BM25 alone misses conceptual / paraphrased questions
        - Together they cover both — this is the industry standard approach

    Args:
        vectorstore:     Populated Chroma vectorstore from embeddings.py.
        chunks:          Original Document chunks from ingestion.py (for BM25).
        k:               Number of chunks each retriever fetches before merging.
        semantic_weight: Weight for semantic results (default 0.6).
        bm25_weight:     Weight for BM25 results (default 0.4).
                         Note: weights must sum to 1.0

    Returns:
        EnsembleRetriever that merges and reranks results from both retrievers.

    Example:
        retriever = get_hybrid_retriever(vectorstore, chunks, k=4)
        docs = retriever.invoke("What is the revenue for Q3?")
    """
    if round(semantic_weight + bm25_weight, 5) != 1.0:
        raise ValueError(
            f"[ERROR] Weights must sum to 1.0. "
            f"Got semantic={semantic_weight} + bm25={bm25_weight} = {semantic_weight + bm25_weight}"
        )

    semantic_retriever = get_semantic_retriever(vectorstore, k=k)
    bm25_retriever     = get_bm25_retriever(chunks, k=k)

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[bm25_weight, semantic_weight],
    )

    print(f"[INFO] Hybrid retriever ready (BM25={bm25_weight} + Semantic={semantic_weight}, k={k})")
    return hybrid_retriever


# ─── Step 4: MMR Retriever ────────────────────────────────────────────────────
def get_mmr_retriever(
    vectorstore: Chroma,
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
) -> VectorStoreRetriever:
    """
    MMR (Maximal Marginal Relevance) retriever.
    Balances relevance vs diversity — avoids returning 4 nearly identical chunks.

    How MMR works:
        1. Fetches `fetch_k` candidates via similarity search
        2. Picks chunks that are relevant BUT not too similar to each other
        3. `lambda_mult` controls the tradeoff:
             1.0 = pure relevance (like semantic search)
             0.0 = pure diversity
             0.5 = balanced (default, recommended)

    Best for: Long documents where the same content repeats across pages.

    Args:
        vectorstore:  Populated Chroma vectorstore from embeddings.py.
        k:            Number of final chunks to return (default 4).
        fetch_k:      Candidate pool size before MMR reranking (default 20).
        lambda_mult:  Relevance vs diversity tradeoff (default 0.5).

    Returns:
        LangChain VectorStoreRetriever with MMR search type.
    """
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
        },
    )
    print(f"[INFO] MMR retriever ready (k={k}, fetch_k={fetch_k}, lambda={lambda_mult})")
    return retriever


# ─── Main: get_retriever() — single entry point for chain.py ─────────────────
def get_retriever(
    vectorstore: Chroma,
    chunks: List[Document] = None,
    mode: RetrievalMode = "hybrid",
    k: int = 4,
    semantic_weight: float = 0.6,
    bm25_weight: float = 0.4,
    mmr_fetch_k: int = 20,
    mmr_lambda: float = 0.5,
):
    """
    Single entry point for all retrieval modes.
    Import only this function in chain.py and app.py.

    Args:
        vectorstore:     Populated Chroma vectorstore.
        chunks:          Original Document chunks — REQUIRED for hybrid mode.
        mode:            "semantic" | "hybrid" | "mmr"  (default: "hybrid")
        k:               Top-k chunks to retrieve.
        semantic_weight: Weight for semantic in hybrid mode (default 0.6).
        bm25_weight:     Weight for BM25 in hybrid mode (default 0.4).
        mmr_fetch_k:     Candidate pool for MMR (default 20).
        mmr_lambda:      Relevance/diversity tradeoff for MMR (default 0.5).

    Returns:
        A LangChain retriever — compatible with any LCEL chain.

    Example:
        retriever = get_retriever(vectorstore, chunks, mode="hybrid", k=4)
        docs = retriever.invoke("What are the key findings?")
    """
    if mode == "semantic":
        return get_semantic_retriever(vectorstore, k=k)

    elif mode == "hybrid":
        if chunks is None:
            raise ValueError(
                "[ERROR] 'chunks' must be provided for hybrid retrieval mode. "
                "Pass the same chunks list returned by load_and_chunk_pdfs()."
            )
        return get_hybrid_retriever(
            vectorstore, chunks, k=k,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight,
        )

    elif mode == "mmr":
        return get_mmr_retriever(
            vectorstore, k=k,
            fetch_k=mmr_fetch_k,
            lambda_mult=mmr_lambda,
        )

    else:
        raise ValueError(
            f"[ERROR] Unknown retrieval mode: '{mode}'. "
            f"Choose from: 'semantic', 'hybrid', 'mmr'."
        )



