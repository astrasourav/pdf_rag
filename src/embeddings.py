"""
embeddings.py
-------------
Handles embedding generation and vector store management using
sentence-transformers (free, local) and ChromaDB.

Functions:
    - get_embedding_model()   : loads the sentence-transformer model
    - get_vectorstore()       : creates or loads a ChromaDB collection
    - add_chunks_to_vectorstore() : embeds chunks and stores in ChromaDB
    - load_vectorstore()      : loads an existing ChromaDB collection from disk

Usage in other modules:
    from embeddings import add_chunks_to_vectorstore, load_vectorstore

    # After ingestion:
    vectorstore = add_chunks_to_vectorstore(chunks)

    # Later (e.g. in retriever.py):
    vectorstore = load_vectorstore()
"""

import os
from typing import List, Optional

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from loguru import logger
import torch

# ─── Config ───────────────────────────────────────────────────────────────────
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # fast + free
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"   # better quality, slightly slower
COLLECTION_NAME      = "multi_pdf_rag"    # ChromaDB collection name


# ─── Step 1: Load Embedding Model ─────────────────────────────────────────────
def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    """
    Load a HuggingFace sentence-transformer embedding model.
    The model is downloaded once and cached locally by HuggingFace.

    Args:
        model_name: HuggingFace model ID. Defaults to all-MiniLM-L6-v2.
                    Other good free options:
                      - "BAAI/bge-small-en-v1.5"   (better quality, slightly slower)
                      - "BAAI/bge-base-en-v1.5"    (even better, larger)

    Returns:
        HuggingFaceEmbeddings object ready to use with LangChain.
    """
    logger.info(f"Loading embedding model: {model_name}")

    # Auto-detect — uses GPU if available, falls back to CPU silently
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    logger.info(f"Using device: {device}")

    # Batch sizes according to the device
    if device == "cuda":
        batch_size = 64
    else:
        batch_size = 32
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},   
        encode_kwargs={"normalize_embeddings": True,
                      "batch_size": batch_size
                      },  
    )

    logger.info("Embedding model loaded successfully.")
    return embeddings


# ─── Step 2: Create / Load Vector Store ───────────────────────────────────────
def get_vectorstore(
    embedding_model: Optional[HuggingFaceEmbeddings] = None,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """
    Create a new ChromaDB vectorstore or load one that already exists on disk.
    ChromaDB automatically persists to disk at persist_dir.

    Args:
        embedding_model:  HuggingFaceEmbeddings instance. Loaded fresh if None.
        persist_dir:      Folder path for ChromaDB storage.
        collection_name:  Name of the ChromaDB collection.

    Returns:
        Chroma vectorstore instance.
    """
    if embedding_model is None:
        embedding_model = get_embedding_model()

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
    )

    return vectorstore


# ─── Step 3: Embed Chunks and Store ───────────────────────────────────────────
def add_chunks_to_vectorstore(
    chunks: List[Document],
    embedding_model: Optional[HuggingFaceEmbeddings] = None,
    collection_name: str = COLLECTION_NAME,
    batch_size: int = 100,
) -> Chroma:
    """
    Embed document chunks and store them in ChromaDB.
    Chunks are processed in batches to avoid memory issues with large PDFs.

    Args:
        chunks:          List of Document chunks from ingestion.py.
        embedding_model: HuggingFaceEmbeddings instance. Loaded fresh if None.
        persist_dir:     Folder path for ChromaDB storage.
        collection_name: Name of the ChromaDB collection.
        batch_size:      Number of chunks to embed at once (default 100).

    Returns:
        Populated Chroma vectorstore.

    Example:
        from ingestion import load_and_chunk_pdfs
        from embeddings import add_chunks_to_vectorstore

        chunks      = load_and_chunk_pdfs(["data/paper.pdf"])
        vectorstore = add_chunks_to_vectorstore(chunks)
    """
    if embedding_model is None:
        embedding_model = get_embedding_model()

    logger.info(f"Embedding {len(chunks)} chunks in batches of {batch_size}...")

    # Process in batches
    vectorstore = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        if vectorstore is None:
            # First batch — create the collection
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embedding_model,
                collection_name=collection_name,
            )
        else:
            # Subsequent batches — add to existing collection
            vectorstore.add_documents(batch)

        logger.info (f"[INFO] Embedded batch {i // batch_size + 1} → {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

    return vectorstore


# ─── Step 4: Load Existing Vector Store ───────────────────────────────────────
def load_vectorstore(
    collection_name: str = COLLECTION_NAME,
    embedding_model: Optional[HuggingFaceEmbeddings] = None,
) -> Chroma:
    """
    Load an already-persisted ChromaDB vectorstore from disk.
    Use this in retriever.py instead of re-embedding every time.

    Args:
        persist_dir:      Folder where ChromaDB was saved.
        collection_name:  Name of the ChromaDB collection.
        embedding_model:  HuggingFaceEmbeddings instance. Loaded fresh if None.

    Returns:
        Chroma vectorstore loaded from disk.

    Raises:
        FileNotFoundError: If persist_dir does not exist.

    Example:
        vectorstore = load_vectorstore()
        results = vectorstore.similarity_search("What is RAG?", k=4)
    """
    if embedding_model is None:
        embedding_model = get_embedding_model()


    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
    )

    count = vectorstore._collection.count()
    logger.info(f"Loaded vectorstore with {count} stored chunks.")
    return vectorstore




