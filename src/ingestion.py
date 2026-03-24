"""
ingestion.py
------------
Handles PDF loading, text splitting, and chunk preparation.
Returns chunks ready to be embedded and stored in a vector DB.

Usage:
    from ingestion import load_and_chunk_pdfs

    chunks = load_and_chunk_pdfs(
        pdf_paths=["data/paper1.pdf", "data/paper2.pdf"],
        chunk_size=1000,
        chunk_overlap=200,
    )
"""

import os
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


# ─── Constants (easy to override) ────────────────────────────────────────────
DEFAULT_CHUNK_SIZE    = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_SEPARATORS    = ["\n\n", "\n", ".", " "]


# ─── Step 1: Load PDFs ────────────────────────────────────────────────────────
def load_pdfs(pdf_paths: List[str]) -> List[Document]:
    """
    Load one or more PDFs and return a flat list of LangChain Documents.
    Each document page gets metadata: source filename + page number.

    Args:
        pdf_paths: List of file paths to PDF files.

    Returns:
        List of Document objects (one per page across all PDFs).
    """
    all_docs: List[Document] = []

    for path in pdf_paths:
        if not os.path.exists(path):
            print(f"[WARNING] File not found, skipping: {path}")
            logger.error(f"File not found, skipping: {path}")
            continue

        loader = PyMuPDFLoader(path)
        docs   = loader.load()  # returns one Document per page

        # Enrich metadata with a clean filename for citations in the UI
        filename = os.path.basename(path)
        for doc in docs:
            doc.metadata["source"]   = filename
            doc.metadata["filepath"] = path
            # PyMuPDFLoader already adds 'page' (0-indexed), make it 1-indexed
            doc.metadata["page"] = doc.metadata.get("page", 0) + 1

        all_docs.extend(docs)
        logger.info(f"Loaded '{filename}' → {len(docs)} pages")

    logger.info(f"Total pages loaded: {len(all_docs)}")
    return all_docs


# ─── Step 2: Split into Chunks ───────────────────────────────────────────────
def split_documents(
    docs: List[Document],
    chunk_size: int    = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split a list of Documents into smaller chunks for embedding.
    Metadata (source, page) is preserved in every chunk automatically
    by LangChain's text splitter.

    Args:
        docs:          List of Documents from load_pdfs().
        chunk_size:    Max number of characters per chunk.
        chunk_overlap: Number of characters to overlap between chunks.

    Returns:
        List of chunk Documents ready for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=DEFAULT_SEPARATORS,
        length_function=len,          # character-based; swap for token-based later
    )

    chunks = splitter.split_documents(docs)
    print(f"[INFO] Total chunks created: {len(chunks)}")
    return chunks


# ─── Step 3: Main Pipeline Function ──────────────────────────────────────────
def load_and_chunk_pdfs(
    pdf_paths: List[str],
    chunk_size: int    = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Full ingestion pipeline: load PDFs → split into chunks.
    This is the only function other modules need to import.

    Args:
        pdf_paths:     List of paths to PDF files.
        chunk_size:    Characters per chunk (default 1000).
        chunk_overlap: Overlap between chunks (default 200).

    Returns:
        List of chunk Documents with metadata (source, page, filepath).

    Example:
        chunks = load_and_chunk_pdfs(["data/report.pdf"], chunk_size=500)
        print(chunks[0].page_content)     # chunk text
        print(chunks[0].metadata)         # {'source': 'report.pdf', 'page': 3, ...}
    """
    docs   = load_pdfs(pdf_paths)
    chunks = split_documents(docs, chunk_size, chunk_overlap)
    return chunks


# ─── Quick Test (run this file directly to verify) ────────────────────────────
if __name__ == "__main__":
    import sys

    # Pass PDF paths as CLI args: python ingestion.py data/a.pdf data/b.pdf
    paths = sys.argv[1:] if len(sys.argv) > 1 else []

    if not paths:
        print("Usage: python ingestion.py path/to/file1.pdf path/to/file2.pdf")
        print("\nRunning with a dummy text file for demo purposes...")

        # Create a tiny dummy file so you can test without a real PDF
        os.makedirs("data", exist_ok=True)
        with open("data/sample.txt", "w") as f:
            f.write("LangChain is a framework for building LLM applications.\n\n")
            f.write("RAG stands for Retrieval Augmented Generation.\n\n")
            f.write("ChromaDB is a vector database for storing embeddings.\n\n")

        # Use TextLoader for the dummy .txt test
        from langchain_community.document_loaders import TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        loader = TextLoader("data/sample.txt")
        docs   = loader.load()
        docs[0].metadata["source"] = "sample.txt"
        docs[0].metadata["page"]   = 1

        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        chunks   = splitter.split_documents(docs)

    else:
        chunks = load_and_chunk_pdfs(paths)

    print(f"\n{'─'*50}")
    print(f"Total chunks: {len(chunks)}")
    print(f"\nSample chunk [0]:")
    print(f"  Content : {chunks[0].page_content[:120]}...")
    print(f"  Metadata: {chunks[0].metadata}")