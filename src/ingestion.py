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

import pdfplumber
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


# ─── Constants (easy to override) ────────────────────────────────────────────
DEFAULT_CHUNK_SIZE    = 1000
DEFAULT_CHUNK_OVERLAP = 200
# DEFAULT_SEPARATORS    = ["\n\n", "\n", ".", " "]
DEFAULT_SEPARATORS = [
    "\n\n",           # paragraph breaks
    "\n",             # line breaks
    "[TABLE]",        # our table marker — keep tables as their own chunks
    ".",
    " ",
]


# ─── Step 1: Load PDFs ────────────────────────────────────────────────────────
def load_pdfs(pdf_paths: List[str]) -> List[Document]:
    all_docs = []

    for path in pdf_paths:
        if not os.path.exists(path):
            print(f"[WARNING] File not found, skipping: {path}")
            continue

        filename = os.path.basename(path)

        # Try pdfplumber first
        print(f"[INFO] Attempting pdfplumber for {filename}...")
        docs = _load_with_pdfplumber(path, filename)

        # Fallback to PyMuPDF
        if not docs:
            print(f"[INFO] pdfplumber returned 0 docs — falling back to PyMuPDF for {filename}...")
            docs = _load_with_pymupdf(path, filename)

        # Final fallback — OCR for scanned/image-based PDFs  ← was missing!
        if not docs:
            print(f"[INFO] PyMuPDF also empty — trying OCR for {filename}...")
            docs = _load_with_ocr_fallback(path, filename)

        if not docs:
            print(f"[ERROR] All extractors failed for {filename}. PDF may be corrupted.")
        else:
            all_docs.extend(docs)
            print(f"[INFO] Loaded '{filename}' → {len(docs)} pages")

    print(f"[INFO] Total pages gathered: {len(all_docs)}")
    return all_docs


def _load_with_pdfplumber(path: str, filename: str) -> List[Document]:
    """
    Load PDF using pdfplumber.
    Handles:
        - Multi-column layouts (CVs)
        - Tables → converts to readable markdown-style text
        - Mixed text + table pages
    """
    docs = []

    try:
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = ""

                # ── Extract tables first ──────────────────────────────────
                tables = page.extract_tables()
                table_texts = []

                for table in tables:
                    if not table:
                        continue
                    # Convert table rows to readable pipe-separated text
                    formatted_rows = []
                    for row in table:
                        clean_row = [str(cell).strip() if cell else "" for cell in row]
                        formatted_rows.append(" | ".join(clean_row))
                    table_texts.append("\n".join(formatted_rows))

                # ── Extract plain text (excluding table areas) ────────────
                # filter_table_areas removes table bounding boxes from text extraction
                # so we don't get duplicate content
                if tables:
                    table_areas = [page.find_tables()[i].bbox for i in range(len(tables))]
                    filtered = page
                    for bbox in table_areas:
                        try:
                            filtered = filtered.filter(
                                lambda obj: not _in_bbox(obj, bbox)
                            )
                        except Exception:
                            pass
                    plain_text = filtered.extract_text() or ""
                else:
                    plain_text = page.extract_text() or ""

                # ── Combine plain text + tables ───────────────────────────
                if table_texts:
                    page_text = plain_text + "\n\n[TABLE]\n" + "\n\n[TABLE]\n".join(table_texts)
                else:
                    page_text = plain_text

                if page_text.strip():
                    docs.append(Document(
                        page_content=page_text.strip(),
                        metadata={
                            "source":   filename,
                            "filepath": path,
                            "page":     page_num,
                        }
                    ))

    except Exception as e:
        print(f"[WARN] pdfplumber failed for {filename}: {e}")
        return []

    return docs


def _load_with_pymupdf(path: str, filename: str) -> List[Document]:
    """Fallback loader using PyMuPDF for simple text PDFs."""
    try:
        loader = PyMuPDFLoader(path)
        docs   = loader.load()
        for doc in docs:
            doc.metadata["source"]   = filename
            doc.metadata["filepath"] = path
            doc.metadata["page"]     = doc.metadata.get("page", 0) + 1
        return docs
    except Exception as e:
        print(f"[WARN] PyMuPDF also failed for {filename}: {e}")
        return []


def _in_bbox(obj, bbox) -> bool:
    """Check if a PDF object falls within a bounding box."""
    try:
        x0, y0, x1, y1 = bbox
        ox = obj.get("x0", 0)
        oy = obj.get("top", 0)
        # Handle cases where obj is None or missing keys
        if ox is None: ox = 0
        if oy is None: oy = 0
        return x0 <= ox <= x1 and y0 <= oy <= y1
    except Exception:
        return False


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

