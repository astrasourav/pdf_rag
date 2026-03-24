"""
utils.py
--------
Utility helpers for the Multi-PDF RAG Streamlit app.

Currently contains:
    - save_uploaded_files()  : persists Streamlit UploadedFile objects to disk
    - clear_upload_dir()     : cleans up the temp folder between sessions
    - get_pdf_paths()        : returns all PDF paths currently in the upload dir

Usage in app.py:
    from utils import save_uploaded_files, clear_upload_dir

    pdf_paths = save_uploaded_files(uploaded_files)
    chunks    = load_and_chunk_pdfs(pdf_paths)
"""

import os
import shutil
from typing import List

import streamlit as st


# ─── Config ──────────────────────────────────────────────────────────────────
UPLOAD_DIR = "uploaded_pdfs"   # local folder where PDFs are saved


# ─── save_uploaded_files ─────────────────────────────────────────────────────
def save_uploaded_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[str]:
    """
    Save Streamlit UploadedFile objects to disk and return their file paths.

    Streamlit's UploadedFile is an in-memory buffer — it has no path on disk.
    LangChain loaders (PyMuPDFLoader etc.) need a real file path, so we
    write each file to a local folder first.

    Args:
        uploaded_files: List of files from st.file_uploader().

    Returns:
        List of absolute file paths for the saved PDFs.

    Example:
        uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        pdf_paths = save_uploaded_files(uploaded_files)
        # pdf_paths → ["uploaded_pdfs/paper1.pdf", "uploaded_pdfs/paper2.pdf"]
    """
    # Create the upload directory if it doesn't exist
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    saved_paths: List[str] = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

        # Write bytes to disk
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Reset buffer pointer so the file can be re-read if needed
        uploaded_file.seek(0)

        saved_paths.append(file_path)
        print(f"[INFO] Saved: {file_path}")

    return saved_paths


# ─── clear_upload_dir ────────────────────────────────────────────────────────
def clear_upload_dir() -> None:
    """
    Delete all files in the upload directory.
    Call this when the user starts a new session or clicks 'Clear'.

    Example:
        if st.button("Clear all"):
            clear_upload_dir()
    """
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
        print(f"[INFO] Cleared upload directory: {UPLOAD_DIR}")


# ─── get_pdf_paths ───────────────────────────────────────────────────────────
def get_pdf_paths() -> List[str]:
    """
    Return all PDF file paths currently saved in the upload directory.
    Useful for reloading paths after a Streamlit rerun without re-uploading.

    Returns:
        List of file paths ending in .pdf

    Example:
        paths = get_pdf_paths()
        # ["uploaded_pdfs/doc1.pdf", "uploaded_pdfs/doc2.pdf"]
    """
    if not os.path.exists(UPLOAD_DIR):
        return []

    return [
        os.path.join(UPLOAD_DIR, f)
        for f in os.listdir(UPLOAD_DIR)
        if f.lower().endswith(".pdf")
    ]


# ─── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Upload dir  :", UPLOAD_DIR)
    print("PDFs on disk:", get_pdf_paths())