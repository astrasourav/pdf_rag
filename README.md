# 🧠 DocMind — Multi-PDF RAG Assistant

A production-grade Retrieval Augmented Generation (RAG) application that lets you upload multiple PDFs and have intelligent, cited, multi-turn conversations with your documents. Built with LangChain, Groq, ChromaDB, and Streamlit — evaluated using RAGAS.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📸 Demo

> Upload PDFs → Ingest & Index → Ask anything → Get cited, grounded answers

**Live Demo:** [app](https://souravpdf1.streamlit.app/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [RAG Pipeline](#rag-pipeline)
- [RAGAS Evaluation](#ragas-evaluation)
- [Chunking Strategy](#chunking-strategy)
- [Retrieval Modes](#retrieval-modes)
- [Deployment](#deployment)
- [Known Limitations](#known-limitations)
- [Future Improvements](#future-improvements)

---

## Overview

DocMind is a **Multi-PDF RAG (Retrieval Augmented Generation)** assistant that solves a common real-world problem: extracting information from large document collections and CV's through natural conversation.

Traditional PDF tools require you to manually search, scroll, and read. DocMind lets you simply ask questions and get precise, grounded answers — with the exact source page cited.

### What makes this different from a simple chatbot?

A regular chatbot hallucinates — it generates answers from its training data regardless of what's in your documents. DocMind uses RAG to **strictly ground every answer in your uploaded documents**. If the answer isn't in the PDFs, the system says so rather than making something up.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                    │
│                                                             │
│  PDF Files → pdfplumber/PyMuPDF → Text Chunks → Embeddings │
│                                        ↓                    │
│                                   ChromaDB                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                        QUERY PIPELINE                        │
│                                                             │
│  User Question → Contextualize → Retrieve Chunks           │
│                                        ↓                    │
│                              Groq LLM (Llama 3)            │
│                                        ↓                    │
│                          Grounded Answer + Sources         │
└─────────────────────────────────────────────────────────────┘
```

### Detailed Flow

```
1. User uploads PDF(s)
         ↓
2. pdfplumber extracts text (handles tables, multi-column)
   └─ Falls back to PyMuPDF for simple text PDFs
         ↓
3. RecursiveCharacterTextSplitter chunks the text
   └─ Configurable chunk_size and chunk_overlap
         ↓
4. sentence-transformers/all-MiniLM-L6-v2 embeds each chunk
         ↓
5. ChromaDB stores embeddings + metadata (source, page)
         ↓
6. User asks a question
         ↓
7. History-aware retriever rewrites follow-up questions
   └─ "Can you elaborate?" → "Can you elaborate on the revenue findings?"
         ↓
8. Retriever fetches top-k most relevant chunks
   └─ Semantic / Hybrid (BM25 + Semantic) / MMR
         ↓
9. Groq LLM generates grounded answer using retrieved context
         ↓
10. Answer displayed in chat with source citation
```

---

## Features

### Core Features
- **Multi-PDF ingestion** — upload and index multiple PDFs simultaneously
- **Multi-turn conversation** — the system remembers previous questions in the same session
- **Grounded answers** — LLM is strictly instructed to only use document context
- **Smart PDF parsing** — handles tables, multi-column layouts via pdfplumber with PyMuPDF fallback
- **Configurable pipeline** — chunk size, overlap, embedding model, retrieval mode, top-k all adjustable

### Retrieval
- **Semantic Search** — pure vector similarity using sentence-transformers
- **Hybrid Search** — BM25 keyword search + semantic search combined via EnsembleRetriever
- **MMR Reranking** — Maximal Marginal Relevance to reduce redundant retrieved chunks

### Evaluation
- **RAGAS integration** — evaluate pipeline quality without human labels
- **4 metrics tracked** — Faithfulness, Answer Relevancy, Context Precision, Context Recall
- **Synthetic dataset generation** — auto-generate evaluation Q&A pairs from your documents
- **Visual results** — metric cards and bar charts in the UI

### UI
- Clean, modern Streamlit interface
- Real-time stat cards (PDFs loaded, chunks indexed, queries asked)
- Chat interface using `st.chat_message()` — native markdown rendering
- Pipeline settings panel in sidebar

---

## Project Structure

```
docmind/
│
├── app.py                    # Streamlit UI — only calls methods, no logic
│
├── src/
│   ├── ingestion.py          # PDF loading (pdfplumber + PyMuPDF), chunking
│   ├── embeddings.py         # Embedding model, ChromaDB vectorstore management
│   ├── retriever.py          # Semantic, Hybrid (BM25), MMR retrievers
│   ├── chain.py              # Groq LLM, RAG chain, conversation memory
│   ├── evaluate.py           # RAGAS evaluation pipeline
│   └── utils.py              # File saving, upload dir management
│
├── uploaded_pdfs/            # Temp storage for uploaded PDFs (gitignored)
├── chroma_db/                # ChromaDB persistence directory (gitignored)
│
├── requirements.txt          # All dependencies
├── .env                      # API keys (gitignored)
├── .gitignore
└── README.md
```

### Module Responsibilities

| Module | Responsibility | Key Functions |
|---|---|---|
| `ingestion.py` | Load PDFs, split into chunks | `load_and_chunk_pdfs()` |
| `embeddings.py` | Generate embeddings, manage ChromaDB | `add_chunks_to_vectorstore()`, `load_vectorstore()` |
| `retriever.py` | Fetch relevant chunks from vectorstore | `get_retriever()` |
| `chain.py` | Wire LLM + retriever, manage memory | `build_conversational_chain()`, `ask()` |
| `evaluate.py` | Run RAGAS evaluation | `evaluate_pipeline()` |
| `utils.py` | File system utilities | `save_uploaded_files()`, `clear_upload_dir()` |
| `app.py` | Streamlit UI — calls all modules | Pure UI layer |

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM** | Groq (Llama 3.1 8B Instant) | Fast, free inference |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 | Local, free, no API key |
| **Vector Store** | ChromaDB | Local persistent vector database |
| **RAG Framework** | LangChain (LCEL) | Chain orchestration |
| **PDF Parsing** | pdfplumber + PyMuPDF | Dual-loader for complex layouts |
| **Hybrid Search** | rank-bm25 + EnsembleRetriever | BM25 + semantic combined |
| **Evaluation** | RAGAS | Pipeline quality measurement |
| **UI** | Streamlit | Web interface |

---

## Installation

### Prerequisites

- Python 3.10+
- A free Groq API key from [console.groq.com](https://console.groq.com)

### Step 1 — Clone the repository

```bash
git clone https://github.com/astrasourav/pdf_rag
cd docmind
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Set up environment variables

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

Get your free key at [console.groq.com](https://console.groq.com).

### Step 5 — Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## Configuration

All pipeline settings are configurable from the sidebar UI at runtime:

| Setting | Default | Description |
|---|---|---|
| Chunk Size | 512 | Max characters per chunk. Smaller = more focused retrieval |
| Chunk Overlap | 64 | Characters shared between adjacent chunks. Prevents boundary cuts |
| Embedding Model | all-MiniLM-L6-v2 | Model used to convert chunks to vectors |
| Retrieval Mode | Hybrid | How chunks are fetched — Semantic / Hybrid / MMR |
| Top-K | 4 | Number of chunks retrieved per query |

---

## Usage

### Basic Usage

```
1. Open the app at localhost:8501
2. Upload one or more PDFs using the sidebar uploader
3. Adjust pipeline settings if needed (or leave defaults)
4. Click "🚀 Ingest & Index PDFs"
5. Wait for indexing to complete
6. Type your question in the chat input and press Send
```

### Tips for Better Results

- **Be specific** — "What is the revenue for Q3 2023?" works better than "tell me about revenue"
- **Use Hybrid mode** for documents with proper nouns, names, or specific terms
- **Use Semantic mode** for conceptual questions like "explain the methodology"
- **Increase Top-K** to 6-8 if answers feel incomplete
- **Reduce chunk size** to 200-300 for short, dense documents

### Multi-turn Conversation

The system maintains conversation memory within a session. You can ask follow-up questions naturally:

```
User:  "What are the key findings?"
AI:    "The study found three main things: ..."

User:  "Can you elaborate on the second point?"
AI:    [understands "second point" refers to previous answer] "The second finding..."

User:  "What methodology was used to reach that conclusion?"
AI:    [still referring to the second finding] "The researchers used..."
```

---

## RAG Pipeline

### 1. Ingestion

```python
from src.ingestion import load_and_chunk_pdfs

chunks = load_and_chunk_pdfs(
    pdf_paths=["report.pdf", "paper.pdf"],
    chunk_size=512,
    chunk_overlap=64
)
```

The ingestion pipeline uses a dual-loader strategy:
- **pdfplumber** — handles tables, multi-column layouts, extracts table content as pipe-separated text
- **PyMuPDF** — fallback for simple text PDFs, faster for standard layouts

### 2. Embedding & Storage

```python
from src.embeddings import add_chunks_to_vectorstore

vectorstore = add_chunks_to_vectorstore(chunks)
```

- Chunks are embedded in batches of 100 to manage memory
- ChromaDB persists embeddings to disk — no re-embedding on app restart
- Each chunk stores metadata: `source` (filename), `page` (page number)

### 3. Retrieval

```python
from src.retriever import get_retriever

retriever = get_retriever(
    vectorstore,
    chunks=chunks,    # needed for BM25 index
    mode="hybrid",    # semantic | hybrid | mmr
    k=4
)
```

**Hybrid retrieval** (recommended) combines two complementary signals:
- **BM25** — matches exact keywords, names, IDs, technical terms
- **Semantic** — matches meaning and concepts even when phrased differently
- Results merged with configurable weights (default: 60% semantic, 40% BM25)

### 4. Generation

```python
from src.chain import build_conversational_chain, ask

chain    = build_conversational_chain(retriever)
response = ask(chain, "What is the main finding?", chat_history=[])

print(response["answer"])   # the LLM's grounded answer
print(response["sources"])  # list of source chunks used
```

The chain has 3 internal stages:
1. **Question contextualization** — rewrites follow-up questions to be standalone before retrieval
2. **Document retrieval** — fetches top-k relevant chunks from ChromaDB
3. **Answer generation** — Groq LLM generates answer strictly from retrieved context

---

## RAGAS Evaluation

RAGAS evaluates your RAG pipeline quality without needing human-labeled data.

### Metrics

| Metric | What it measures | Good score |
|---|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? | > 0.80 |
| **Answer Relevancy** | Does the answer actually address the question? | > 0.80 |
| **Context Precision** | Are the retrieved chunks relevant to the question? | > 0.75 |
| **Context Recall** | Did we retrieve enough context to answer fully? | > 0.75 |

### Running Evaluation

In the app, go to the **RAGAS Evaluation** tab:

1. Either upload a JSON eval dataset or use synthetic generation
2. Set number of questions (3-5 recommended for Groq free tier)
3. Click **▶️ Run RAGAS Evaluation**
4. View scores in metric cards and progress bars

### Eval Dataset Format

```json
[
    {
        "question": "What is the main research objective?",
        "ground_truth": "The research objective is to investigate..."
    },
    {
        "question": "What methodology was used?",
        "ground_truth": "The study used a mixed-methods approach..."
    }
]
```

### My RAGAS Scores

Evaluated on a 5-question synthetic dataset over a 28-page technical document:

| Metric | Score |
|---|---|
| Faithfulness | 0.75 |
| Context Precision | 0.75 |
| Context Recall | 1.00 |

> Note: Answer Relevancy occasionally times out on Groq free tier due to rate limits. Scores above are from successful evaluation runs.

---

## Chunking Strategy

The choice of chunk size and overlap significantly impacts retrieval quality.

### Why it matters

```
Too large (2000 chars): Multiple topics per chunk → noisy retrieval
Too small (100 chars):  Incomplete context → LLM can't answer properly
Just right (300-512):   One semantic unit per chunk → precise retrieval
```

### Recommended settings by use case

| Document Type | Chunk Size | Overlap | Retrieval Mode | Top-K |
|---|---|---|---|---|
| Research Papers | 512 | 64 | Semantic | 4 |
| Technical Docs | 512 | 100 | Hybrid | 4 |
| Dense reports | 400 | 100 | Hybrid | 6 |

### How overlap prevents boundary cuts

```
Without overlap:
  Chunk 1: "...revenue grew 40% in Q3"
  Chunk 2: "due to expansion into APAC markets..."
  → Neither chunk alone can answer "Why did revenue grow?"

With overlap (100 chars):
  Chunk 1: "...revenue grew 40% in Q3 due to"
  Chunk 2: "revenue grew 40% in Q3 due to expansion into APAC markets..."
  → Both chunks now contain the complete cause-effect relationship
```

---

## Retrieval Modes

### Semantic Search
Pure vector similarity. Best for conceptual and paraphrased questions.
```
Query: "explain the research methodology"
→ Finds chunks semantically similar even if exact words don't match
```

### Hybrid Search (BM25 + Semantic) — Recommended
Combines keyword matching with semantic understanding. Best for most use cases.
```
Query: "What did Vikramshila Education do?"
→ BM25 matches "Vikramshila Education" exactly
→ Semantic matches surrounding context
→ Combined result is more accurate than either alone
```

### MMR Reranking
Maximal Marginal Relevance — reduces redundancy in retrieved chunks. Best for long documents where the same information repeats across multiple pages.

---

## Deployment

### Streamlit Cloud (Recommended — Free)

1. Push your code to a GitHub repository
2. Add `.gitignore` entries:
   ```
   .env
   chroma_db/
   uploaded_pdfs/
   __pycache__/
   *.pyc
   ```
3. Go to [share.streamlit.io](https://share.streamlit.io)
4. Connect your GitHub repo
5. Set **Main file path**: `app.py`
6. Add secrets: **Settings → Secrets**:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```
7. Click **Deploy**

> **Important:** Add `pysqlite3-binary` to `requirements.txt` for Streamlit Cloud compatibility (ChromaDB requires SQLite 3.35+ which older Ubuntu images don't have).

---

## Requirements

```
streamlit
langchain
langchain-groq
langchain-community
langchain-core
langchain-text-splitters
groq
sentence-transformers
chromadb
pysqlite3-binary
pymupdf
pdfplumber
pypdf
rank-bm25
ragas
datasets
python-dotenv
tiktoken
tqdm
pandas
numpy
```

---

## Known Limitations

- **Scanned PDFs** — image-based PDFs with no text layer cannot be processed without OCR. Install `tesseract-ocr` and `pytesseract` for OCR support.
- **Rate limits** — Groq free tier has token-per-day limits. RAGAS evaluation may hit limits on large datasets. Keep synthetic questions ≤ 5.
- **Session persistence** — conversation history is lost on page refresh. ChromaDB embeddings persist to disk but chat history does not.
- **Very large PDFs** — PDFs over 500 pages may take several minutes to ingest due to embedding time.
- **Non-English PDFs** — the current embedding model is optimised for English. Results on other languages may vary.

---

## Future Improvements

- [ ] Switch to `BAAI/bge-small-en-v1.5` for better retrieval quality
- [ ] Add OCR support for scanned PDFs via pytesseract
- [ ] Persistent chat history across sessions using SQLite
- [ ] Support for `.docx` and `.txt` file formats
- [ ] Re-ranking using a cross-encoder model for improved precision
- [ ] Streaming responses for better UX on long answers
- [ ] Docker containerisation for easy self-hosting

---

## License

MIT License — free to use, modify, and distribute.

---

## Acknowledgements

- [LangChain](https://langchain.com) — RAG orchestration framework
- [Groq](https://groq.com) — blazing fast free LLM inference
- [ChromaDB](https://trychroma.com) — open source vector database
- [RAGAS](https://docs.ragas.io) — RAG evaluation framework
- [Hugging Face](https://huggingface.co) — open source embedding models
- [Streamlit](https://streamlit.io) — rapid UI development

---

*Built as a portfolio project to demonstrate production-grade RAG system design, evaluation, and deployment.*
