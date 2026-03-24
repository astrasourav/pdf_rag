"""
chain.py
--------
Wires the Groq LLM + retriever into a full RAG Q&A chain.
Supports single-turn and multi-turn (memory) conversations.

Functions:
    - get_llm()                      : loads the Groq LLM
    - get_rag_prompt()               : builds the RAG prompt template
    - get_contextualize_prompt()     : rewrites follow-up questions for retrieval
    - build_conversational_chain()   : full LCEL RAG chain with memory
    - ask()                          : single entry point — call this from app.py

Usage in app.py:
    from chain import build_conversational_chain, ask

    chain    = build_conversational_chain(retriever)
    response = ask(chain, "What are the key findings?", chat_history)
"""

import os
from typing import List, Tuple

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()


# ─── Config ───────────────────────────────────────────────────────────────────
DEFAULT_MODEL       = "llama-3.1-8b-instant"    # fast + free on Groq
FALLBACK_MODEL      = "llama-3.3-70b-versatile"  # smarter, still free
DEFAULT_TEMPERATURE = 0                           # 0 = deterministic, best for RAG


# ─── Step 1: Load Groq LLM ────────────────────────────────────────────────────
def get_llm(
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> ChatGroq:
    """
    Load the Groq LLM via LangChain's ChatGroq integration.
    Reads GROQ_API_KEY automatically from .env file.

    Args:
        model:       Groq model ID. Default: llama-3.1-8b-instant (fastest).
                     Other free options:
                       - "llama-3.3-70b-versatile"  (smarter answers)
                       - "mixtral-8x7b-32768"        (large context window)
        temperature: 0 = focused/deterministic (best for RAG factual answers).
                     Higher values = more creative (not recommended for RAG).

    Returns:
        ChatGroq LLM instance ready to use in a LangChain chain.

    Raises:
        ValueError: If GROQ_API_KEY is not found in the .env file.
    """
    api_key = os.getenv("GROQ_API_KEY") # Our GROQ API KEY
    if not api_key:
        raise ValueError(
            "[ERROR] GROQ_API_KEY not found. "
            "Add it to your .env file: GROQ_API_KEY=your_key_here\n"
            "Get a free key at: https://console.groq.com"
        )

    llm = ChatGroq(
        model=model,
        temperature=temperature,
        api_key=api_key,
    )
    print(f"[INFO] Groq LLM loaded: {model}")
    return llm


# ─── Step 2: RAG Prompt Template ─────────────────────────────────────────────
def get_rag_prompt() -> ChatPromptTemplate:
    """
    Build the RAG system prompt template.

    Design decisions:
        - Instructs the LLM to ONLY use provided context (reduces hallucination)
        - Asks it to cite source document when possible (powers UI citations)
        - Tells it to say a clear message if context is insufficient
        - MessagesPlaceholder enables multi-turn memory

    Returns:
        ChatPromptTemplate with system instructions + history + question slots.
    """
    system_prompt = """You are a helpful assistant that answers questions \
strictly based on the provided document context.

Rules:
- Answer ONLY using the context below. Do not use outside knowledge.
- If the answer is not in the context, say: \
"I couldn't find this in the uploaded documents."
- Always mention which document or section your answer comes from when possible.
- Keep answers clear, concise, and well-structured.
- If asked to summarise, provide bullet points for clarity.

Context:
{context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    return prompt


# ─── Step 3: Contextualize Question Prompt ───────────────────────────────────
def get_contextualize_prompt() -> ChatPromptTemplate:
    """
    Prompt that rewrites the latest user question to be fully standalone,
    factoring in the chat history before retrieval happens.

    Why this matters:
        Without this, follow-up questions like "Can you elaborate on that?"
        or "What did it say about revenue?" get sent to the retriever as-is.
        The retriever has no context for "that" or "it" and returns bad chunks.
        This prompt rewrites them into self-contained questions first.

    Returns:
        ChatPromptTemplate for question contextualization.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest user question, "
         "rewrite the question to be fully standalone and self-contained. "
         "Do NOT answer the question — only rewrite it if needed. "
         "If it is already standalone, return it exactly as-is."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    return prompt


# ─── Step 4: Build Full Conversational RAG Chain ─────────────────────────────
def build_conversational_chain(retriever, model: str = DEFAULT_MODEL):
    """
    Build a complete conversational RAG chain using LangChain LCEL.

    The chain has 3 internal stages:
        Stage A — History-aware retriever:
                  Rewrites follow-up questions → retrieves relevant chunks
        Stage B — Document chain:
                  Stuffs chunks into the RAG prompt → passes to LLM
        Stage C — Full RAG chain:
                  Connects A and B, returns answer + source documents

    Args:
        retriever: Any retriever from retriever.py (semantic / hybrid / mmr).
        model:     Groq model ID (default: llama-3.1-8b-instant).

    Returns:
        A LangChain chain that accepts:
            {
                "input":        str,
                "chat_history": List[BaseMessage]
            }
        And returns:
            {
                "answer":  str,
                "context": List[Document],   # retrieved source chunks
                "input":   str               # rewritten question
            }

    Example:
        chain    = build_conversational_chain(retriever)
        response = chain.invoke({
            "input": "What is the main finding?",
            "chat_history": []
        })
        print(response["answer"])
        print(response["context"])
    """
    llm = get_llm(model=model)

    # Stage A: history-aware retriever
    # Rewrites follow-up questions before retrieval using chat history
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=get_contextualize_prompt(),
    )

    # Stage B: document stuffing + answer generation chain
    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=get_rag_prompt(),
    )

    # Stage C: combine retrieval + generation into one end-to-end chain
    rag_chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=question_answer_chain,
    )

    print("[INFO] Conversational RAG chain built successfully.")
    return rag_chain


# ─── Step 5: ask() — single entry point for app.py ───────────────────────────
def ask(
    chain,
    question: str,
    chat_history: List[Tuple[str, str]] = None,
) -> dict:
    """
    Send a question to the RAG chain and return a clean structured response.
    This is the ONLY function app.py needs to import and call from chain.py.

    Handles:
        - Converting session_state chat history tuples → LangChain messages
        - Invoking the chain
        - Extracting and deduplicating source documents for UI citations

    Args:
        chain:        Chain returned by build_conversational_chain().
        question:     The user's question string.
        chat_history: List of (human_msg, ai_msg) string tuples.
                      From st.session_state.messages in app.py.
                      Pass [] or None for the very first question.

    Returns:
        dict:
            {
                "answer":   str,   the LLM's grounded answer
                "sources":  list,  [{"file": str, "page": int, "snippet": str}]
                "question": str,   the rewritten standalone question
            }

    Example:
        history  = [("What is RAG?", "RAG stands for Retrieval Augmented...")]
        response = ask(chain, "How does retrieval work?", history)

        print(response["answer"])
        for src in response["sources"]:
            print(f"{src['file']}  page {src['page']}")
            print(src["snippet"])
    """
    if chat_history is None:
        chat_history = []

    # Convert (human, ai) string tuples → LangChain BaseMessage objects
    lc_history = []
    for human_msg, ai_msg in chat_history:
        lc_history.append(HumanMessage(content=human_msg))
        lc_history.append(AIMessage(content=ai_msg))

    # Invoke the chain
    response = chain.invoke({
        "input":        question,
        "chat_history": lc_history,
    })

    # Extract and deduplicate source documents for citation cards in the UI
    sources = []
    seen    = set()
    for doc in response.get("context", []):
        file    = doc.metadata.get("source",  "Unknown")
        page    = doc.metadata.get("page",    "?")
        snippet = doc.page_content[:200].strip().replace("\n", " ")
        key     = (file, page)

        if key not in seen:
            seen.add(key)
            sources.append({
                "file":    file,
                "page":    page,
                "snippet": snippet,
            })

    return {
        "answer":   response["answer"],
        "sources":  sources,
        "question": response.get("input", question),
    }


# ─── Quick Test ───────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     import sys
#     from ingestion import load_and_chunk_pdfs
#     from embeddings import add_chunks_to_vectorstore
#     from retriever import get_retriever

#     paths = sys.argv[1:] if len(sys.argv) > 1 else []

#     if not paths:
#         print("Usage: python chain.py path/to/file.pdf")
#         sys.exit(1)

#     print("=== Building full RAG pipeline ===\n")
#     chunks      = load_and_chunk_pdfs(paths)
#     vectorstore = add_chunks_to_vectorstore(chunks)
#     retriever   = get_retriever(vectorstore, chunks=chunks, mode="hybrid", k=4)
#     chain       = build_conversational_chain(retriever)

#     # Turn 1
#     print("\n--- Turn 1 ---")
#     r1 = ask(chain, "What is this document about?", chat_history=[])
#     print(f"Answer  : {r1['answer']}")
#     print(f"Sources : {[f\"{s['file']} p.{s['page']}\" for s in r1['sources']]}")

#     # Turn 2 — follow-up that tests contextualization
#     print("\n--- Turn 2 (follow-up) ---")
#     history = [(r1["question"], r1["answer"])]
#     r2      = ask(chain, "Can you elaborate on that?", chat_history=history)
#     print(f"Answer  : {r2['answer']}")
#     print(f"Sources : {[f\"{s['file']} p.{s['page']}\" for s in r2['sources']]}")