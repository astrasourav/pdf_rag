"""
evaluate.py
-----------
RAGAS-based evaluation pipeline for the Multi-PDF RAG project.

Key fixes over v1:
    - Switched to llama-3.1-8b-instant (higher free-tier token limits)
    - Added per-question delay to avoid hitting TPD (tokens per day) limit
    - Capped max questions to 5 by default (safe for free tier)
    - Added retry logic with backoff for RateLimitError and TimeoutError
    - Set raise_exceptions=False in RAGAS evaluate() to avoid crashes on partial failures
    - ragas_max_workers=1 to serialize LLM calls and avoid parallel token bursts

Metrics evaluated:
    - Faithfulness        : Is the answer grounded in the retrieved context?
    - Answer Relevancy    : Does the answer actually address the question?
    - Context Precision   : Are the retrieved chunks relevant to the question?
    - Context Recall      : Did we retrieve enough context to answer fully?

Usage in app.py:
    from evaluate import evaluate_pipeline

    scores = evaluate_pipeline(
        chain=st.session_state.chain,
        chunks=st.session_state.chunks,
        num_synthetic=5,
    )
"""

import json
import os
import random
import time
from typing import List, Dict, Optional

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


# ─── Config ───────────────────────────────────────────────────────────────────
# llama-3.1-8b-instant: 14,400 RPD, 500,000 TPM free tier
# llama-3.3-70b-versatile: only 100,000 TPD — exhausted quickly by RAGAS
# Always use 8b-instant for eval on Groq free tier.
DEFAULT_MODEL      = "llama-3.1-8b-instant"
EVAL_TEMPERATURE   = 0
EVAL_DELAY_SECS    = 4     # wait between questions to avoid TPM bursts
EVAL_MAX_RETRIES   = 3     # retries on RateLimitError / TimeoutError
EVAL_RETRY_WAIT    = 15    # seconds to wait before retrying after rate limit
EVAL_MAX_QUESTIONS = 5     # cap to protect free-tier token budget (increase if on paid tier)

DEFAULT_METRICS = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]


# ─── Step 1: Load Eval Dataset from JSON ─────────────────────────────────────
def load_eval_dataset(json_path: str) -> List[Dict]:
    """
    Load evaluation Q&A pairs from a JSON file.

    Expected format:
        [
            {"question": "What is X?", "ground_truth": "X is ..."},
            ...
        ]

    Args:
        json_path: Path to JSON file.

    Returns:
        List of {"question": str, "ground_truth": str} dicts.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"[ERROR] Eval dataset not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("[ERROR] JSON must be a list of objects.")

    for i, item in enumerate(data):
        if "question" not in item:
            raise ValueError(f"[ERROR] Item {i} missing 'question' key.")
        if "ground_truth" not in item:
            raise ValueError(f"[ERROR] Item {i} missing 'ground_truth' key.")

    print(f"[INFO] Loaded {len(data)} eval pairs from {json_path}")
    return data


# ─── Step 2: Generate Synthetic Eval Questions ────────────────────────────────
def generate_synthetic_dataset(
    chunks: List[Document],
    llm: ChatGroq,
    num_questions: int = EVAL_MAX_QUESTIONS,
) -> List[Dict]:
    """
    Auto-generate evaluation Q&A pairs from document chunks.
    Samples random chunks and asks the LLM to produce a question + answer.

    Includes:
        - Per-chunk delay (EVAL_DELAY_SECS) to avoid TPM spikes
        - Retry on RateLimitError with backoff
        - Skips failed chunks gracefully

    Args:
        chunks:        Document chunks from ingestion.py.
        llm:           ChatGroq LLM instance.
        num_questions: Max Q&A pairs to generate. Capped at EVAL_MAX_QUESTIONS.

    Returns:
        List of {"question": str, "ground_truth": str} dicts.
    """
    # Hard cap to protect free-tier token budget
    num_questions = min(num_questions, EVAL_MAX_QUESTIONS)
    sampled       = random.sample(chunks, min(num_questions, len(chunks)))
    dataset       = []

    print(f"[INFO] Generating {len(sampled)} synthetic Q&A pairs (delay={EVAL_DELAY_SECS}s each)...")

    for i, chunk in enumerate(sampled):
        prompt = f"""Based ONLY on the following text, generate one clear question and its accurate answer.

Text:
{chunk.page_content[:800]}

Respond in this exact JSON format with no extra text:
{{
  "question": "your question here",
  "ground_truth": "your answer here"
}}"""

        for attempt in range(EVAL_MAX_RETRIES):
            try:
                response = llm.invoke(prompt)
                raw      = response.content.strip()

                # Strip markdown fences if present
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                raw = raw.strip()

                parsed = json.loads(raw)

                if "question" in parsed and "ground_truth" in parsed:
                    dataset.append({
                        "question":     parsed["question"],
                        "ground_truth": parsed["ground_truth"],
                    })
                    print(f"[INFO] Generated Q&A {i+1}/{len(sampled)}")
                    break

            except Exception as e:
                err = str(e)
                if "rate_limit" in err.lower() or "429" in err:
                    wait = EVAL_RETRY_WAIT * (attempt + 1)
                    print(f"[WARN] Rate limit hit — waiting {wait}s before retry {attempt+1}/{EVAL_MAX_RETRIES}")
                    time.sleep(wait)
                elif "timeout" in err.lower():
                    print(f"[WARN] Timeout on chunk {i+1}, retry {attempt+1}/{EVAL_MAX_RETRIES}")
                    time.sleep(5)
                else:
                    print(f"[WARN] Skipping chunk {i+1}: {err}")
                    break

        # Polite delay between each chunk to avoid bursting TPM limit
        if i < len(sampled) - 1:
            time.sleep(EVAL_DELAY_SECS)

    print(f"[INFO] Successfully generated {len(dataset)} Q&A pairs.")
    return dataset


# ─── Step 3: Build RAGAS Dataset ─────────────────────────────────────────────
def build_ragas_dataset(
    chain,
    qa_pairs: List[Dict],
) -> Dataset:
    """
    Run the RAG pipeline on each question and build the HuggingFace Dataset
    that RAGAS expects as input.

    Collected per question:
        - question     : the test question
        - answer       : LLM's generated answer
        - contexts     : retrieved chunk texts
        - ground_truth : expected correct answer

    Args:
        chain:    Conversational RAG chain from chain.py.
        qa_pairs: List of {"question": str, "ground_truth": str} dicts.

    Returns:
        HuggingFace Dataset with columns: question, answer, contexts, ground_truth
    """
    questions, answers, contexts_list, ground_truths = [], [], [], []

    print(f"[INFO] Building RAGAS dataset from {len(qa_pairs)} questions...")

    for i, pair in enumerate(qa_pairs):
        question     = pair["question"]
        ground_truth = pair["ground_truth"]

        for attempt in range(EVAL_MAX_RETRIES):
            try:
                response = chain.invoke({
                    "input":        question,
                    "chat_history": [],
                })

                answer   = response.get("answer", "")
                contexts = [doc.page_content for doc in response.get("context", [])]

                questions.append(question)
                answers.append(answer)
                contexts_list.append(contexts)
                ground_truths.append(ground_truth)

                print(f"[INFO] Processed {i+1}/{len(qa_pairs)}: {question[:60]}...")
                break

            except Exception as e:
                err = str(e)
                if "rate_limit" in err.lower() or "429" in err:
                    wait = EVAL_RETRY_WAIT * (attempt + 1)
                    print(f"[WARN] Rate limit — waiting {wait}s (retry {attempt+1})")
                    time.sleep(wait)
                else:
                    print(f"[WARN] Skipping question {i+1}: {err}")
                    break

        # Delay between pipeline calls to avoid token bursts
        if i < len(qa_pairs) - 1:
            time.sleep(EVAL_DELAY_SECS)

    if not questions:
        raise ValueError("[ERROR] No questions were processed successfully.")

    ragas_dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts_list,
        "ground_truth": ground_truths,
    })

    print(f"[INFO] RAGAS dataset built: {len(questions)} rows.")
    return ragas_dataset


# ─── Step 4: Run RAGAS Evaluation ────────────────────────────────────────────
def run_evaluation(
    ragas_dataset: Dataset,
    llm: ChatGroq,
    metrics: list = None,
) -> Dict[str, float]:
    """
    Run RAGAS metrics on the prepared dataset.

    Key settings:
        - raise_exceptions=False  : partial failures don't crash the whole eval

    Args:
        ragas_dataset: HuggingFace Dataset from build_ragas_dataset().
        llm:           ChatGroq LLM for RAGAS internal scoring.
        metrics:       RAGAS metric objects. Defaults to all 4.

    Returns:
        Dict of metric name -> float score.
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    from ragas.llms import LangchainLLMWrapper
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    # parallel token bursts hitting Groq free-tier rate limits.
    # answer_relevancy is the slowest metric — it generates multiple
    # questions internally, so it needs a longer timeout window.
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    # Inject LLM + embeddings into each metric
    for metric in metrics:
        metric.llm        = ragas_llm
        metric.embeddings = ragas_embeddings
        # Groq free tier does not support n > 1 (sampling).
        # AnswerRelevancy uses n variations (strictness) which defaults to 3+.
        # We must set it to 1 for Groq.
        if hasattr(metric, "strictness"):
            metric.strictness = 1

    print("[INFO] Running RAGAS evaluation (this may take a few minutes)...")

    # Run metrics in two passes:
    #   Pass 1 — fast metrics first (faithfulness, context_precision, context_recall)
    #   Pass 2 — answer_relevancy separately with a pause before it
    # answer_relevancy is the slowest metric — it generates several questions
    # internally per row, consuming the most tokens and time.
    # Splitting passes gives Groq rate limits time to recover between them.

    fast_metrics = [m for m in metrics if m.name != "answer_relevancy"]
    slow_metrics = [m for m in metrics if m.name == "answer_relevancy"]

    all_results = {}

    # Pass 1: fast metrics
    if fast_metrics:
        r1 = evaluate(ragas_dataset, metrics=fast_metrics, raise_exceptions=False)
        try:
            df1 = r1.to_pandas()
            for col in df1.columns:
                if col != "question":
                    all_results[col] = df1[col]
        except Exception as e:
            print(f"[WARN] Pass 1 parse error: {e}")

    # Pause before the slow metric to let token limits recover
    if slow_metrics:
        print("[INFO] Pausing before answer_relevancy evaluation...")
        time.sleep(10)
        try:
            r2 = evaluate(ragas_dataset, metrics=slow_metrics, raise_exceptions=False)
            df2 = r2.to_pandas()
            for col in df2.columns:
                if col != "question":
                    all_results[col] = df2[col]
        except Exception as e:
            print(f"[WARN] Pass 2 (answer_relevancy) failed: {e}")
            print("[INFO] Continuing with partial scores.")

    import pandas as pd
    result_df = pd.DataFrame(all_results)

    name_map = {
        "faithfulness":      "Faithfulness",
        "answer_relevancy":  "Answer Relevancy",
        "context_precision": "Context Precision",
        "context_recall":    "Context Recall",
    }

    # result_df is already a plain pandas DataFrame built from both passes.
    import math
    scores = {}
    try:
        for key, label in name_map.items():
            if key in result_df.columns:
                val = result_df[key].mean()
                if val is not None and not math.isnan(float(val)):
                    scores[label] = round(float(val), 4)
    except Exception as e:
        print(f"[WARN] Could not parse scores: {e}")

    if not scores:
        print("[WARN] No valid scores returned. Likely all questions hit rate limits.")
    else:
        print(f"[INFO] Evaluation complete: {scores}")

    return scores


# ─── Step 5: evaluate_pipeline() — single entry point for app.py ─────────────
def evaluate_pipeline(
    chain,
    chunks: List[Document],
    llm: Optional[ChatGroq] = None,
    questions: Optional[List[str]] = None,
    json_path: Optional[str] = None,
    num_synthetic: int = EVAL_MAX_QUESTIONS,
) -> Dict[str, float]:
    """
    Full evaluation pipeline — only function app.py needs to import.

    3 input modes (checked in order):
        Mode B — json_path provided   : load Q&A from file (most accurate)
        Mode A — questions provided   : use your questions, auto-gen ground truths
        Mode C — default              : fully synthetic from indexed chunks

    Token budget tip:
        Each question costs ~4 LLM calls (one per RAGAS metric).
        At 5 questions: ~20 LLM calls total.
        Keep num_synthetic <= 5 on Groq free tier.

    Args:
        chain:         Conversational RAG chain from chain.py.
        chunks:        Document chunks from ingestion.py.
        llm:           ChatGroq instance. Loaded fresh (8b-instant) if None.
        questions:     Optional list of question strings.
        json_path:     Optional path to JSON eval file.
        num_synthetic: Questions to generate if no file/questions provided.

    Returns:
        Dict of metric name -> float score.
    """
    if llm is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("[ERROR] GROQ_API_KEY not set in .env")
        llm = ChatGroq(
            model=DEFAULT_MODEL,      # 8b-instant — higher free-tier limits
            temperature=EVAL_TEMPERATURE,
            api_key=api_key,
        )
        print(f"[INFO] Eval LLM: {DEFAULT_MODEL}")

    # ── Determine Q&A source ──────────────────────────────────────────────────
    if json_path:
        print("[INFO] Eval mode: JSON file")
        qa_pairs = load_eval_dataset(json_path)

    elif questions:
        print("[INFO] Eval mode: provided questions → generating ground truths")
        qa_pairs = []
        for q in questions[:EVAL_MAX_QUESTIONS]:   # cap here too
            try:
                response = chain.invoke({"input": q, "chat_history": []})
                qa_pairs.append({
                    "question":     q,
                    "ground_truth": response.get("answer", ""),
                })
                time.sleep(EVAL_DELAY_SECS)
            except Exception as e:
                print(f"[WARN] Skipping question '{q[:40]}': {e}")

    else:
        print("[INFO] Eval mode: synthetic generation from chunks")
        num_synthetic = min(num_synthetic, EVAL_MAX_QUESTIONS)
        qa_pairs      = generate_synthetic_dataset(chunks, llm, num_questions=num_synthetic)

    if not qa_pairs:
        raise ValueError("[ERROR] No Q&A pairs available. Check your PDFs are indexed.")

    ragas_dataset = build_ragas_dataset(chain, qa_pairs)
    scores        = run_evaluation(ragas_dataset, llm)

    return scores


# ─── Quick Test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from ingestion import load_and_chunk_pdfs
    from embeddings import add_chunks_to_vectorstore
    from retriever import get_retriever
    from chain import build_conversational_chain

    paths = sys.argv[1:] if len(sys.argv) > 1 else []
    if not paths:
        print("Usage: python evaluate.py path/to/file.pdf")
        sys.exit(1)

    print("=== Full RAG + RAGAS Evaluation ===\n")
    chunks      = load_and_chunk_pdfs(paths)
    vectorstore = add_chunks_to_vectorstore(chunks)
    retriever   = get_retriever(vectorstore, chunks=chunks, mode="hybrid", k=4)
    chain       = build_conversational_chain(retriever)

    scores = evaluate_pipeline(chain=chain, chunks=chunks, num_synthetic=3)

    print("\n=== RAGAS Scores ===")
    for metric, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {metric:<22} {score:.4f}  {bar}")