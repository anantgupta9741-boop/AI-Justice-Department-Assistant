"""
evaluate.py — Run the QA evaluation set through the RAG pipeline.

Input:  data/eval/ — one CSV or JSON file with columns/keys:
            question, reference_answer, source_section (optional)

Output:
    data/eval/results.csv   — per-question scores
    Printed summary table

Metrics:
    - Retrieval hit@k: whether any retrieved chunk mentions the expected source section.
    - ROUGE-L F1: n-gram overlap between generated and reference answer.
    - Embedding cosine similarity: semantic similarity between generated and reference answer.

Usage:
    python evaluate.py
"""

import os
import sys
import json
import pathlib
import textwrap

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

from rag import answer_question

load_dotenv()

EVAL_DIR = pathlib.Path("data/eval")
RESULTS_PATH = EVAL_DIR / "results.csv"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_eval_set() -> pd.DataFrame:
    """Load the first CSV or JSON file found in data/eval/."""
    csv_files = list(EVAL_DIR.glob("*.csv"))
    json_files = list(EVAL_DIR.glob("*.json"))

    if csv_files:
        df = pd.read_csv(csv_files[0])
    elif json_files:
        with open(json_files[0], encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        print(f"ERROR: No CSV or JSON file found in {EVAL_DIR}.")
        sys.exit(1)

    required = {"question", "reference_answer"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: Eval file is missing columns: {missing}")
        sys.exit(1)

    if "source_section" not in df.columns:
        df["source_section"] = ""

    return df.dropna(subset=["question", "reference_answer"])


def retrieval_hit(retrieved_docs, source_section: str) -> bool:
    """
    Return True if the expected source_section string appears in any retrieved chunk.
    When source_section is blank, this metric is skipped (returns None).
    """
    if not source_section or not source_section.strip():
        return None
    needle = source_section.strip().lower()
    for doc in retrieved_docs:
        combined = (doc.page_content + " " + str(doc.metadata)).lower()
        if needle in combined:
            return True
    return False


def rouge_l_score(generated: str, reference: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = scorer.score(reference, generated)
    return round(scores["rougeL"].fmeasure, 4)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    if np.linalg.norm(vec_a) == 0 or np.linalg.norm(vec_b) == 0:
        return 0.0
    return round(float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))), 4)


def run_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    embed_model = SentenceTransformer(EMBED_MODEL)
    rows = []

    total = len(df)
    for idx, row in df.iterrows():
        question = str(row["question"]).strip()
        reference = str(row["reference_answer"]).strip()
        expected_section = str(row.get("source_section", "")).strip()

        print(f"  [{idx + 1}/{total}] {textwrap.shorten(question, 70)}", end="", flush=True)

        try:
            result = answer_question(question)
        except Exception as exc:
            print(f" ERROR: {exc}")
            rows.append({
                "question": question,
                "reference_answer": reference,
                "generated_answer": "ERROR",
                "rouge_l": None,
                "cosine_sim": None,
                "retrieval_hit": None,
                "error": str(exc),
            })
            continue

        generated = result["answer"]
        retrieved_docs = result.get("retrieved_docs", [])

        rl = rouge_l_score(generated, reference)
        ref_vec, gen_vec = embed_model.encode([reference, generated])
        cos = cosine_similarity(ref_vec, gen_vec)
        hit = retrieval_hit(retrieved_docs, expected_section)

        print(f"  ROUGE-L={rl:.3f}  cos={cos:.3f}  hit={hit}")

        rows.append({
            "question": question,
            "reference_answer": reference,
            "generated_answer": generated,
            "rouge_l": rl,
            "cosine_sim": cos,
            "retrieval_hit": hit,
            "error": "",
        })

    return pd.DataFrame(rows)


def print_summary(results: pd.DataFrame):
    valid = results[results["error"] == ""]
    hit_rows = valid[valid["retrieval_hit"].notna()]

    print("\n" + "=" * 54)
    print(f"{'Metric':<35} {'Value':>10}")
    print("-" * 54)
    print(f"{'Total questions':<35} {len(results):>10}")
    print(f"{'Errors':<35} {(results['error'] != '').sum():>10}")
    print(f"{'Mean ROUGE-L':<35} {valid['rouge_l'].mean():>10.4f}")
    print(f"{'Mean cosine similarity':<35} {valid['cosine_sim'].mean():>10.4f}")
    if not hit_rows.empty:
        hit_rate = hit_rows["retrieval_hit"].mean()
        print(f"{'Retrieval hit@k (where applicable)':<35} {hit_rate:>10.4f}")
    else:
        print(f"{'Retrieval hit@k':<35} {'N/A':>10}")
    print("=" * 54)


def main():
    print("=== AI Justice Department Assistant — Evaluation ===\n")

    if not EVAL_DIR.exists():
        print(f"ERROR: {EVAL_DIR} does not exist.")
        sys.exit(1)

    print("Loading evaluation set ...")
    df = load_eval_set()
    print(f"Found {len(df)} question(s).\n")

    print("Running pipeline ...")
    results = run_evaluation(df)

    results.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved to {RESULTS_PATH}")

    print_summary(results)


if __name__ == "__main__":
    main()

