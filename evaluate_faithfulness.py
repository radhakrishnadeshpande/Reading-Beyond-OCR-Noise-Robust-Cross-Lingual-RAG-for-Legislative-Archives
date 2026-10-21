"""
evaluate_faithfulness.py
Second evaluation axis alongside retrieval accuracy: is the GENERATED ANSWER
actually supported by the retrieved page, or is the VLM hallucinating?

This script runs the full retrieve -> generate pipeline over a labeled query
set and produces a spreadsheet-friendly CSV for human annotation
(is_grounded: yes/no/partial), plus optional pre-filled auto flags to speed
up manual review. Automated hallucination detection is an open problem
(see project README) -- this is a lightweight starting point, not a
substitute for human judgement, especially for a government-facing use case.

Usage:
    python evaluate_faithfulness.py \
        --image_dir /path/to/page_images \
        --queries_json eval_queries.json \
        --output_csv faithfulness_review.csv
"""

import argparse
import csv
import json
import os

import torch

from config import RESULTS_DIR, GENERATION_MODEL_ID
from retrieval_zeroshot_demo import load_model, collect_images, embed_images, embed_query, retrieve
from generation import load_generation_model, generate_answer


def contains_abstention(answer: str) -> bool:
    """Very rough heuristic flag: did the model refuse / say it doesn't know?
    Useful as a quick sanity signal, not a substitute for human review."""
    markers = ["don't know", "do not know", "cannot find", "not visible",
               "no image supports", "unable to find", "not present in"]
    lowered = answer.lower()
    return any(m in lowered for m in markers)


def main():
    parser = argparse.ArgumentParser(description="Retrieve + generate + prepare for faithfulness review.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--queries_json", required=True,
                         help="JSON list of {'query': ..., 'expected_doc': ..., "
                              "'expected_answer_notes': ...} -- expected_answer_notes is "
                              "optional free text you fill in from ground truth to help "
                              "the human reviewer, not used for scoring automatically.")
    parser.add_argument("--top_k", type=int, default=3,
                         help="Fewer images than default retrieval top_k keeps the "
                              "generation step focused and reduces cross-image confusion.")
    parser.add_argument("--output_csv", default=os.path.join(RESULTS_DIR, "faithfulness_review.csv"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    entries = collect_images(args.image_dir)
    if not entries:
        raise SystemExit(f"No images found under {args.image_dir}. Run pdf_to_images.py first.")

    with open(args.queries_json, "r", encoding="utf-8") as f:
        queries = json.load(f)

    retriever_model, retriever_processor = load_model(args.device)
    print("Embedding corpus ...")
    doc_embeddings = embed_images(retriever_model, retriever_processor, entries, args.device)

    gen_model, gen_processor = load_generation_model(GENERATION_MODEL_ID, args.device)

    rows = []
    for q in queries:
        query_embedding = embed_query(retriever_model, retriever_processor, q["query"], args.device)
        top_results = retrieve(query_embedding, doc_embeddings, entries, args.top_k)
        image_paths = [r["path"] for r in top_results]

        answer = generate_answer(gen_model, gen_processor, image_paths, q["query"], args.device)
        abstained = contains_abstention(answer)

        rows.append({
            "query": q["query"],
            "expected_doc": q.get("expected_doc", ""),
            "expected_answer_notes": q.get("expected_answer_notes", ""),
            "retrieved_doc_ids": " | ".join(r["doc_id"] for r in top_results),
            "generated_answer": answer,
            "model_abstained": abstained,
            # Human fills these in during manual review:
            "is_grounded_yes_no_partial": "",
            "reviewer_notes": "",
        })
        print(f"\nQuery: {q['query']}\nAnswer: {answer}\n{'-'*60}")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {args.output_csv}")
    print("Next step: open this CSV and manually fill in 'is_grounded_yes_no_partial' "
          "for each row by checking the generated answer against the retrieved page image(s).")


if __name__ == "__main__":
    main()
