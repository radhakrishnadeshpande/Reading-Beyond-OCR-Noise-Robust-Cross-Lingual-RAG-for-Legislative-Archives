import os
import json
import time
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

QUERIES_FILE = "/home/radhakrishna/ws/adapt_dr/data_queries/example_queries.jsonl"
DOCS_DIR = "/home/radhakrishna/ws/adapt_dr/data_queries/data_text"

TOP_K = [1, 3, 5]

EMBEDDING_MODELS = {
    "multilingual-e5-large": "intfloat/multilingual-e5-large",
    "bge-m3": "BAAI/bge-m3",
    "labse": "sentence-transformers/LaBSE",
    "muril": "google/muril-base-cased",
    "embeddinggemma-300m": "google/embeddinggemma-300m",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR = "/home/radhakrishna/ws/adapt_dr/ourocr"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Load queries
# ============================================================

def load_queries(path):
    queries = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            q = json.loads(line)

            queries.append({
                "query_id": q.get("query_id", q.get("id", "")),
                "query": q.get("query", q.get("text", "")),
                "relevant_doc_ids": set(
                    q.get("relevant_doc_ids", [])
                )
            })

    return queries


# ============================================================
# Load OCR documents
# ============================================================

def load_documents(docs_dir):

    documents = []

    txt_files = [
        f for f in os.listdir(docs_dir)
        if f.endswith(".txt")
    ]

    txt_files.sort()

    for filename in tqdm(
        txt_files,
        desc="Loading documents"
    ):

        path = os.path.join(docs_dir, filename)

        with open(
            path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            text = f.read().strip()

        if not text:
            continue

        # Convert:
        # document.txt -> document.jpg
        doc_id = os.path.splitext(filename)[0] + ".jpg"

        documents.append({
            "doc_id": doc_id,
            "text": text
        })

    return documents


# ============================================================
# Encode documents
# ============================================================

def encode_documents(model, documents, batch_size=32):

    texts = [d["text"] for d in documents]

    start = time.perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    elapsed = time.perf_counter() - start

    return embeddings, elapsed


# ============================================================
# Encode queries
# ============================================================

def encode_queries(model, queries, model_name):

    texts = []

    for q in queries:

        query = q["query"]

        # E5 models work better with query: prefix
        if "e5" in model_name.lower():
            query = "query: " + query

        texts.append(query)

    start = time.perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    elapsed = time.perf_counter() - start

    return embeddings, elapsed


# ============================================================
# Retrieval
# ============================================================

def retrieve(
    query_embedding,
    document_embeddings,
    documents,
    top_k=5
):

    # Because embeddings are normalized,
    # dot product = cosine similarity
    scores = np.dot(
        document_embeddings,
        query_embedding
    )

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in top_indices:

        results.append({
            "doc_id": documents[idx]["doc_id"],
            "score": float(scores[idx])
        })

    return results


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(retrieved, relevant_docs, k):

    retrieved_ids = [
        r["doc_id"]
        for r in retrieved[:k]
    ]

    hits = [
        doc_id in relevant_docs
        for doc_id in retrieved_ids
    ]

    num_hits = sum(hits)

    # Hit@k
    hit = 1.0 if num_hits > 0 else 0.0

    # Precision@k
    precision = (
        num_hits / k
        if k > 0
        else 0.0
    )

    # Recall@k
    recall = (
        num_hits / len(relevant_docs)
        if len(relevant_docs) > 0
        else 0.0
    )

    # Reciprocal Rank
    reciprocal_rank = 0.0

    for rank, is_hit in enumerate(hits, start=1):

        if is_hit:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "hit": hit,
        "precision": precision,
        "recall": recall,
        "mrr": reciprocal_rank
    }


# ============================================================
# Evaluate one embedding model
# ============================================================

def evaluate_model(
    model_name,
    model_path,
    documents,
    queries
):

    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    start = time.perf_counter()

    model = SentenceTransformer(
        model_path,
        device=DEVICE
    )

    model_load_time = time.perf_counter() - start

    print(
        f"Model loaded in {model_load_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # Document embeddings
    # --------------------------------------------------------

    document_embeddings, document_time = encode_documents(
        model,
        documents
    )

    print(
        f"Document encoding time: "
        f"{document_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # Query embeddings
    # --------------------------------------------------------

    query_embeddings, query_time = encode_queries(
        model,
        queries,
        model_name
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    rows = []

    overall_metrics = {
        k: {
            "hit": [],
            "precision": [],
            "recall": [],
            "mrr": []
        }
        for k in TOP_K
    }

    retrieval_start = time.perf_counter()

    for i, q in enumerate(queries):

        retrieved = retrieve(
            query_embeddings[i],
            document_embeddings,
            documents,
            max(TOP_K)
        )

        for k in TOP_K:

            metrics = calculate_metrics(
                retrieved,
                q["relevant_doc_ids"],
                k
            )

            overall_metrics[k]["hit"].append(
                metrics["hit"]
            )

            overall_metrics[k]["precision"].append(
                metrics["precision"]
            )

            overall_metrics[k]["recall"].append(
                metrics["recall"]
            )

            overall_metrics[k]["mrr"].append(
                metrics["mrr"]
            )

        # Save detailed result for top-5
        rows.append({
            "model": model_name,
            "query_id": q["query_id"],
            "query": q["query"],
            "relevant_doc_ids": json.dumps(
                list(q["relevant_doc_ids"]),
                ensure_ascii=False
            ),
            "retrieved_doc_ids@5": json.dumps(
                [r["doc_id"] for r in retrieved[:5]]
            ),
            "retrieval_scores@5": json.dumps(
                [r["score"] for r in retrieved[:5]]
            )
        })

    retrieval_time = (
        time.perf_counter() - retrieval_start
    )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    summary = {
        "model": model_name,
        "num_queries": len(queries),
        "num_documents": len(documents),
        "model_load_time_sec": model_load_time,
        "document_encoding_time_sec": document_time,
        "query_encoding_time_sec": query_time,
        "retrieval_time_sec": retrieval_time
    }

    for k in TOP_K:

        summary[f"MRR@{k}"] = np.mean(
            overall_metrics[k]["mrr"]
        )

        summary[f"Hit@{k}"] = np.mean(
            overall_metrics[k]["hit"]
        )

        summary[f"Precision@{k}"] = np.mean(
            overall_metrics[k]["precision"]
        )

        summary[f"Recall@{k}"] = np.mean(
            overall_metrics[k]["recall"]
        )

    return summary, rows


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("EMBEDDING MODEL COMPARISON")
    print("=" * 70)

    print(f"Device: {DEVICE}")

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    queries = load_queries(QUERIES_FILE)

    documents = load_documents(DOCS_DIR)

    print(
        f"\nQueries: {len(queries)}"
    )

    print(
        f"Documents: {len(documents)}"
    )

    # --------------------------------------------------------
    # Evaluate models
    # --------------------------------------------------------

    all_summaries = []
    all_details = []

    for model_name, model_path in EMBEDDING_MODELS.items():

        try:

            summary, details = evaluate_model(
                model_name,
                model_path,
                documents,
                queries
            )

            all_summaries.append(summary)
            all_details.extend(details)

        except Exception as e:

            print(
                f"\nERROR with {model_name}: {e}"
            )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        all_summaries
    )

    details_df = pd.DataFrame(
        all_details
    )

    summary_path = os.path.join(
        RESULTS_DIR,
        "embedding_model_comparison.csv"
    )

    details_path = os.path.join(
        RESULTS_DIR,
        "embedding_model_per_query.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    details_df.to_csv(
        details_path,
        index=False
    )

    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL EMBEDDING MODEL COMPARISON")
    print("=" * 70)

    print(
        summary_df[
            [
                "model",
                "MRR@1",
                "MRR@3",
                "MRR@5",
                "Hit@1",
                "Hit@3",
                "Hit@5",
                "Precision@3",
                "Recall@5"
            ]
        ].to_string(index=False)
    )

    print("\nResults saved to:")

    print(summary_path)
    print(details_path)


if __name__ == "__main__":
    main()