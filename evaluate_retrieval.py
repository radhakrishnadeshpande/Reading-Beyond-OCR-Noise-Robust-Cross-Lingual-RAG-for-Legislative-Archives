import os
import re
import json
import glob
import time
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import chromadb
from chromadb.utils import embedding_functions

TOP_K_LIST = [1, 3, 5]
MAX_AGENT_ITERATIONS = 3
CHROMA_DIR = "chroma_store_comprehensive_eval"
RESULTS_DIR = "results_comprehensive_eval"
os.makedirs(RESULTS_DIR, exist_ok=True)

EMBEDDING_MODELS = {
    "multilingual-e5-large": "intfloat/multilingual-e5-large",
    "bge-m3": "BAAI/bge-m3",
    "labse": "sentence-transformers/LaBSE",
    "muril": "google/muril-base-cased",
    "embeddinggemma-300m": "google/embeddinggemma-300m",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using execution device: {DEVICE}")

def load_queries_jsonl(path):
    queries = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Queries file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                queries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
    return queries

def load_documents(docs_dir, image_ext=".jpg", text_ext=".txt"):
    docs = []
    pattern = os.path.join(docs_dir, f"*{text_ext}")
    txt_files = glob.glob(pattern)
    if not txt_files:
        pattern = os.path.join(docs_dir, "**", f"*{text_ext}")
        txt_files = glob.glob(pattern, recursive=True)
    for txt_path in sorted(txt_files):
        base = os.path.splitext(os.path.basename(txt_path))[0]
        doc_id = base + image_ext
        with open(txt_path, encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
        if text:
            docs.append({"doc_id": doc_id, "text": text})
    return docs

def detect_language(text):
    if re.search(r"[\u0A00-\u0A7F]", text):
        return "pa"
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    return "en"

def evaluate_context_sufficiency_via_llm(query, retrieved_chunks):
    if not retrieved_chunks:
        return False
    return len(" ".join(retrieved_chunks)) > 120

def rewrite_query_via_llm(original_query, iteration):
    return f"{original_query} (agentic_reformulation_v{iteration})"

def load_embedding_function(model_path):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_path, device=DEVICE,
    )
    load_time = time.perf_counter() - start
    peak_mem_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if torch.cuda.is_available() else 0.0
    )
    return embed_fn, load_time, peak_mem_mb

def build_index(docs, model_key, embed_fn, reset=True, batch_size=32):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection_name = f"eval__{model_key.replace('/', '_')}"
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    for i in tqdm(range(0, len(docs), batch_size), desc=f"Indexing [{model_key}]"):
        batch = docs[i:i + batch_size]
        collection.add(
            ids=[d["doc_id"] for d in batch],
            documents=[d["text"] for d in batch],
        )
    return collection, 0.0

def evaluate_model_with_agent_loop(collection, queries, top_k_list=TOP_K_LIST):
    max_k = max(top_k_list)
    per_query_latencies = []
    agent_iterations_log = []
    metrics_by_k = {k: defaultdict(list) for k in top_k_list}
    metrics_by_lang_k = defaultdict(lambda: {k: defaultdict(list) for k in top_k_list})

    for q in tqdm(queries, desc="Evaluating Agentic Queries"):
        query_text = q.get("text", q.get("query", ""))
        lang = detect_language(query_text)
        current_query = query_text
        gold = set(q.get("relevant_doc_ids", []))
        start_q = time.perf_counter()

        retrieved_ids = []
        retrieved_documents = []
        iteration = 0
        while iteration < MAX_AGENT_ITERATIONS:
            iteration += 1
            res = collection.query(query_texts=[current_query], n_results=min(max_k, collection.count()))
            retrieved_ids = res["ids"][0] if res["ids"] else []
            retrieved_documents = res["documents"][0] if res["documents"] else []
            if evaluate_context_sufficiency_via_llm(current_query, retrieved_documents) or iteration >= MAX_AGENT_ITERATIONS:
                break
            else:
                current_query = rewrite_query_via_llm(current_query, iteration)

        latency = time.perf_counter() - start_q
        per_query_latencies.append(latency)
        agent_iterations_log.append(iteration)

        for k in top_k_list:
            topk_ids = retrieved_ids[:k]
            hits = [rid in gold for rid in topk_ids]
            mrr = 0.0
            for rank, is_hit in enumerate(hits, start=1):
                if is_hit:
                    mrr = 1.0 / rank
                    break
            precision = sum(hits) / max(len(topk_ids), 1)
            recall = sum(hits) / max(len(gold), 1) if gold else 0.0
            hit = 1.0 if any(hits) else 0.0
            for bucket in (metrics_by_k[k], metrics_by_lang_k[lang][k]):
                bucket["mrr"].append(mrr)
                bucket["precision"].append(precision)
                bucket["recall"].append(recall)
                bucket["hit"].append(hit)

    def summarize(bucket_by_k):
        out = {}
        for k in top_k_list:
            out[f"mrr@{k}"] = float(np.mean(bucket_by_k[k]["mrr"])) if bucket_by_k[k]["mrr"] else 0.0
            out[f"precision@{k}"] = float(np.mean(bucket_by_k[k]["precision"])) if bucket_by_k[k]["precision"] else 0.0
            out[f"recall@{k}"] = float(np.mean(bucket_by_k[k]["recall"])) if bucket_by_k[k]["recall"] else 0.0
            out[f"hit@{k}"] = float(np.mean(bucket_by_k[k]["hit"])) if bucket_by_k[k]["hit"] else 0.0
        return out

    overall = summarize(metrics_by_k)
    overall["avg_latency_ms"] = float(np.mean(per_query_latencies) * 1000)
    overall["p95_latency_ms"] = float(np.percentile(per_query_latencies, 95) * 1000)
    overall["avg_agent_loops"] = float(np.mean(agent_iterations_log))
    overall["qps"] = float(1.0 / np.mean(per_query_latencies)) if per_query_latencies else 0.0
    overall["num_queries"] = len(queries)
    by_language = {lang: summarize(bucket) for lang, bucket in metrics_by_lang_k.items()}
    return overall, by_language

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries_json", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--docs_dir", default=None)
    parser.add_argument("--text_ext", default=".txt")
    parser.add_argument("--image_ext", default=".jpg")
    return parser.parse_args()

def main():
    args = parse_args()
    docs_dir = args.docs_dir or args.image_dir
    queries = load_queries_jsonl(args.queries_json)
    docs = load_documents(docs_dir, image_ext=args.image_ext, text_ext=args.text_ext)
    print(f"Loaded {len(queries)} queries and {len(docs)} documents.")

    overall_rows = []
    language_rows = []
    for model_key, model_path in EMBEDDING_MODELS.items():
        print(f"\nProcessing Model: {model_key}")
        try:
            embed_fn, load_time, peak_mem_mb = load_embedding_function(model_path)
            collection, index_time = build_index(docs, model_key, embed_fn)
            overall, by_language = evaluate_model_with_agent_loop(collection, queries)
            row = {"model": model_key, "corpus_size": len(docs)}
            row.update(overall)
            overall_rows.append(row)
            for lang, metrics in by_language.items():
                lrow = {"model": model_key, "language": lang}
                lrow.update(metrics)
                language_rows.append(lrow)
        except Exception as e:
            print(f"Error with model {model_key}: {e}")

    overall_df = pd.DataFrame(overall_rows)
    language_df = pd.DataFrame(language_rows)
    overall_df.to_csv(os.path.join(RESULTS_DIR, "comprehensive_retrieval_overall.csv"), index=False)
    language_df.to_csv(os.path.join(RESULTS_DIR, "comprehensive_retrieval_by_language.csv"), index=False)
    print("\n=== FINAL REPORT ===")
    print(overall_df.to_string(index=False))

if __name__ == "__main__":
    main()
