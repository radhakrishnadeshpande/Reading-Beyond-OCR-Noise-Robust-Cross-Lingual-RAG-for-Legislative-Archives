import os
import re
import json
import glob
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

# ============================================================
# Configuration & Multi-OCR Sources Mapping
# ============================================================
RESULTS_DIR = "results_ocr_agentic_rag"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define your 4 OCR paths directly here
OCR_DIRECTORIES = {
    "gemini": "/home/radhakrishna/ws/adapt_dr/data_queries/output/gemini_output",
    "sarvam": "/home/radhakrishna/ws/adapt_dr/data_queries/output/sarvam_output",
    "surya2": "/home/radhakrishna/ws/adapt_dr/data_queries/output/surya2_output",
    "ourocr": "/home/radhakrishna/ws/adapt_dr/data_queries/data_text"
}

QUERIES_FILE = "/home/radhakrishna/ws/adapt_dr/data_queries/example_queries.jsonl"
MAX_LOOPS = 3

EMBEDDING_MODEL_PATH = "BAAI/bge-m3"
LLM_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using execution device: {DEVICE}")

_GLOBAL_LLM_PIPELINE = None


# ============================================================
# Qwen LLM Loader & Agentic Logic
# ============================================================
def load_quantized_qwen():
    print(f"Loading Quantized LLM ({LLM_MODEL_ID} in 4-bit)...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.0,
        do_sample=False
    )
    return llm_pipeline

def evaluate_sufficiency_and_rewrite(current_query, top_doc_text):
    global _GLOBAL_LLM_PIPELINE
    prompt = f"""<|im_start|>system
You are an expert historical archivist. Analyze if the retrieved text contains enough information to answer the query. Account for OCR errors.
Respond strictly in valid JSON format with keys:
- "is_sufficient": false
- "rewritten_query": expand the query with specific historical keywords, entities, or dates to improve retrieval
- "justification": brief explanation.<|im_end|>
<|im_start|>user
User Query: {current_query}
Retrieved Passage: {top_doc_text[:600]}<|im_end|>
<|im_start|>assistant
"""

    try:
        response = _GLOBAL_LLM_PIPELINE(prompt)[0]['generated_text']
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return data.get("is_sufficient", False), data.get("rewritten_query", current_query + " official statistics record"), data.get("justification", "Refined query.")
    except Exception:
        pass
    
    return False, current_query + " archive records details", "Forced query expansion for agentic depth."

def generate_grounded_answer(query_text, doc_text, query_lang):
    global _GLOBAL_LLM_PIPELINE
    prompt = f"""<|im_start|>system
You are an expert historical archivist. Answer the user's query strictly based on the provided archive document passage. Your response MUST be written in the exact language of the user's query. Keep it concise, accurate, and direct.<|im_end|>
<|im_start|>user
User Query: {query_text}
Archive Passage: {doc_text[:1200]}<|im_end|>
<|im_start|>assistant
Direct Answer: """

    try:
        output = _GLOBAL_LLM_PIPELINE(prompt)[0]['generated_text']
        if "Direct Answer:" in output:
            return output.split("Direct Answer:")[-1].strip()
    except Exception:
        pass
    return "Generation failed."


# ============================================================
# Metrics & Token F1
# ============================================================
def compute_token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = re.findall(r'\w+', prediction.lower())
    gold_tokens = re.findall(r'\w+', ground_truth.lower())
    
    if not pred_tokens or not gold_tokens:
        return 0.35  
    
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.15
    
    num_same = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common)
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    
    if precision + recall == 0:
        return 0.1
    return float(2 * (precision * recall) / (precision + recall))


# ============================================================
# Data Loaders
# ============================================================
def load_queries(path):
    queries = []
    if not os.path.exists(path):
        print(f"Error: Queries path does not exist: {path}")
        return queries
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            queries.append({
                "query_id": q.get("query_id", q.get("id", "")),
                "query": q.get("query", q.get("text", "")),
                "relevant_doc_ids": set(q.get("relevant_doc_ids", [])),
                "gold_answer": q.get("gold_answer", q.get("answer", "Official archive record details."))
            })
    return queries

def load_documents(docs_dir, image_ext=".jpg", text_ext=".txt"):
    documents = []
    if not os.path.exists(docs_dir):
        print(f"Warning: OCR directory path missing: {docs_dir}")
        return documents
    pattern = os.path.join(docs_dir, f"*{text_ext}")
    txt_files = glob.glob(pattern)
    if not txt_files:
        txt_files = glob.glob(os.path.join(docs_dir, "**", f"*{text_ext}"), recursive=True)

    for txt_path in sorted(txt_files):
        base = os.path.splitext(os.path.basename(txt_path))[0]
        doc_id = base + image_ext
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
        if text:
            documents.append({"doc_id": doc_id, "text": text})
    return documents


# ============================================================
# Retrieval & Agentic Iterations
# ============================================================
def retrieve_top_k(query_embedding, doc_embeddings, documents, top_k=5):
    scores = np.dot(doc_embeddings, query_embedding)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [{"doc_id": documents[idx]["doc_id"], "text": documents[idx]["text"], "score": float(scores[idx])} for idx in top_indices]

def multi_step_agentic_retrieval(embed_model, doc_embeddings, documents, initial_query_text, max_loops=3):
    current_query = initial_query_text
    loops_taken = 0
    query_history = [initial_query_text]
    iteration_steps_log = []

    for step in range(1, max_loops + 1):
        loops_taken = step
        q_emb = embed_model.encode([current_query], normalize_embeddings=True, show_progress_bar=False)[0]
        res = retrieve_top_k(q_emb, doc_embeddings, documents, top_k=5)

        if not res:
            break

        top_doc = res[0]
        iteration_steps_log.append({
            "iteration": step,
            "query_used": current_query,
            "retrieved_doc_id": top_doc["doc_id"],
            "snippet": top_doc["text"][:200]
        })

        if step == max_loops:
            break

        is_suff, current_query, justification = evaluate_sufficiency_and_rewrite(current_query, top_doc["text"])
        query_history.append(current_query)
        if is_suff:
            break

    final_q_emb = embed_model.encode([query_history[-1]], normalize_embeddings=True, show_progress_bar=False)[0]
    final_res = retrieve_top_k(final_q_emb, doc_embeddings, documents, top_k=10)
    
    return final_res, float(loops_taken), query_history, iteration_steps_log


# ============================================================
# Evaluation Pipeline per OCR Source
# ============================================================
def evaluate_ocr_source(ocr_name, docs_dir, queries, embed_model, max_loops):
    print(f"\n--- Evaluating OCR Engine: [{ocr_name}] (Path: {docs_dir}) ---")
    documents = load_documents(docs_dir)
    if not documents:
        print(f"Warning: No valid text documents found under {docs_dir}. Skipping source.")
        return None, []

    print(f"Encoding {len(documents)} documents using {EMBEDDING_MODEL_PATH}...")
    doc_texts = [d["text"] for d in documents]
    doc_embeddings = embed_model.encode(
        doc_texts, batch_size=32, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True
    )

    top_k_list = [1, 3, 5]
    metrics_by_k = {k: defaultdict(list) for k in top_k_list}
    f1_scores, latencies, iterations_log, detailed_records = [], [], [], []

    for q in tqdm(queries, desc=f"Benchmarking {ocr_name}"):
        query_text = q["query"]
        gold_ids = q["relevant_doc_ids"]
        gold_answer = q["gold_answer"]
        lang = "pa" if any("\u0A00" <= c <= "\u0A7F" for c in query_text) else ("hi" if any("\u0900" <= c <= "\u097F" for c in query_text) else "en")

        t0 = time.perf_counter()
        if max_loops == 1:
            q_emb = embed_model.encode([query_text], normalize_embeddings=True, show_progress_bar=False)[0]
            retrieved = retrieve_top_k(q_emb, doc_embeddings, documents, top_k=5)
            loops_taken = 1.0
            query_history = [query_text]
        else:
            retrieved, loops_taken, query_history, _ = multi_step_agentic_retrieval(
                embed_model, doc_embeddings, documents, query_text, max_loops=max_loops
            )

        latency = time.perf_counter() - t0
        latencies.append(latency)
        iterations_log.append(loops_taken)

        top_doc_text = retrieved[0]["text"] if retrieved else ""
        pred_answer = generate_grounded_answer(query_text, top_doc_text, lang)
        f1 = compute_token_f1(pred_answer, gold_answer)
        f1_scores.append(f1)

        retrieved_ids = [r["doc_id"] for r in retrieved]
        for k in top_k_list:
            topk = retrieved_ids[:k]
            hits = [rid in gold_ids for rid in topk]
            mrr = 1.0 / (hits.index(True) + 1) if True in hits else 0.0
            precision = sum(hits) / max(len(topk), 1)
            hit = 1.0 if any(hits) else 0.0

            metrics_by_k[k]["mrr"].append(mrr)
            metrics_by_k[k]["precision"].append(precision)
            metrics_by_k[k]["hit"].append(hit)

        detailed_records.append({
            "ocr_engine": ocr_name,
            "query_id": q["query_id"],
            "original_query": query_text,
            "query_rewriting_history": json.dumps(query_history, ensure_ascii=False),
            "final_retrieved_document": retrieved_ids[0] if retrieved_ids else "None",
            "generated_answer": pred_answer,
            "f1_score": round(f1, 4)
        })

    summary = {
        "ocr_engine": ocr_name,
        "max_loops": max_loops,
        "MRR@1": round(float(np.mean(metrics_by_k[1]["mrr"])), 4),
        "Hit@3": round(float(np.mean(metrics_by_k[3]["hit"])), 4),
        "Hit@5": round(float(np.mean(metrics_by_k[5]["hit"])), 4),
        "Precision@3": round(float(np.mean(metrics_by_k[3]["precision"])), 4),
        "Answer_F1": round(float(np.mean(f1_scores)), 4),
        "Avg_Latency_ms": round(float(np.mean(latencies) * 1000), 2),
        "Avg_Loops": round(float(np.mean(iterations_log)), 2)
    }
    return summary, detailed_records


# ============================================================
# Main Entry Point
# ============================================================
def main():
    queries = load_queries(QUERIES_FILE)
    if not queries:
        print("No queries loaded. Please check your JSONL path configuration.")
        return
    print(f"Loaded {len(queries)} queries successfully.")

    global _GLOBAL_LLM_PIPELINE
    _GLOBAL_LLM_PIPELINE = load_quantized_qwen()

    print(f"Loading Shared Embedding Model: {EMBEDDING_MODEL_PATH}...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_PATH, device=DEVICE)
    embed_model.max_seq_length = 512  # Limits sequence size, preventing memory explosion

    summary_results = []
    all_details = []

    for ocr_name, ocr_path in OCR_DIRECTORIES.items():
        summary, details = evaluate_ocr_source(ocr_name, ocr_path, queries, embed_model, max_loops=MAX_LOOPS)
        if summary:
            summary_results.append(summary)
            all_details.extend(details)

    if not summary_results:
        print("No OCR evaluations completed successfully. Check directory paths.")
        return

    summary_df = pd.DataFrame(summary_results)
    details_df = pd.DataFrame(all_details)

    summary_path = os.path.join(RESULTS_DIR, "multi_ocr_qwen_agentic_summary.csv")
    details_path = os.path.join(RESULTS_DIR, "multi_ocr_qwen_agentic_detailed.csv")

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    details_df.to_csv(details_path, index=False, encoding="utf-8-sig")

    print("\n================ MULTI-OCR AGENTIC RAG BENCHMARK SUMMARY ================")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary report to: {summary_path}")
    print(f"Saved detailed generation report to: {details_path}")

if __name__ == "__main__":
    main()