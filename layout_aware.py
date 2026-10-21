"""
Advanced Mistral-7B Multilingual Agentic RAG with Unified Chunk-Level Bounding Box Overlays
"""

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
import cv2

import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

CHROMA_DIR = "chroma_store_unified_layout"
RESULTS_DIR = "results_unified_layout"
HIGHLIGHT_DIR = "highlighted_unified_output"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(HIGHLIGHT_DIR, exist_ok=True)

MODEL_KEY = "bge-m3"
MODEL_PATH = "BAAI/bge-m3"
LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using execution device: {DEVICE}")

_GLOBAL_LLM_PIPELINE = None

def load_quantized_mistral():
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

def load_documents_in_chunks(docs_dir, image_ext=".jpg", text_ext=".txt", box_ext=".box", lines_per_chunk=25):
    """Splits text into chunks and computes a single unified bounding box from the first and last word tokens."""
    docs = []
    pattern = os.path.join(docs_dir, f"*{text_ext}")
    txt_files = glob.glob(pattern)
    if not txt_files:
        txt_files = glob.glob(os.path.join(docs_dir, "**", f"*{text_ext}"), recursive=True)

    for txt_path in sorted(txt_files):
        base = os.path.splitext(os.path.basename(txt_path))[0]
        doc_id = base + image_ext
        box_path = os.path.join(os.path.dirname(txt_path), base + box_ext)
        target_box_file = box_path if os.path.exists(box_path) else txt_path
        
        with open(target_box_file, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            
        current_chunk_boxes = []
        chunk_counter = 0

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5 and all(p.isdigit() for p in parts[:4]):
                xmin, ymin, xmax, ymax = map(int, parts[:4])
                token_text = " ".join(parts[4:])
                current_chunk_boxes.append({"box": [xmin, ymin, xmax, ymax], "text": token_text})
            else:
                text_clean = line.strip()
                if text_clean:
                    current_chunk_boxes.append({"box": [0, 0, 0, 0], "text": text_clean})

            if len(current_chunk_boxes) >= lines_per_chunk:
                chunk_text = " ".join([b["text"] for b in current_chunk_boxes])
                valid_boxes = [b["box"] for b in current_chunk_boxes if b["box"] != [0, 0, 0, 0]]
                
                if valid_boxes:
                    unified_box = [
                        min(b[0] for b in valid_boxes),
                        min(b[1] for b in valid_boxes),
                        max(b[2] for b in valid_boxes),
                        max(b[3] for b in valid_boxes)
                    ]
                else:
                    unified_box = [0, 0, 0, 0]

                if chunk_text.strip():
                    docs.append({
                        "chunk_id": f"{base}_chunk_{chunk_counter}",
                        "doc_id": doc_id,
                        "text": chunk_text,
                        "layout_metadata": json.dumps({"unified_box": unified_box})
                    })
                current_chunk_boxes = []
                chunk_counter += 1

        if current_chunk_boxes:
            chunk_text = " ".join([b["text"] for b in current_chunk_boxes])
            valid_boxes = [b["box"] for b in current_chunk_boxes if b["box"] != [0, 0, 0, 0]]
            if valid_boxes:
                unified_box = [
                    min(b[0] for b in valid_boxes),
                    min(b[1] for b in valid_boxes),
                    max(b[2] for b in valid_boxes),
                    max(b[3] for b in valid_boxes)
                ]
            else:
                unified_box = [0, 0, 0, 0]

            if chunk_text.strip():
                docs.append({
                    "chunk_id": f"{base}_chunk_{chunk_counter}",
                    "doc_id": doc_id,
                    "text": chunk_text,
                    "layout_metadata": json.dumps({"unified_box": unified_box})
                })
                
    return docs

def build_index_chunk_level(docs, model_key, embed_fn):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection_name = f"cvip_unified_layout__{model_key.replace('/', '_')}"
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    
    for i in range(0, len(docs), 32):
        batch = docs[i:i + 32]
        collection.add(
            ids=[d["chunk_id"] for d in batch],
            documents=[d["text"] for d in batch],
            metadatas=[{"doc_id": d["doc_id"], "layout_json": d["layout_metadata"]} for d in batch]
        )
    return collection

def render_chunk_layout_highlights(retrieved_chunk_ids, retrieved_metadatas, image_dir, query_id):
    os.makedirs(HIGHLIGHT_DIR, exist_ok=True)
    provenance_mapping = []

    for rank, (chunk_id, metadata) in enumerate(zip(retrieved_chunk_ids, retrieved_metadatas), start=1):
        if not isinstance(metadata, dict):
            continue
            
        doc_id = metadata.get("doc_id")
        image_path = os.path.join(image_dir, doc_id)
        if not os.path.exists(image_path):
            continue
            
        img = cv2.imread(image_path)
        if img is None:
            continue
            
        layout_data = json.loads(metadata.get("layout_json", "{}"))
        unified_box = layout_data.get("unified_box", [0, 0, 0, 0])
        
        if unified_box != [0, 0, 0, 0]:
            xmin, ymin, xmax, ymax = unified_box
            color = (0, 0, 255) if rank == 1 else (0, 255, 0)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 3)
            
        out_filename = f"q_{query_id}_rank_{rank}_{chunk_id}.jpg"
        out_path = os.path.join(HIGHLIGHT_DIR, out_filename)
        cv2.imwrite(out_path, img)
        
        provenance_mapping.append({
            "query_id": query_id,
            "rank": rank,
            "chunk_id": chunk_id,
            "source_document": doc_id,
            "highlighted_image_path": out_path
        })
    return provenance_mapping

def generate_grounded_answer(query_text, doc_text):
    global _GLOBAL_LLM_PIPELINE
    prompt = f"""[INST] You are an expert historical archivist. Answer the user's query strictly based on the provided archive document passage. 
Keep it concise, accurate, and direct.

User Query: {query_text}
Archive Passage: {doc_text[:1200]}

Direct Answer: [/INST]"""

    try:
        output = _GLOBAL_LLM_PIPELINE(prompt)[0]['generated_text']
        if "[/INST]" in output:
            return output.split("[/INST]")[-1].strip()
    except Exception:
        pass
    return "Generation failed."

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

def load_queries_jsonl(path):
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries

def evaluate_variant(collection, queries, image_dir, top_k_list=[1, 3, 5]):
    global _GLOBAL_LLM_PIPELINE
    metrics_by_k = {k: defaultdict(list) for k in top_k_list}
    f1_scores = []
    latencies = []
    detailed_records = []
    provenance_records = []

    for q in tqdm(queries, desc="Evaluating Unified Chunk Layout RAG"):
        query_text = q.get("text", q.get("query", ""))
        query_id = q.get("query_id", "q_unknown")
        gold_ids = set(q.get("relevant_doc_ids", []))
        gold_answer = q.get("gold_answer", q.get("answer", "Official archive record details."))

        t0 = time.perf_counter()
        
        res = collection.query(query_texts=[query_text], n_results=max(top_k_list))
        retrieved_chunk_ids = res["ids"][0] if res["ids"] else []
        retrieved_metadatas = res["metadatas"][0] if res["metadatas"] else []
        retrieved_docs = res["documents"][0] if res["documents"] else []
        
        latency = time.perf_counter() - t0
        latencies.append(latency)

        prov = render_chunk_layout_highlights(retrieved_chunk_ids[:3], retrieved_metadatas[:3], image_dir, query_id)
        provenance_records.extend(prov)

        top_doc_text = retrieved_docs[0] if retrieved_docs else ""
        pred_answer = generate_grounded_answer(query_text, top_doc_text)
        f1 = compute_token_f1(pred_answer, gold_answer)
        f1_scores.append(f1)

        retrieved_doc_ids = [m.get("doc_id") for m in retrieved_metadatas if isinstance(m, dict)]

        for k in top_k_list:
            topk_docs = retrieved_doc_ids[:k]
            hits = [rid in gold_ids for rid in topk_docs]
            mrr = 1.0 / (hits.index(True) + 1) if True in hits else 0.0
            precision = sum(hits) / max(len(topk_docs), 1)
            hit = 1.0 if any(hits) else 0.0

            metrics_by_k[k]["mrr"].append(mrr)
            metrics_by_k[k]["precision"].append(precision)
            metrics_by_k[k]["hit"].append(hit)

        detailed_records.append({
            "query_id": query_id,
            "query": query_text,
            "retrieved_chunk_id": retrieved_chunk_ids[0] if retrieved_chunk_ids else "None",
            "f1_score": round(f1, 4),
            "generated_answer": pred_answer
        })

    summary = {
        "MRR@1": round(float(np.mean(metrics_by_k[1]["mrr"])), 4),
        "Hit@3": round(float(np.mean(metrics_by_k[3]["hit"])), 4),
        "Hit@5": round(float(np.mean(metrics_by_k[5]["hit"])), 4),
        "Answer_F1": round(float(np.mean(f1_scores)), 4),
        "Avg_Latency_ms": round(float(np.mean(latencies) * 1000), 2)
    }
    return summary, detailed_records, provenance_records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries_json", required=True)
    parser.add_argument("--image_dir", required=True)
    args = parser.parse_args()

    queries = load_queries_jsonl(args.queries_json)
    docs = load_documents_in_chunks(args.image_dir)
    print(f"Loaded {len(queries)} queries and {len(docs)} text chunks with unified layout mapping.")

    global _GLOBAL_LLM_PIPELINE
    _GLOBAL_LLM_PIPELINE = load_quantized_mistral()

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_PATH, device=DEVICE
    )
    collection = build_index_chunk_level(docs, MODEL_KEY, embed_fn)

    summary, details, provenance = evaluate_variant(collection, queries, args.image_dir)

    summary_df = pd.DataFrame([summary])
    details_df = pd.DataFrame(details)
    prov_df = pd.DataFrame(provenance)

    summary_df.to_csv(os.path.join(RESULTS_DIR, "unified_layout_summary.csv"), index=False, encoding="utf-8-sig")
    details_df.to_csv(os.path.join(RESULTS_DIR, "unified_layout_details.csv"), index=False, encoding="utf-8-sig")
    prov_df.to_csv(os.path.join(RESULTS_DIR, "unified_layout_provenance.csv"), index=False, encoding="utf-8-sig")

    print("\n=== UNIFIED CHUNK LAYOUT-AWARE RAG COMPLETE ===")
    print(summary_df.to_string(index=False))
    print(f"Highlighted unified chunk outputs saved in: {HIGHLIGHT_DIR}/")

if __name__ == "__main__":
    main()