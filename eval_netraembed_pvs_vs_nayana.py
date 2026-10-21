import json
import glob
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from PIL import Image
from colpali_engine.models import BiGemma3, BiGemmaProcessor3

MODEL_NAME = "Cognitive-Lab/NetraEmbed"


def load_model():
    model = BiGemma3.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
    processor = BiGemmaProcessor3.from_pretrained(MODEL_NAME)
    model.eval()
    return model, processor


@torch.no_grad()
def embed_images(model, processor, image_paths: list[str], dim: int, batch_size: int = 8) -> torch.Tensor:
    all_embs = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        batch = processor.process_images(images).to(model.device)
        embs = model(**batch)
        all_embs.append(embs[:, :dim].float().cpu())
    return torch.nn.functional.normalize(torch.cat(all_embs, dim=0), p=2, dim=-1)


@torch.no_grad()
def embed_queries(model, processor, queries: list[str], dim: int) -> torch.Tensor:
    batch = processor.process_queries(queries).to(model.device)
    embs = model(**batch)
    embs = embs[:, :dim].float().cpu()
    return torch.nn.functional.normalize(embs, p=2, dim=-1)


# ---- metrics ----

def ndcg_at_k(ranked_ids, relevances: dict, k: int) -> float:
    dcg = sum((2 ** relevances.get(d, 0) - 1) / np.log2(i + 2) for i, d in enumerate(ranked_ids[:k]))
    ideal = sorted(relevances.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids, relevances: dict, k: int) -> float:
    relevant = {d for d, r in relevances.items() if r > 0}
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def mrr_at_k(ranked_ids, relevances: dict, k: int) -> float:
    for i, d in enumerate(ranked_ids[:k]):
        if relevances.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def run_eval(model, processor, image_paths, doc_ids, queries, qrels, dim, k=5):
    doc_embs = embed_images(model, processor, image_paths, dim)
    ndcgs, recalls, mrrs = [], [], []
    for q in queries:
        if q["query_id"] not in qrels:
            continue
        q_emb = embed_queries(model, processor, [q["text"]], dim)
        scores = (q_emb @ doc_embs.T).squeeze(0)
        ranked_ids = [doc_ids[i] for i in scores.argsort(descending=True).tolist()]
        rels = qrels[q["query_id"]]
        ndcgs.append(ndcg_at_k(ranked_ids, rels, k))
        recalls.append(recall_at_k(ranked_ids, rels, 10))
        mrrs.append(mrr_at_k(ranked_ids, rels, 10))
    return {f"NDCG@{k}": float(np.mean(ndcgs)), "Recall@10": float(np.mean(recalls)), "MRR@10": float(np.mean(mrrs))}


# ---- A. Punjab Vidhan Sabha pages -----------------------------------

def eval_on_pvs(model, processor, pages_dir: str, qrels_file: str, dim: int):
    image_paths = sorted(glob.glob(str(Path(pages_dir) / "*.png")) + glob.glob(str(Path(pages_dir) / "*.jpg")))
    doc_ids = [Path(p).name for p in image_paths]

    rows = [json.loads(l) for l in Path(qrels_file).read_text().splitlines() if l.strip()]
    queries = [{"query_id": r["query_id"], "text": r["text"]} for r in rows]
    qrels = {r["query_id"]: {d: 1 for d in r["relevant_doc_ids"]} for r in rows}

    print(f"[PVS] {len(image_paths)} pages, {len(queries)} queries")
    return run_eval(model, processor, image_paths, doc_ids, queries, qrels, dim)


# ---- B. Nayana-IR Parquet subset -----------------------------------
import io

def eval_on_nayana_ir(model, processor, nayana_dir: str, dim: int):
    d = Path(nayana_dir)
    
    corpus_files = list(d.glob("corpus/*.parquet")) + list(d.glob("**/corpus/*.parquet")) + list(d.glob("*.parquet"))
    queries_files = list(d.glob("queries/*.parquet")) + list(d.glob("**/queries/*.parquet"))
    qrels_files = list(d.glob("qrels/*.parquet")) + list(d.glob("**/qrels/*.parquet"))

    corpus_df = pd.read_parquet(corpus_files[0]) if corpus_files else pd.DataFrame()
    queries_df = pd.read_parquet(queries_files[0]) if queries_files else pd.DataFrame()
    qrels_df = pd.read_parquet(qrels_files[0]) if qrels_files else pd.DataFrame()

    qid_col = next((c for c in ["query_id", "qid", "id"] if c in queries_df.columns), queries_df.columns[0])
    text_col = next((c for c in ["text", "query", "question"] if c in queries_df.columns), queries_df.columns[1])
    
    queries = [{"query_id": str(row[qid_col]), "text": row[text_col]} for _, row in queries_df.iterrows()]
    
    q_col = next((c for c in ["query_id", "qid"] if c in qrels_df.columns), qrels_df.columns[0])
    d_col = next((c for c in ["doc_id", "corpus_id", "did"] if c in qrels_df.columns), qrels_df.columns[1])
    rel_col = next((c for c in ["relevance", "score", "rel"] if c in qrels_df.columns), qrels_df.columns[2] if len(qrels_df.columns) > 2 else None)

    qrels = defaultdict(dict)
    for _, row in qrels_df.iterrows():
        rel_val = int(row[rel_col]) if rel_col else 1
        qrels[str(row[q_col])][str(row[d_col])] = rel_val

    doc_col = next((c for c in ["doc_id", "id"] if c in corpus_df.columns), corpus_df.columns[0])
    doc_ids = [str(row[doc_col]) for _, row in corpus_df.iterrows()]

    # Override embed_images or load PIL images directly from binary bytes if stored as bytes
    # Let's override the image loading step for Nayana-IR:
    img_col = next((c for c in ["image", "bytes", "image_bytes", "image_path", "path"] if c in corpus_df.columns), None)
    
    all_embs = []
    batch_size = 8
    
    print(f"[Nayana-IR subset] Processing {len(corpus_df)} docs and {len(queries)} queries...")
    
    # Process images in batches directly from dataframe
    for i in range(0, len(corpus_df), batch_size):
        batch_rows = corpus_df.iloc[i:i + batch_size]
        images = []
        for _, row in batch_rows.iterrows():
            val = row[img_col]
            if isinstance(val, bytes):
                img = Image.open(io.BytesIO(val)).convert("RGB")
            elif isinstance(val, str) and Path(val).exists():
                img = Image.open(val).convert("RGB")
            else:
                # Fallback if it's a dict containing bytes
                if isinstance(val, dict) and "bytes" in val:
                    img = Image.open(io.BytesIO(val["bytes"])).convert("RGB")
                else:
                    raise ValueError(f"Could not parse image data from column {img_col}")
            images.append(img)
            
        batch = processor.process_images(images).to(model.device)
        with torch.no_grad():
            embs = model(**batch)
            all_embs.append(embs[:, :dim].float().cpu())
            
    doc_embs = torch.nn.functional.normalize(torch.cat(all_embs, dim=0), p=2, dim=-1)

    # Run evaluation loop using precomputed doc_embs
    ndcgs, recalls, mrrs = [], [], []
    for q in queries:
        if q["query_id"] not in qrels:
            continue
        q_emb = embed_queries(model, processor, [q["text"]], dim)
        scores = (q_emb @ doc_embs.T).squeeze(0)
        ranked_ids = [doc_ids[idx] for idx in scores.argsort(descending=True).tolist()]
        rels = qrels[q["query_id"]]
        ndcgs.append(ndcg_at_k(ranked_ids, rels, 5))
        recalls.append(recall_at_k(ranked_ids, rels, 10))
        mrrs.append(mrr_at_k(ranked_ids, rels, 10))
        
    return {"NDCG@5": float(np.mean(ndcgs)), "Recall@10": float(np.mean(recalls)), "MRR@10": float(np.mean(mrrs))}


if __name__ == "__main__":
    pvs_pages = "/home/radhakrishna/ws/adapt_dr/data_queries/data"
    pvs_qrels = "/home/radhakrishna/ws/adapt_dr/data_queries/example_queries.jsonl"
    nayana_dir = "/home/radhakrishna/ws/adapt_dr/nayana_subset"
    dim = 1536

    print("Loading NetraEmbed...")
    model, processor = load_model()

    pvs_scores = eval_on_pvs(model, processor, pvs_pages, pvs_qrels, dim)
    nayana_scores = eval_on_nayana_ir(model, processor, nayana_dir, dim)

    print("\n=== RESULTS (dim=%d) ===" % dim)
    print("Punjab Vidhan Sabha :", pvs_scores)
    print("Nayana-IR subset    :", nayana_scores)

    Path("netraembed_pvs_vs_nayana_results_1000pages_200queries_1536dim.json").write_text(
        json.dumps({"punjab_vidhan_sabha": pvs_scores, "nayana_ir_subset": nayana_scores}, indent=2)
    )
