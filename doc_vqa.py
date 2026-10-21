import json
import glob
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from datasets import load_dataset

from colpali_engine.models import BiGemma3, BiGemmaProcessor3

MODEL_NAME = "Cognitive-Lab/NetraEmbed"
DIM = 1536


def load_model():
    model = BiGemma3.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
    processor = BiGemmaProcessor3.from_pretrained(MODEL_NAME)
    model.eval()
    return model, processor


@torch.no_grad()
def embed_images(model, processor, images, dim, batch_size=8):
    all_embs = []
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i + batch_size]
        batch = processor.process_images(batch_imgs).to(model.device)
        embs = model(**batch)
        all_embs.append(embs[:, :dim].float().cpu())
    return torch.nn.functional.normalize(torch.cat(all_embs, dim=0), p=2, dim=-1)


@torch.no_grad()
def embed_queries(model, processor, queries, dim):
    batch = processor.process_queries(queries).to(model.device)
    embs = model(**batch)
    embs = embs[:, :dim].float().cpu()
    return torch.nn.functional.normalize(embs, p=2, dim=-1)


def recall_at_k(ranked_ids, correct_id, k):
    return 1.0 if correct_id in ranked_ids[:k] else 0.0


def mrr_at_k(ranked_ids, correct_id, k):
    if correct_id in ranked_ids[:k]:
        return 1.0 / (ranked_ids[:k].index(correct_id) + 1)
    return 0.0


def eval_on_docvqa(model, processor, dim, split="validation", n_samples=500, k_list=(1, 5, 10)):
    """
    DocVQA retrieval-style eval: each question is matched to its own document image.
    Uses the HF 'docvqa' dataset — each row already carries image + question + doc identity.
    """
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split=split)
    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))

    # Build unique doc pool (dedupe repeated images per doc_id/ucsf_document_id)
    id_col = "ucsf_document_id" if "ucsf_document_id" in ds.column_names else "image_id"
    seen = {}
    for row in ds:
        doc_id = row[id_col]
        if doc_id not in seen:
            seen[doc_id] = row["image"]

    doc_ids = list(seen.keys())
    images = [seen[d] for d in doc_ids]

    print(f"[DocVQA] Embedding {len(images)} unique document images...")
    doc_embs = embed_images(model, processor, images, dim)

    results = {f"Recall@{k}": [] for k in k_list}
    results.update({f"MRR@{k}": [] for k in k_list})

    print(f"[DocVQA] Evaluating {len(ds)} questions...")
    for row in ds:
        question = row["question"]
        gold_doc_id = row[id_col]

        q_emb = embed_queries(model, processor, [question], dim)
        scores = (q_emb @ doc_embs.T).squeeze(0)
        ranked_ids = [doc_ids[i] for i in scores.argsort(descending=True).tolist()]

        for k in k_list:
            results[f"Recall@{k}"].append(recall_at_k(ranked_ids, gold_doc_id, k))
            results[f"MRR@{k}"].append(mrr_at_k(ranked_ids, gold_doc_id, k))

    return {k: float(np.mean(v)) for k, v in results.items()}


if __name__ == "__main__":
    print("Loading NetraEmbed...")
    model, processor = load_model()

    scores = eval_on_docvqa(model, processor, DIM, split="validation", n_samples=500)
    print("\n=== DocVQA RESULTS (dim=%d) ===" % DIM)
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")

    Path("docvqa_results.json").write_text(json.dumps(scores, indent=2))