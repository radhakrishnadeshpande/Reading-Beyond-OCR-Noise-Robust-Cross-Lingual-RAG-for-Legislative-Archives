"""
baseline_comparison.py
Phase 3 of the project plan: run the SAME sample pages + queries through both
NetraEmbed (multilingual) and ColPali (English-centric baseline) to
reproduce, on your own domain, the paper's central finding: English-centric
retrievers degrade sharply on non-English script, while a multilingual
retriever holds up.

This is the single most persuasive table for your professor pitch.

Usage:
    python baseline_comparison.py \
        --image_dir /path/to/page_images \
        --queries_json queries.json \
        --output_csv results.csv
"""

import argparse
import csv
import json
import os

import torch
from PIL import Image
from pathlib import Path

from config import (
    NETRAEMBED_MODEL_ID,
    COLPALI_BASELINE_MODEL_ID,
    EMBED_BATCH_SIZE,
    TOP_K,
    RESULTS_DIR,
)


def collect_images(image_dir: str):
    paths = sorted(Path(image_dir).rglob("*.png")) + sorted(Path(image_dir).rglob("*.jpg"))
    return [{"doc_id": f"{p.parent.name}_{p.stem}", "path": str(p)} for p in paths]


class RetrieverWrapper:
    """Thin common interface so NetraEmbed and ColPali can be swapped in the eval loop."""

    def __init__(self, name: str, model_id: str, model_cls_name: str, processor_cls_name: str, device: str):
        from colpali_engine import models as ce_models

        self.name = name
        self.device = device
        model_cls = getattr(ce_models, model_cls_name)
        processor_cls = getattr(ce_models, processor_cls_name)

        print(f"Loading {name} ({model_id}) ...")
        self.model = model_cls.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=device
        ).eval()
        self.processor = processor_cls.from_pretrained(model_id)

    @torch.no_grad()
    def embed_images(self, entries, batch_size=EMBED_BATCH_SIZE):
        embeddings = []
        for i in range(0, len(entries), batch_size):
            batch = entries[i:i + batch_size]
            images = [Image.open(e["path"]).convert("RGB") for e in batch]
            inputs = self.processor.process_images(images).to(self.device)
            emb = self.model(**inputs)
            embeddings.append(emb.to(torch.float32).cpu())
        return torch.cat(embeddings, dim=0)

    @torch.no_grad()
    def embed_query(self, query: str):
        inputs = self.processor.process_queries([query]).to(self.device)
        return self.model(**inputs).to(torch.float32).cpu()

    def retrieve(self, query_embedding, doc_embeddings, entries, top_k=TOP_K):
        q = torch.nn.functional.normalize(query_embedding, dim=-1)
        d = torch.nn.functional.normalize(doc_embeddings, dim=-1)
        scores = (q @ d.T).squeeze(0)
        top_scores, top_idx = torch.topk(scores, k=min(top_k, len(entries)))
        return [entries[i]["doc_id"] for i in top_idx.tolist()]


def evaluate_retriever(wrapper: RetrieverWrapper, entries, queries, top_k):
    print(f"\n=== Embedding corpus with {wrapper.name} ===")
    doc_embeddings = wrapper.embed_images(entries)

    hits = 0
    n_labeled = 0
    rows = []
    for q in queries:
        query_embedding = wrapper.embed_query(q["query"])
        top_doc_ids = wrapper.retrieve(query_embedding, doc_embeddings, entries, top_k)
        hit = None
        if "expected_doc" in q:
            hit = q["expected_doc"] in top_doc_ids
            n_labeled += 1
            hits += int(hit)
        rows.append({
            "retriever": wrapper.name,
            "query": q["query"],
            "expected_doc": q.get("expected_doc", ""),
            "top_k_doc_ids": " | ".join(top_doc_ids),
            "hit": hit,
        })

    accuracy = hits / n_labeled if n_labeled else None
    return rows, accuracy


def main():
    parser = argparse.ArgumentParser(description="Compare NetraEmbed vs ColPali on the same sample.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--queries_json", required=True)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--output_csv", default=os.path.join(RESULTS_DIR, "baseline_comparison.csv"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    entries = collect_images(args.image_dir)
    if not entries:
        raise SystemExit(f"No images found under {args.image_dir}. Run pdf_to_images.py first.")

    with open(args.queries_json, "r", encoding="utf-8") as f:
        queries = json.load(f)

    retrievers = [
        RetrieverWrapper("NetraEmbed (multilingual)", NETRAEMBED_MODEL_ID,
                          "BiGemma3", "BiGemmaProcessor3", args.device),
        RetrieverWrapper("ColPali (English-centric baseline)", COLPALI_BASELINE_MODEL_ID,
                          "ColPali", "ColPaliProcessor", args.device),
    ]

    all_rows = []
    summary = {}
    for wrapper in retrievers:
        rows, accuracy = evaluate_retriever(wrapper, entries, queries, args.top_k)
        all_rows.extend(rows)
        summary[wrapper.name] = accuracy
        # Free GPU memory before loading the next model
        del wrapper.model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["retriever", "query", "expected_doc", "top_k_doc_ids", "hit"])
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n=== Summary: retrieval accuracy (expected doc in top-k) ===")
    for name, acc in summary.items():
        acc_str = f"{acc:.2%}" if acc is not None else "N/A (no expected_doc labels provided)"
        print(f"  {name}: {acc_str}")
    print(f"\nDetailed results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
