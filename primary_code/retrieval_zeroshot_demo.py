"""
retrieval_zeroshot_demo.py
Phase 2 of the project plan: run NetraEmbed zero-shot (no fine-tuning) over a
small sample of page images and check whether hand-written queries retrieve
the correct page. This is the first concrete evidence you show your professor.

Usage:
    python retrieval_zeroshot_demo.py \
        --image_dir /path/to/page_images \
        --queries_json queries.json

queries.json format:
[
  {"query": "What was the budget allocated for rural roads?", "expected_doc": "session12_page003"},
  ...
]

Requires:
    pip install git+https://github.com/adithya-s-k/colpali.git
    pip install torch --index-url https://download.pytorch.org/whl/cu121   (match your CUDA)
"""

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image

from config import NETRAEMBED_MODEL_ID, EMBED_BATCH_SIZE, TOP_K, RESULTS_DIR


def load_model(device: str):
    """Load NetraEmbed and its processor."""
    from colpali_engine.models import BiGemma3, BiGemmaProcessor3

    print(f"Loading {NETRAEMBED_MODEL_ID} on {device} ...")
    model = BiGemma3.from_pretrained(
        NETRAEMBED_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()
    processor = BiGemmaProcessor3.from_pretrained(NETRAEMBED_MODEL_ID)
    return model, processor


def collect_images(image_dir: str):
    """Gather all page images under image_dir (recursively), keeping a stable doc_id."""
    paths = sorted(Path(image_dir).rglob("*.png")) + sorted(Path(image_dir).rglob("*.jpg"))
    entries = []
    for p in paths:
        # doc_id combines the parent folder (source PDF) and filename for a readable key
        doc_id = f"{p.parent.name}_{p.stem}"
        entries.append({"doc_id": doc_id, "path": str(p)})
    return entries


@torch.no_grad()
def embed_images(model, processor, entries, device, batch_size=EMBED_BATCH_SIZE):
    """Embed every page image in batches. Returns a tensor [N, dim] and the entries list."""
    all_embeddings = []
    for i in range(0, len(entries), batch_size):
        batch = entries[i:i + batch_size]
        images = [Image.open(e["path"]).convert("RGB") for e in batch]
        inputs = processor.process_images(images).to(device)
        embeddings = model(**inputs)
        all_embeddings.append(embeddings.to(torch.float32).cpu())
        print(f"  embedded {min(i + batch_size, len(entries))}/{len(entries)} pages")
    return torch.cat(all_embeddings, dim=0)


@torch.no_grad()
def embed_query(model, processor, query: str, device):
    inputs = processor.process_queries([query]).to(device)
    embedding = model(**inputs)
    return embedding.to(torch.float32).cpu()


def retrieve(query_embedding, doc_embeddings, entries, top_k=TOP_K):
    """Cosine similarity search over precomputed document embeddings."""
    query_norm = torch.nn.functional.normalize(query_embedding, dim=-1)
    doc_norm = torch.nn.functional.normalize(doc_embeddings, dim=-1)
    scores = (query_norm @ doc_norm.T).squeeze(0)  # [N]
    top_scores, top_idx = torch.topk(scores, k=min(top_k, len(entries)))
    return [
        {"doc_id": entries[i]["doc_id"], "path": entries[i]["path"], "score": float(s)}
        for s, i in zip(top_scores.tolist(), top_idx.tolist())
    ]


def main():
    parser = argparse.ArgumentParser(description="Zero-shot NetraEmbed retrieval demo.")
    parser.add_argument("--image_dir", required=True, help="Directory of rendered page images.")
    parser.add_argument("--queries_json", required=True, help="JSON file with test queries.")
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    entries = collect_images(args.image_dir)
    if not entries:
        raise SystemExit(f"No images found under {args.image_dir}. Run pdf_to_images.py first.")
    print(f"Found {len(entries)} page images.")

    with open(args.queries_json, "r", encoding="utf-8") as f:
        queries = json.load(f)

    model, processor = load_model(args.device)

    print("Embedding corpus (one-time cost) ...")
    doc_embeddings = embed_images(model, processor, entries, args.device)

    results_summary = []
    correct = 0
    for q in queries:
        query_embedding = embed_query(model, processor, q["query"], args.device)
        top_results = retrieve(query_embedding, doc_embeddings, entries, args.top_k)
        top_doc_ids = [r["doc_id"] for r in top_results]
        hit = q.get("expected_doc") in top_doc_ids if "expected_doc" in q else None
        if hit:
            correct += 1

        print(f"\nQuery: {q['query']}")
        for rank, r in enumerate(top_results, start=1):
            marker = " <-- expected" if r["doc_id"] == q.get("expected_doc") else ""
            print(f"  {rank}. {r['doc_id']}  (score={r['score']:.4f}){marker}")

        results_summary.append({
            "query": q["query"],
            "expected_doc": q.get("expected_doc"),
            "top_k_doc_ids": top_doc_ids,
            "hit": hit,
        })

    if any(r["hit"] is not None for r in results_summary):
        n_labeled = sum(1 for r in results_summary if r["hit"] is not None)
        print(f"\nRetrieval accuracy (expected doc in top-{args.top_k}): "
              f"{correct}/{n_labeled} = {correct / n_labeled:.2%}")

    out_path = os.path.join(RESULTS_DIR, "zeroshot_retrieval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
