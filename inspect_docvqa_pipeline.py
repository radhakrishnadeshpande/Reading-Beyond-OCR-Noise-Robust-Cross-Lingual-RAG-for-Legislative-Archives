import json
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

def qualitative_docvqa(model, processor, dim, split="validation", n_samples=100, top_k=5, save_dir="docvqa_qual"):
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split=split)
    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))

    id_col = "ucsf_document_id" if "ucsf_document_id" in ds.column_names else "image_id"

    seen = {}
    for row in ds:
        doc_id = row[id_col]
        if doc_id not in seen:
            seen[doc_id] = row["image"]

    doc_ids = list(seen.keys())
    images = [seen[d] for d in doc_ids]
    doc_id_to_image = dict(zip(doc_ids, images))   # <-- lookup for saving later

    print(f"Embedding {len(images)} unique document images...")
    doc_embs = embed_images(model, processor, images, dim)

    correct_cases = []
    incorrect_cases = []

    print(f"Evaluating {len(ds)} questions...")
    for row in ds:
        question = row["question"]
        gold_doc_id = row[id_col]
        gold_answers = row.get("answers", None)

        q_emb = embed_queries(model, processor, [question], dim)
        scores = (q_emb @ doc_embs.T).squeeze(0)
        sorted_idx = scores.argsort(descending=True).tolist()
        ranked_ids = [doc_ids[i] for i in sorted_idx]
        ranked_scores = [scores[i].item() for i in sorted_idx]

        rank_of_gold = ranked_ids.index(gold_doc_id) + 1 if gold_doc_id in ranked_ids else None
        top1_correct = ranked_ids[0] == gold_doc_id

        case = {
            "question": question,
            "gold_doc_id": gold_doc_id,
            "gold_answers": gold_answers,
            "top1_predicted_doc_id": ranked_ids[0],
            "top1_score": ranked_scores[0],
            "rank_of_gold_doc": rank_of_gold,
            "gold_doc_score": scores[doc_ids.index(gold_doc_id)].item() if gold_doc_id in doc_ids else None,
            "top_k_predictions": [
                {"doc_id": ranked_ids[i], "score": ranked_scores[i]} for i in range(min(top_k, len(ranked_ids)))
            ],
        }

        (correct_cases if top1_correct else incorrect_cases).append(case)

    print(f"\nTotal: {len(ds)} | Correct (top-1): {len(correct_cases)} | Incorrect (top-1): {len(incorrect_cases)}")
    print(f"Top-1 accuracy: {len(correct_cases) / len(ds):.4f}")

    out_dir = Path(save_dir)
    out_dir.mkdir(exist_ok=True)
    (out_dir / "correct_cases.json").write_text(json.dumps(correct_cases, indent=2))
    (out_dir / "incorrect_cases.json").write_text(json.dumps(incorrect_cases, indent=2))

    summary = {
        "n_total": len(ds),
        "n_correct_top1": len(correct_cases),
        "n_incorrect_top1": len(incorrect_cases),
        "top1_accuracy": len(correct_cases) / len(ds),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved full reports to: {out_dir.resolve()}")

    return correct_cases, incorrect_cases, summary, doc_id_to_image


def save_case_images(cases, doc_id_to_image, save_dir, n=10, label=""):
    """
    Saves gold (and predicted, for incorrect cases) page images to disk as PNGs
    so you can open and view them.
    """
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, case in enumerate(cases[:n]):
        gold_img = doc_id_to_image.get(case["gold_doc_id"])
        if gold_img:
            gold_img.save(out_dir / f"{label}_{i}_gold_{case['gold_doc_id']}.png")

        pred_id = case.get("top1_predicted_doc_id")
        if pred_id and pred_id != case["gold_doc_id"]:
            pred_img = doc_id_to_image.get(pred_id)
            if pred_img:
                pred_img.save(out_dir / f"{label}_{i}_predicted_{pred_id}.png")

    print(f"Saved {min(n, len(cases))} {label} case images to: {out_dir.resolve()}")


if __name__ == "__main__":
    print("Loading NetraEmbed...")
    model, processor = load_model()

    correct_cases, incorrect_cases, summary, doc_id_to_image = qualitative_docvqa(
        model, processor, DIM, split="validation", n_samples=100, top_k=5
    )

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    # Save actual image files to disk so you can view them
    save_case_images(correct_cases, doc_id_to_image, "docvqa_qual/images/correct", n=10, label="correct")
    save_case_images(incorrect_cases, doc_id_to_image, "docvqa_qual/images/incorrect", n=10, label="incorrect")