"""
synthetic_query_generation.py
Generates training/eval queries per document page image, following the
M3DR paper's recipe: several query types per page (factual, long-answer,
multiple-choice/short-answer, cross-paragraph reasoning), produced by
prompting a capable VLM. This is what turns a folder of raw scanned pages
into a labeled dataset you can fine-tune and evaluate on.

For a legislative Q&A register specifically, the built-in structure (member
name, question number, ministry, reply) makes some query types easy to get
right at higher quality -- see the generation prompt below.

Usage:
    python synthetic_query_generation.py \
        --image_dir /path/to/page_images \
        --output_json synthetic_queries.json \
        --queries_per_page 5
"""

import argparse
import json
import os

import torch
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageTextToText

from config import GENERATION_MODEL_ID, QUERY_DIR

QUERY_GENERATION_PROMPT = """You are creating training data for a document retrieval system.
Look at this page from a legislative (Vidhan Sabha) proceedings document and
generate {n} diverse questions that a real user might ask, whose answers are
found ONLY on this page. Cover a mix of these types:

1. Factual (e.g. "What was the budget allocated for X?")
2. Long-answer (requires summarizing part of the page)
3. Short-answer / multiple-choice style (a specific name, date, or number)
4. Cross-paragraph reasoning (requires connecting two parts of the page)

Respond ONLY with a JSON list of objects, no other text:
[{{"query": "...", "query_type": "factual|long_answer|short_answer|cross_paragraph"}}, ...]"""


def collect_images(image_dir: str):
    paths = sorted(Path(image_dir).rglob("*.png")) + sorted(Path(image_dir).rglob("*.jpg"))
    return [{"doc_id": f"{p.parent.name}_{p.stem}", "path": str(p)} for p in paths]


def load_model(model_id: str, device: str):
    print(f"Loading query-generation model {model_id} on {device} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


@torch.no_grad()
def generate_queries_for_page(model, processor, image_path: str, n: int, device):
    image = Image.open(image_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": QUERY_GENERATION_PROMPT.format(n=n)},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(device)
    output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7)
    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    raw_text = processor.decode(generated, skip_special_tokens=True)

    try:
        # Model sometimes wraps JSON in ```json fences -- strip those defensively.
        cleaned = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        queries = json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"  [warning] could not parse JSON output for {image_path}, skipping. Raw output:\n{raw_text}")
        queries = []
    return queries


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training/eval queries per page.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_json", default=os.path.join(QUERY_DIR, "synthetic_queries.json"))
    parser.add_argument("--queries_per_page", type=int, default=5)
    parser.add_argument("--model_id", default=GENERATION_MODEL_ID)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    entries = collect_images(args.image_dir)
    if not entries:
        raise SystemExit(f"No images found under {args.image_dir}. Run pdf_to_images.py first.")

    model, processor = load_model(args.model_id, args.device)

    all_pairs = []
    for entry in entries:
        print(f"Generating queries for {entry['doc_id']} ...")
        queries = generate_queries_for_page(model, processor, entry["path"], args.queries_per_page, args.device)
        for q in queries:
            all_pairs.append({
                "query": q.get("query", ""),
                "query_type": q.get("query_type", "unknown"),
                "expected_doc": entry["doc_id"],
                "image_path": entry["path"],
            })

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated {len(all_pairs)} query-page pairs from {len(entries)} pages.")
    print(f"Saved to {args.output_json}")
    print("\nIMPORTANT: run a quality filter pass (manual spot-check or an LLM-based "
          "filter) before using this as training data -- see README for guidance.")


if __name__ == "__main__":
    main()
