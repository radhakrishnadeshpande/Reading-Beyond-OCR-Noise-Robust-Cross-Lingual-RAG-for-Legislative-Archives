"""
generation.py
The answer-generation step: takes the top-k retrieved page images plus the
user's question and produces a grounded, cited answer using a small local
vision-language model (Gemma 3 4B-IT by default — same family as NetraEmbed,
so no extra model download, and it fits comfortably on a 16GB GPU).

This script deliberately keeps generation separate from retrieval
(retrieval_zeroshot_demo.py) so you can swap generation models independently
-- e.g. to compare a small local VLM against a frontier API model.

Usage:
    python generation.py --images page3.png page7.png --query "What was the budget for rural roads?"
"""

import argparse

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from config import GENERATION_MODEL_ID

GROUNDED_ANSWER_PROMPT = """You are answering a user's question using only the provided document image(s).
If the answer isn't visible in the images, say you don't know -- don't guess.

Question: {question}

Answer using only what's shown in the images above. Cite which image number
(1, 2, 3, ...) supports your answer. If no image supports the answer, say so
explicitly instead of fabricating a response."""


def load_generation_model(model_id: str, device: str):
    print(f"Loading generation model {model_id} on {device} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def build_messages(image_paths, question: str):
    content = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in image_paths]
    content.append({"type": "text", "text": GROUNDED_ANSWER_PROMPT.format(question=question)})
    return [{"role": "user", "content": content}]


@torch.no_grad()
def generate_answer(model, processor, image_paths, question: str, device, max_new_tokens=512):
    messages = build_messages(image_paths, question)
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(device)

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    answer = processor.decode(generated, skip_special_tokens=True)
    return answer


def main():
    parser = argparse.ArgumentParser(description="Generate a grounded answer from retrieved page images.")
    parser.add_argument("--images", nargs="+", required=True,
                         help="Paths to the retrieved page images, in retrieval-rank order.")
    parser.add_argument("--query", required=True, help="The user's question.")
    parser.add_argument("--model_id", default=GENERATION_MODEL_ID)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, processor = load_generation_model(args.model_id, args.device)
    answer = generate_answer(model, processor, args.images, args.query, args.device)

    print("\n=== Question ===")
    print(args.query)
    print("\n=== Retrieved images (in order) ===")
    for i, p in enumerate(args.images, start=1):
        print(f"  [{i}] {p}")
    print("\n=== Answer ===")
    print(answer)


if __name__ == "__main__":
    main()
