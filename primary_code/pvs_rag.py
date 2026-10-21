import json
import glob
from pathlib import Path
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from colpali_engine.models import BiGemma3, BiGemmaProcessor3

RETRIEVAL_MODEL_NAME = "Cognitive-Lab/NetraEmbed"
VLM_MODEL_NAME = "google/gemma-3-4b-it"


@torch.no_grad()
def retrieve_top_pages(model, processor, query_text: str, image_paths: list[str], doc_ids: list[str], dim: int = 768, top_k: int = 3):
    """Retrieves top pages using the already loaded NetraEmbed model in VRAM."""
    device = model.device
    
    # Encode Query
    q_batch = processor.process_queries([query_text]).to(device)
    q_emb = torch.nn.functional.normalize(model(**q_batch)[:, :dim], p=2, dim=-1)

    # Encode Document Pages in batches (cached or pre-computed can speed this up further if needed)
    doc_embs = []
    batch_size = 8  
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        img_batch = processor.process_images(images).to(device)
        embs = model(**img_batch)[:, :dim].float().cpu()
        doc_embs.append(embs)
    
    doc_embs = torch.nn.functional.normalize(torch.cat(doc_embs, dim=0), p=2, dim=-1)

    # Calculate similarity scores
    scores = (q_emb.cpu() @ doc_embs.T).squeeze(0)
    top_indices = scores.argsort(descending=True)[:top_k].tolist()

    return [(image_paths[idx], doc_ids[idx], scores[idx].item()) for idx in top_indices]


def generate_answer(vlm_model, vlm_processor, query_text: str, retrieved_pages: list):
    """Generates native-language response using the already loaded VLM model in VRAM."""
    context_images = [Image.open(page_path).convert("RGB") for page_path, _, _ in retrieved_pages]

    prompt = f"""You are an expert historian and archivist analyzing official 1952 Punjab Vidhan Sabha legislative records.
Based strictly on the provided historical document page image(s), answer the user's question accurately. 
CRITICAL: Respond in the exact same language and script that the user used in their query (e.g., if the query is in Punjabi, write your answer in Punjabi script; if in English, write in English). Do not translate names or official citations.

User Question: {query_text}

Answer:"""

    messages = [
        {
            "role": "user",
            "content": [
                *({"type": "image", "image": img} for img in context_images),
                {"type": "text", "text": prompt}
            ]
        }
    ]

    text_input = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = vlm_processor(
        text=[text_input],
        images=context_images,
        return_tensors="pt"
    ).to(vlm_model.device)

    with torch.no_grad():
        generated_ids = vlm_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False
        )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return vlm_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]


if __name__ == "__main__":
    pvs_pages_dir = Path("/home/radhakrishna/ws/adapt_dr/data_queries/data")
    pvs_qrels_file = Path("/home/radhakrishna/ws/adapt_dr/data_queries/example_queries_9.jsonl")
    dim = 7  

    if not pvs_qrels_file.exists():
        print(f"[ERROR] Query file not found at: {pvs_qrels_file}")
        exit(1)

    image_paths = sorted(glob.glob(str(pvs_pages_dir / "*.png")) + glob.glob(str(pvs_pages_dir / "*.jpg")))
    doc_ids = [Path(p).name for p in image_paths]
    
    queries = [json.loads(line) for line in pvs_qrels_file.read_text().splitlines() if line.strip()]
    print(f"[INFO] Loaded {len(queries)} queries and {len(image_paths)} pages.")

    # --- LOAD MODELS ONCE INTO VRAM BEFORE THE LOOP ---
    print("\n[1/2] Loading NetraEmbed retrieval model into VRAM...")
    retrieval_model = BiGemma3.from_pretrained(RETRIEVAL_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    retrieval_processor = BiGemmaProcessor3.from_pretrained(RETRIEVAL_MODEL_NAME)

    print("\n[2/2] Loading Gemma-3 VLM generation model into VRAM...")
    vlm_processor = AutoProcessor.from_pretrained(VLM_MODEL_NAME)
    vlm_model = Gemma3ForConditionalGeneration.from_pretrained(
        VLM_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    print("All models loaded successfully! Starting inference loop...\n")

    rag_results = []

    for i, q in enumerate(queries):
        query_id = q.get("query_id", f"q_{i}")
        query_text = q.get("text", q.get("query", ""))

        print(f"[Query {i+1}/{len(queries)}] ({query_id}): {query_text}")

        # Retrieve
        top_pages = retrieve_top_pages(retrieval_model, retrieval_processor, query_text, image_paths, doc_ids, dim=dim, top_k=3)
        
        # Generate
        answer = generate_answer(vlm_model, vlm_processor, query_text, top_pages)
        print(f"-> Answer: {answer[:150]}...\n")

        rag_results.append({
            "query_id": query_id,
            "query": query_text,
            "retrieved_sources": [doc_id for _, doc_id, _ in top_pages],
            "generated_answer": answer
        })

    output_path = Path("pvs_rag_generation_output.json")
    output_path.write_text(json.dumps(rag_results, indent=2, ensure_ascii=False))
    print(f"RAG generation results saved successfully to {output_path}")