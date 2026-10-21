import torch
import numpy as np
from PIL import Image
from colpali_engine.models import BiGemma3, BiGemmaProcessor3

MODEL_NAME = "Cognitive-Lab/NetraEmbed"

def load_model():
    model = BiGemma3.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cuda")
    processor = BiGemmaProcessor3.from_pretrained(MODEL_NAME)
    model.eval()
    return model, processor


@torch.no_grad()
def embed_image_full(model, processor, image_path):
    """No truncation — returns the FULL raw embedding for inspection."""
    img = Image.open(image_path).convert("RGB")
    batch = processor.process_images([img]).to(model.device)
    emb = model(**batch).float()
    return emb, img


def diagnose(image_paths, query, model, processor):
    from colpali_engine.models import BiGemmaProcessor3

    # embed query, full dim
    q_batch = processor.process_queries([query]).to(model.device)
    with torch.no_grad():
        q_emb_full = model(**q_batch).float()  # [1, 2560]

    print(f"Raw query embedding shape: {q_emb_full.shape}")
    print(f"Query embedding norm (before normalize): {q_emb_full.norm().item():.4f}\n")

    results = []
    for path in image_paths:
        emb_full, img = embed_image_full(model, processor, path)

        # check basic image stats — is it actually near-blank?
        arr = np.array(img.convert("L"))  # grayscale
        mean_brightness = arr.mean()
        std_brightness = arr.std()

        # compare truncated (1536) vs full (2560) dim similarity
        for dim_label, dim in [("trunc_1536", 1536), ("full_2560", 2560)]:
            e = emb_full[:, :dim]
            e_norm = torch.nn.functional.normalize(e, p=2, dim=-1)
            q = q_emb_full[:, :dim]
            q_norm = torch.nn.functional.normalize(q, p=2, dim=-1)
            score = (q_norm @ e_norm.T).item()
            results.append({
                "path": path,
                "dim": dim_label,
                "score": score,
                "img_mean_brightness": mean_brightness,
                "img_std_brightness": std_brightness,
            })

    print(f"{'IMAGE':<50} {'DIM':<12} {'SCORE':<10} {'BRIGHTNESS(mean/std)'}")
    for r in sorted(results, key=lambda x: (x["dim"], -x["score"])):
        print(f"{r['path']:<50} {r['dim']:<12} {r['score']:<10.4f} "
              f"{r['img_mean_brightness']:.1f} / {r['img_std_brightness']:.1f}")

    return results


if __name__ == "__main__":
    model, processor = load_model()

    image_paths = [
        f"/home/radhakrishna/ws/adapt_dr/data_queries/data/20.10.1952_0001-{i:03d}.jpg"
        for i in range(1, 21)
    ]

    diagnose(image_paths, "What do sardar achhar singh chhina do?", model, processor)