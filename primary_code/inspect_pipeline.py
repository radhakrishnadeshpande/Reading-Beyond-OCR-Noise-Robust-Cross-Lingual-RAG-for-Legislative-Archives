import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from colpali_engine.models import BiGemma3, BiGemmaProcessor3

MODEL_NAME = "Cognitive-Lab/NetraEmbed"
DIM = 1536


def load_model():
    model = BiGemma3.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cuda")
    processor = BiGemmaProcessor3.from_pretrained(MODEL_NAME)
    model.eval()
    return model, processor


@torch.no_grad()
def embed_image(model, processor, image_path, dim):
    img = Image.open(image_path).convert("RGB")
    batch = processor.process_images([img]).to(model.device)
    emb = model(**batch)[:, :dim].float().cpu()
    return torch.nn.functional.normalize(emb, p=2, dim=-1)


@torch.no_grad()
def embed_query(model, processor, query, dim):
    batch = processor.process_queries([query]).to(model.device)
    emb = model(**batch)[:, :dim].float().cpu()
    return torch.nn.functional.normalize(emb, p=2, dim=-1)


def annotate_image(image_path, rank, score, thumb_size=(300, 424)):
    """Load an image, resize to thumbnail, and draw rank + score + filename on it."""
    img = Image.open(image_path).convert("RGB")
    img.thumbnail(thumb_size)

    # pad onto a canvas so all thumbnails are the same size (for grid alignment)
    canvas = Image.new("RGB", thumb_size, "white")
    canvas.paste(img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    label = f"#{rank}  score={score:.4f}"
    filename = Path(image_path).name

    # black background strip behind text for readability over any image content
    draw.rectangle([(0, 0), (thumb_size[0], 45)], fill=(0, 0, 0))
    draw.text((5, 3), label, fill="lime" if rank == 1 else "white", font=font)
    draw.text((5, 24), filename, fill="yellow", font=font)

    return canvas


def make_grid(images, cols=4, padding=8):
    """Stitch a list of same-size PIL images into a grid."""
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * (w + padding), rows * (h + padding)), "gray")

    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        grid.paste(img, (c * (w + padding), r * (h + padding)))

    return grid


def check_retrieval_and_save(image_paths: list[str], queries: list[str], dim=DIM,
                              save_dir="retrieval_results", top_k=None):
    model, processor = load_model()
    out_dir = Path(save_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"Embedding {len(image_paths)} candidate images...")
    img_embs = torch.cat([embed_image(model, processor, p, dim) for p in image_paths], dim=0)

    for qi, q in enumerate(queries):
        q_emb = embed_query(model, processor, q, dim)
        scores = (q_emb @ img_embs.T).squeeze(0)

        ranked = sorted(zip(image_paths, scores.tolist()), key=lambda x: x[1], reverse=True)
        if top_k:
            ranked = ranked[:top_k]

        print(f"\nQuery: {q}")
        annotated = []
        for rank, (path, score) in enumerate(ranked, start=1):
            print(f"  #{rank}  {score:.4f}  {path}")
            annotated.append(annotate_image(path, rank, score))

        grid = make_grid(annotated, cols=4)

        # save the grid + a text file with the query for reference
        safe_query = "".join(c if c.isalnum() else "_" for c in q)[:50]
        grid_path = out_dir / f"query{qi}_{safe_query}.png"
        grid.save(grid_path)

        # also save just the top-1 image full-size, separately, for a closer look
        top_path, top_score = ranked[0]
        top_img = Image.open(top_path).convert("RGB")
        top_img_annotated = annotate_image(top_path, 1, top_score, thumb_size=top_img.size)
        top_img_annotated.save(out_dir / f"query{qi}_top1_{safe_query}.png")

        print(f"  Saved grid to: {grid_path}")

    print(f"\nAll results saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    check_retrieval_and_save(
        image_paths=[
            "/home/radhakrishna/ws/adapt_dr/data_queries/data/20.10.1952_0001-001.jpg",
            "/home/radhakrishna/ws/adapt_dr/data_queries/data/20.10.1952_0001-002.jpg",
            "/home/radhakrishna/ws/adapt_dr/data_queries/data/20.10.1952_0001-003.jpg",
            "/home/radhakrishna/ws/adapt_dr/data_queries/data/20.10.1952_0001-004.jpg",
            "/home/radhakrishna/ws/adapt_dr/data_queries/data/20.10.1952_0001-005.jpg",
            "/home/radhakrishna/ws/adapt_dr/data_queries/data/20.10.1952_0001-006.jpg",
            "/home/radhakrishna/ws/adapt_dr/data_queries/data/20.10.1952_0001-007.jpg",
            "/home/radhakrishna/ws/adapt_dr/data_queries/data/20.10.1952_0001-008.jpg",
            "/home/radhakrishna/ws/adapt_dr/data_queries/data/20.10.1952_0001-009.jpg",
            "/home/radhakrishna/ws/adapt_dr/data_queries/data/20.10.1952_0001-010.jpg",
            "/home/radhakrishna/ws/adapt_dr/data_queries/data/20.10.1952_0001-011.jpg",
            "/home/radhakrishna/ws/adapt_dr/data_queries/data/20.10.1952_0001-012.jpg",
            "/home/radhakrishna/ws/adapt_dr/data_queries/data/20.10.1952_0001-013.jpg",
            "/home/radhakrishna/ws/adapt_dr/data_queries/data/20.10.1952_0001-014.jpg",
            "/home/radhakrishna/ws/adapt_dr/data_queries/data/20.10.1952_0001-015.jpg",
            "/home/radhakrishna/ws/adapt_dr/data_queries/data/20.10.1952_0001-016.jpg",
            "/home/radhakrishna/ws/adapt_dr/data_queries/data/20.10.1952_0001-017.jpg",
            "/home/radhakrishna/ws/adapt_dr/data_queries/data/20.10.1952_0001-018.jpg",
            "/home/radhakrishna/ws/adapt_dr/data_queries/data/20.10.1952_0001-019.jpg",
            "/home/radhakrishna/ws/adapt_dr/data_queries/data/20.10.1952_0001-020.jpg",
        ],
        queries=[
            "How many number of dacoit cases has happened?",
        ],
        dim=DIM,
        save_dir="/home/radhakrishna/ws/adapt_dr/retrieval_results",
        top_k=None,   # set e.g. 8 to only visualize the top 8 matches instead of all 20
    )