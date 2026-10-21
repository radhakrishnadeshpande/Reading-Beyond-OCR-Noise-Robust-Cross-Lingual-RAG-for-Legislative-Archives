import pandas as pd

def eval_on_nayana_ir(model, processor, nayana_dir: str, dim: int):
    d = Path(nayana_dir)
    
    # Load Parquet files instead of jsonl/tsv
    # Adjust subfolder paths (e.g., corpus/, queries/, qrels/) if they match your directory layout
    corpus_df = pd.read_parquet(list(d.glob("corpus/*.parquet"))[0] if list(d.glob("corpus/*.parquet")) else d / "corpus.parquet")
    queries_df = pd.read_parquet(list(d.glob("queries/*.parquet"))[0] if list(d.glob("queries/*.parquet")) else d / "queries.parquet")
    qrels_df = pd.read_parquet(list(d.glob("qrels/*.parquet"))[0] if list(d.glob("qrels/*.parquet")) else d / "qrels.parquet")

    # Map dataframe rows to match the expected format
    corpus = corpus_df.to_dict(orient="records")
    queries = queries_df.to_dict(orient="records")
    
    qrels = defaultdict(dict)
    for _, row in qrels_df.iterrows():
        qrels[str(row["query_id"])][str(row["doc_id"])] = int(row["relevance"])

    image_paths = [c["image_path"] for c in corpus]
    doc_ids = [c["doc_id"] for c in corpus]

    print(f"[Nayana-IR subset] {len(image_paths)} docs, {len(queries)} queries")
    return run_eval(model, processor, image_paths, doc_ids, queries, qrels, dim)