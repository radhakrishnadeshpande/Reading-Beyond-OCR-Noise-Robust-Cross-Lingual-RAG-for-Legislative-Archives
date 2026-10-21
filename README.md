# Reading Beyond OCR Noise: Robust Cross-Lingual RAG for Legislative Archives

**Radha Krishna Deshpande, Ajoy Mondal, Akash Manna, C V Jawahar**
CVIT, International Institute of Information Technology, Hyderabad, India

[Paper (PDF)](REPLACE_WITH_PAPER_LINK) · [Project page](REPLACE_WITH_PROJECT_PAGE_LINK) · [Code](https://github.com/radhakrishnadeshpande/Reading-Beyond-OCR-Noise-Robust-Cross-Lingual-RAG-for-Legislative-Archives#motivation)

---

## Overview

Decades of Indian parliamentary records survive only as degraded, multi-column scans in Hindi (Devanagari), Punjabi (Gurmukhi) and English. On such pages, OCR and layout errors are the main obstacle to finding information.

This project couples **layout-aware multi-script OCR** with an **agentic retrieval-augmented generation (RAG) loop**. The agent checks whether the retrieved evidence is sufficient, rewrites the query if it is not, and searches again before generating a grounded answer in the user's own language.

We evaluate on the **Punjab Vidhan Sabha (PVS) archive**: about 200,000 pages spanning 1950–2024.

![Pipeline overview](assets/pipeline.png)

## Highlights

- **Native-language querying.** Queries are embedded directly in Hindi, Punjabi or English. No machine-translation step.
- **Agentic loop.** Retrieve → evaluate context sufficiency (LLM) → rewrite query → retrieve again, up to `L` iterations.
- **Fully open-weight and local.** The whole pipeline runs on a single RTX 4060 (16 GB), and every answer traces back to a source page.
- **Hybrid retrieval** with BAAI/bge-m3 embeddings stored in ChromaDB.

## Pipeline

1. **Offline.** Layout segmentation (fine-tuned YOLO) → OCR (CRNN-based recognizer) → chunking → multilingual embeddings with metadata → vector DB.
2. **Online.** The query `q` is embedded as `v_q = E(q)`. The top-k chunks are retrieved by cosine similarity.
3. **Agent loop.** An evaluation LLM checks whether the context is sufficient. If not, a rewriting LLM produces a new query `q'` that keeps the intent of `q` and rephrases terms likely to be lost to OCR errors.
4. **Generation.** An LLM answers in the user's language, conditioned on the query and the validated context.

## Results

### Retrieval on the PVS benchmark (1,000 pages / 200 queries, indexed against the full archive)

| Model / pipeline | MRR@1 | Hit@3 | Hit@5 | Prec@3 |
|---|---|---|---|---|
| MuRIL | 0.000 | 0.005 | 0.015 | 0.002 |
| LaBSE | 0.320 | 0.510 | 0.590 | 0.170 |
| Embedding-gemma (300M) | 0.525 | 0.735 | 0.775 | 0.245 |
| NetraEmbed (dim=768) | 0.580 | 0.816 | 0.890 | 0.272 |
| NetraEmbed (dim=1536) | 0.598 | 0.835 | 0.905 | 0.278 |
| NetraEmbed (dim=2560) | 0.605 | 0.843 | 0.912 | 0.281 |
| Multilingual-E5-Large | 0.620 | 0.855 | 0.915 | 0.285 |
| Standard BGE-M3 | 0.775 | 0.900 | 0.940 | 0.300 |
| **Ours (optimized pipeline)** | **0.810** | **0.925** | **0.955** | **0.325** |

### OCR and layout (20 hand-transcribed pages)

| Layout model | OCR model | CRR (%) | WRR (%) |
|---|---|---|---|
| Gemini-3.1 | Gemini-3.1 | 99.7 | 89.6 |
| Surya2 | Surya2 | 99.6 | 88.9 |
| Sarvam | Sarvam | 99.5 | 89.4 |
| YOLO (fine-tuned) | CRNN (ours) | 95.67 | 80.03 |
| CRAFT | CRNN (ours) | 89.00 | 60.35 |

A better layout model alone lifts WRR from 60.35% to 80.03% with the same recognizer. Our pipeline trails the commercial and large-model OCR systems on recognition, but stays fully local.

### Effect of OCR choice on retrieval

| OCR model | MRR@1 | Hit@3 | Hit@5 | Prec@3 |
|---|---|---|---|---|
| Gemini-3.1 | 0.825 | 0.920 | 0.960 | 0.3533 |
| Sarvam | 0.820 | 0.935 | 0.955 | 0.325 |
| Surya2 | 0.815 | 0.925 | 0.950 | 0.305 |
| Ours | 0.810 | 0.925 | 0.955 | 0.325 |

### Agent-loop ablation (`L` = 1 → 3)

Increasing the maximum number of retrieval–rewrite iterations from 1 to 3 raises MRR@1 from about 0.89 to 0.97 and Precision@3 from about 0.33 to 0.37. Answer F1 rises from about 0.18 to 0.35, a nearly 2× relative gain. The loop mostly recovers answer-relevant context that single-pass retrieval misses entirely.

## Dataset

| | |
|---|---|
| Source | Punjab Vidhan Sabha, [pvs.neva.gov.in](https://pvs.neva.gov.in/) |
| Size | ~200,000 pages, 1950–2024 |
| Scripts | Hindi (Devanagari), Punjabi (Gurmukhi), English, often mixed on one page |
| Content | Proceedings, debates, starred and unstarred questions, budgets, committee reports, notices |
| Evaluation set | 1,000 pages and 200 gold question–answer pairs, stratified by decade, script and document type |
| OCR ground truth | 20 hand-transcribed pages across degradation levels |


## Limitations

- Severe OCR errors on degraded pages can still hurt dense retrieval and entity matching.
- The agent loop and LLM-based sufficiency checks add inference latency over standard retrieval.
- Complex financial and unstructured tables can lose hierarchical structure when converted to text chunks.


## Acknowledgment

This work is supported by MeitY, Government of India, through the NLTM-Bhashini project.

## Contact

Radha Krishna Deshpande · radhakrishna.d@research.iiit.ac.in
Akash Manna · akash.manna@research.iiit.ac.in
Ajoy Mondal · ajoy.mondal@iiit.ac.in
C V Jawahar · jawahar@iiit.ac.in