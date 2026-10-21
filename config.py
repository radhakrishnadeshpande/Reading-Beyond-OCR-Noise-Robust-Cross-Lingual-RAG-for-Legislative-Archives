"""
config.py
Shared configuration for the M3DR -> Vidhan Sabha domain-adaptation project.

Edit the values below to match your machine before running any script.
Everything is centralized here so you only change paths in one place.
"""

import os

# -----------------------------------------------------------------------
# Paths (edit these for your machine — defaults assume /scratch is large)
# -----------------------------------------------------------------------
SCRATCH_DIR = os.environ.get("PROJECT_SCRATCH", "/home/radhakrishna/ws/adapt_dr/primary_code")
HF_CACHE_DIR = os.path.join(SCRATCH_DIR, "hf_cache")

RAW_PDF_DIR = os.path.join(SCRATCH_DIR, "data", "raw_pdfs")          # source PDFs / scans
PAGE_IMAGE_DIR = os.path.join(SCRATCH_DIR, "data", "page_images")    # rendered page images
INDEX_DIR = os.path.join(SCRATCH_DIR, "data", "index")               # saved embeddings / FAISS index
QUERY_DIR = os.path.join(SCRATCH_DIR, "data", "queries")             # synthetic + eval queries
CHECKPOINT_DIR = os.path.join(SCRATCH_DIR, "checkpoints")            # LoRA adapter checkpoints
RESULTS_DIR = os.path.join(SCRATCH_DIR, "results")                   # csv / json result dumps

for _d in [SCRATCH_DIR, HF_CACHE_DIR, RAW_PDF_DIR, PAGE_IMAGE_DIR,
           INDEX_DIR, QUERY_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# Point HF/transformers caches at scratch, not your small root partition.
os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_DIR)

# -----------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------
NETRAEMBED_MODEL_ID = "Cognitive-Lab/NetraEmbed"          # single dense vector retriever
COLNETRAEMBED_MODEL_ID = "Cognitive-Lab/ColNetraEmbed"    # multi-vector (ColBERT-style) retriever
COLPALI_BASELINE_MODEL_ID = "vidore/colpali-v1.3"         # English-centric baseline for comparison
GENERATION_MODEL_ID = "google/gemma-3-4b-it"              # local small VLM for answer generation

# -----------------------------------------------------------------------
# Retrieval / generation settings
# -----------------------------------------------------------------------
TOP_K = 5                     # number of pages to retrieve per query
EMBED_BATCH_SIZE = 4          # keep small on a 16GB consumer GPU
MATRYOSHKA_DIM = 1536          # 768 / 1536 / 2560 supported by NetraEmbed
DEVICE = "cuda"               # falls back to "cpu" automatically in scripts if unavailable

# -----------------------------------------------------------------------
# LoRA domain-adaptation hyperparameters (recipe reused from the M3DR paper)
# -----------------------------------------------------------------------
LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
LEARNING_RATE = 2e-4
NUM_EPOCHS = 2
PER_DEVICE_BATCH_SIZE = 2      # small on purpose for a 16GB card
GRAD_ACCUMULATION_STEPS = 16   # effective batch size ~32, matches the paper's recipe
WARMUP_STEPS = 100
SAVE_STEPS = 500
INFO_NCE_TEMPERATURE = 0.02
