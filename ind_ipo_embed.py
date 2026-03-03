"""
Step 1b — Compute embeddings for all unique texts in ind_pairs.csv.

Reads:
  ind_pairs.csv        — produced by ind_ipo_pair.py
Outputs:
  ind_embeddings.pkl   — dict mapping business-scope text → np.ndarray embedding vector

Run ind_ipo_pair.py first, then this script.
Step 2 (ind_filter.py) reads these outputs and applies the similarity threshold.
"""

import sys
from pathlib import Path
import os
import pandas as pd

# Allow importing embed_utils from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from ind_embed_utils import make_client, batch_embed

# ── Paths ────────────────────────────────────────────────────────────────────
if os.name == "nt":   # Windows
    OUT_DIR = Path(__file__).parent
else:
    OUT_DIR = Path("./")

PAIRS_CSV      = OUT_DIR / "ind_pairs.csv"
EMBEDDINGS_PKL = OUT_DIR / "ind_embeddings.pkl"

# ── Load pairs ────────────────────────────────────────────────────────────────
print(f"Loading pairs from {PAIRS_CSV.name} …")
pairs_df = pd.read_csv(PAIRS_CSV, dtype=str)
print(f"  → {len(pairs_df):,} pairs loaded")

# ── Collect unique texts and embed ───────────────────────────────────────────
all_texts = list(
    set(pairs_df["ipo_scope"].tolist())    | set(pairs_df["rival_scope"].tolist())    |
    set(pairs_df["ipo_main"].tolist())     | set(pairs_df["rival_main"].tolist())     |
    set(pairs_df["ipo_combined"].tolist()) | set(pairs_df["rival_combined"].tolist()) 
    # set(pairs_df["ipo_fn"].tolist())       | set(pairs_df["rival_fn"].tolist())
)
print(f"\n{len(all_texts):,} unique texts to embed (scope / main / combined / fn)")

client     = make_client()
embeddings = batch_embed(all_texts, client, cache_path=EMBEDDINGS_PKL)

print(f"\nAll done. Embeddings cache: {EMBEDDINGS_PKL.name} ({len(embeddings):,} entries)")
print("Now run ind_filter.py to apply the similarity threshold.")
