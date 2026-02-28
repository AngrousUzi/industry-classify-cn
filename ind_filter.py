"""
Step 2 — Compute cosine similarity and filter rivals by threshold.

Reads:
  pairs.csv        — built by step1_embed.py
  embeddings.pkl   — built by step1_embed.py

Writes:
  rival_similarity.csv — columns: ipo_stkcd, rival_stkcd, year, csrc3_code, similarity

Adjust THRESHOLD below and re-run this script freely — no API calls are made.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# ── Settings (the only thing you need to change between runs) ─────────────────
THRESHOLD = 0.5   # keep pairs with cosine similarity >= this value

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent

PAIRS_CSV      = OUT_DIR / "pairs.csv"
EMBEDDINGS_PKL = OUT_DIR / "embeddings.pkl"
RESULT_CSV     = OUT_DIR / "rival_similarity.csv"

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading pairs …")
pairs_df = pd.read_csv(PAIRS_CSV, dtype={"ipo_stkcd": str, "rival_stkcd": str})
print(f"  {len(pairs_df):,} pairs")

print("Loading embeddings …")
with open(EMBEDDINGS_PKL, "rb") as f:
    embeddings: dict[str, np.ndarray] = pickle.load(f)
print(f"  {len(embeddings):,} cached embeddings")

# ── Vectorised cosine similarity ──────────────────────────────────────────────
def cosine_sim_matrix(vecs_a: np.ndarray, vecs_b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two (N, D) arrays."""
    norms_a = np.linalg.norm(vecs_a, axis=1, keepdims=True)
    norms_b = np.linalg.norm(vecs_b, axis=1, keepdims=True)
    # Avoid division by zero
    norms_a = np.where(norms_a == 0, 1e-10, norms_a)
    norms_b = np.where(norms_b == 0, 1e-10, norms_b)
    return np.sum((vecs_a / norms_a) * (vecs_b / norms_b), axis=1)


print("Computing similarities …")

# Look up embeddings for every row
MISSING = np.zeros(1)   # sentinel for missing embeddings

ipo_vecs   = np.array([embeddings.get(t, None) is not None
                        and embeddings[t] or None
                        for t in pairs_df["ipo_scope"]], dtype=object)
rival_vecs = np.array([embeddings.get(t, None) is not None
                        and embeddings[t] or None
                        for t in pairs_df["rival_scope"]], dtype=object)

# Build boolean mask of rows where both embeddings exist
has_emb = np.array(
    [(embeddings.get(row["ipo_scope"]) is not None and
      embeddings.get(row["rival_scope"]) is not None)
     for _, row in pairs_df.iterrows()],
    dtype=bool,
)

n_missing = (~has_emb).sum()
if n_missing:
    print(f"  Warning: {n_missing:,} pairs have a missing embedding and will be skipped.")

valid_pairs = pairs_df[has_emb].copy()

# Stack into matrices for fast batch computation
mat_ipo   = np.vstack([embeddings[t] for t in valid_pairs["ipo_scope"]])
mat_rival = np.vstack([embeddings[t] for t in valid_pairs["rival_scope"]])

sims = cosine_sim_matrix(mat_ipo, mat_rival)
valid_pairs["similarity"] = sims

# ── Apply threshold ───────────────────────────────────────────────────────────
result = valid_pairs[valid_pairs["similarity"] >= THRESHOLD].copy()
result = result[["ipo_stkcd", "rival_stkcd", "year", "csrc3_code", "similarity"]]
result = result.sort_values(["ipo_stkcd", "year", "similarity"], ascending=[True, True, False])

print(f"\nThreshold = {THRESHOLD}")
print(f"  Pairs before filter : {len(valid_pairs):,}")
print(f"  Pairs after filter  : {len(result):,}")
print(f"  Retention rate      : {len(result) / max(len(valid_pairs), 1):.1%}")

# ── Save ──────────────────────────────────────────────────────────────────────
result.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig", float_format="%.6f")
print(f"\nSaved → {RESULT_CSV.name}")
