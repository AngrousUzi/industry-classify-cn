"""
Step 2 — Compute cosine similarity and label rivals by threshold.

Reads:
  ind_pairs.csv        — built by ind_embed.py
  ind_embeddings.pkl   — built by ind_embed.py

Writes:
  ind_rival_similarity.csv — columns: ipo_stkcd, rival_stkcd, year, csrc3_code,
                          sim_scope, sim_main, sim_combined,
                          selected_scope, selected_main, selected_combined

All pairs are kept; the selected_* columns flag whether similarity >= THRESHOLD
for each dimension. Adjust THRESHOLD and re-run freely — no API calls are made.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# ── Settings (the only thing you need to change between runs) ─────────────────
THRESHOLD = 0.7968   # keep pairs with cosine similarity >= this value

# ── Paths ─────────────────────────────────────────────────────────────────────
if os.name == "nt":
    OUT_DIR = Path(__file__).parent
else:
    OUT_DIR = Path("./")
PAIRS_CSV      = OUT_DIR / "ind_pairs.csv"
EMBEDDINGS_PKL = OUT_DIR / "ind_embeddings.pkl"
RESULT_CSV     = OUT_DIR / f"ind_rival_similarity_{THRESHOLD}.csv" 
RESULT_CSV_WITH_CONTENT = OUT_DIR / f"ind_rival_similarity_with_content_{THRESHOLD}.csv"
RETENTION_BY_INDUSTRY_CSV = OUT_DIR / f"ind_rival_retention_by_csrc3_{THRESHOLD}.csv"
# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading pairs …")
pairs_df = pd.read_csv(PAIRS_CSV, dtype={"ipo_stkcd": str, "rival_stkcd": str})
print(f"  {len(pairs_df):,} pairs")

print("Loading embeddings …")
with open(EMBEDDINGS_PKL, "rb") as f:
    embeddings: dict[str, np.ndarray] = pickle.load(f)
print(f"  {len(embeddings):,} cached embeddings")
# print(embeddings.items())
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

# DIMENSIONS = ["scope", "main", "combined", "fn"]
DIMENSIONS = ["scope", "main", "combined"]
# For each dimension, compute row-wise cosine sim and add selected_* flag
result = pairs_df.copy()

for dim in DIMENSIONS:
    ipo_col   = f"ipo_{dim}"
    rival_col = f"rival_{dim}"

    has_emb = np.array(
        [(embeddings.get(row[ipo_col]) is not None and
          embeddings.get(row[rival_col]) is not None)
         for _, row in result.iterrows()],
        dtype=bool,
    )
    n_missing = (~has_emb).sum()
    if n_missing:
        print(f"  [{dim}] Warning: {n_missing:,} pairs have a missing embedding.")

    sims = np.full(len(result), np.nan)
    if has_emb.any():
        idx   = np.where(has_emb)[0]
        texts = result.iloc[idx]
        mat_ipo   = np.vstack([embeddings[t] for t in texts[ipo_col]])
        mat_rival = np.vstack([embeddings[t] for t in texts[rival_col]])
        sims[idx] = cosine_sim_matrix(mat_ipo, mat_rival)

    result[f"sim_{dim}"]      = sims
    result[f"selected_{dim}"] = (sims >= THRESHOLD)

result = result.sort_values(["ipo_stkcd", "year"], ascending=True)

print(f"\nThreshold = {THRESHOLD}")
for dim in DIMENSIONS:
    n_sel = result[f"selected_{dim}"].sum()
    n_val = result[f"sim_{dim}"].notna().sum()
    print(f"  [{dim:8s}]  valid: {n_val:,}  selected: {n_sel:,}  ({n_sel / max(n_val, 1):.1%})")

# ── Retention by industry (csrc3_code) ──────────────────────────────────────
retention_rows = []
for csrc3_code, grp in result.groupby("csrc3_code", dropna=False):
    row = {"csrc3_code": csrc3_code}
    for dim in DIMENSIONS:
        n_val = int(grp[f"sim_{dim}"].notna().sum())
        n_sel = int(grp[f"selected_{dim}"].sum())
        row[f"valid_{dim}"] = n_val
        row[f"selected_{dim}"] = n_sel
        row[f"retention_{dim}"] = (n_sel / n_val) if n_val > 0 else np.nan
    retention_rows.append(row)

retention_df = pd.DataFrame(retention_rows)
retention_df = retention_df.sort_values("csrc3_code", ascending=True)

print("\nRetention by csrc3_code (top 20 by valid_combined):")
preview_cols = [
    "csrc3_code",
    "valid_scope", "selected_scope", "retention_scope",
    "valid_main", "selected_main", "retention_main",
    "valid_combined", "selected_combined", "retention_combined",
]
print(
    retention_df.sort_values("valid_combined", ascending=False)
    .head(20)[preview_cols]
    .to_string(index=False)
)

retention_df.to_csv(
    RETENTION_BY_INDUSTRY_CSV,
    index=False,
    encoding="utf-8-sig",
    float_format="%.6f",
)
print(f"Saved → {RETENTION_BY_INDUSTRY_CSV.name}")

# ── Save ──────────────────────────────────────────────────────────────────────
out_cols = (
    ["ipo_stkcd", "rival_stkcd", "year", "csrc3_code"] +
    [f"sim_{d}" for d in DIMENSIONS] +
    [f"selected_{d}" for d in DIMENSIONS]
)
result[out_cols].to_csv(RESULT_CSV, index=False, encoding="utf-8-sig", float_format="%.6f")
print(f"\nSaved → {RESULT_CSV.name}")

out_cols_with_content = (
    ["ipo_stkcd", "rival_stkcd", "year", "csrc3_code"] +
    [f"ipo_{d}" for d in DIMENSIONS] +
    [f"rival_{d}" for d in DIMENSIONS] +
    [f"sim_{d}" for d in DIMENSIONS] +
    [f"selected_{d}" for d in DIMENSIONS]
)
result[out_cols_with_content].to_csv(RESULT_CSV_WITH_CONTENT, index=False, encoding="utf-8-sig", float_format="%.6f")
print(f"Saved → {RESULT_CSV_WITH_CONTENT.name}")