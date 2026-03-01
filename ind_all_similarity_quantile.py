"""
Compute cosine similarity distributions for all listed companies (2006-2024)
and derive data-driven similarity thresholds following Hoberg & Phillips (2016).

METHODOLOGY
───────────
The paper says:
  "2.89% of all possible firm pairs are in the same industry based on CSRC codes.
   Setting the minimum similarity threshold to 0.727 generates 2.89% membership
   pairs — the same fraction as the CSRC codes."

This means:
  (1) Compute CSRC coverage rate:
        csrc_rate = Σ_k C(n_k,2) / C(N,2)    (pooled across years)
      where N = all listed firms in a year, n_k = firms in CSRC-3 industry k.
      Check whether our sample gives ~2.89%.

  (2) For each year, compute cosine similarity for EVERY possible pair of firms
      (upper triangle of the N×N similarity matrix, vectorised via mat @ mat.T).
      This is O(N²) but fully vectorised — for ~5,000 firms/year the matrix is
      ~100 MB, well within memory.

  (3) Pool all pairwise similarities across years.  The H&P threshold T is:
        T = (1 − target_rate)-th percentile of ALL pairwise similarities
      i.e., the top target_rate fraction of all pairs by similarity score
      become the rival set, replacing the coarse CSRC membership.

  (4) We also report the within-CSRC similarity distribution separately as a
      descriptive quantile table (only pairs sharing the same 3-digit CSRC code).

Reads:
  ind_all_info.csv        — built by ind_all_embed.py
  ind_all_embeddings.pkl  — built by ind_all_embed.py

Writes:
  ind_all_csrc_coverage.csv             — CSRC coverage rate by year
  ind_all_sim_quantiles_all.csv         — quantile table of ALL pairwise sims
  ind_all_sim_quantiles_within.csv      — quantile table of within-CSRC sims only
  ind_all_sim_quantiles_by_year.csv     — all-pairs quantiles broken out by year
  ind_all_sim_threshold_hp.csv          — H&P thresholds for a grid of target rates
  ind_all_sim_pairs_within.csv          — within-CSRC pairs with their sim scores
                                         (toggle SAVE_PAIRS_WITHIN to False to skip)

"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import os

# ── Settings ──────────────────────────────────────────────────────────────────
DIMENSIONS = ["scope", "main", "combined"]

# Percentiles to report (0-100 scale)
QUANTILES = [1, 5, 10, 25, 50, 75, 80, 85, 90, 95, 99]

# Paper's reference coverage rate; set None to use our sample's observed rate
TARGET_CSRC_RATE = 0.0289  # 2.89% as stated in the paper

# Save within-CSRC pairs with their sim scores (can be large but manageable)
SAVE_PAIRS_WITHIN = True

# ── Paths ─────────────────────────────────────────────────────────────────────
if os.name == "nt":
    OUT_DIR = Path(__file__).parent
else:
    OUT_DIR = Path("./")
INFO_CSV       = OUT_DIR / "ind_all_info.csv"
EMBEDDINGS_PKL = OUT_DIR / "ind_all_embeddings.pkl"

COVERAGE_CSV      = OUT_DIR / "ind_all_csrc_coverage.csv"
ALL_QUANTILE_CSV  = OUT_DIR / "ind_all_sim_quantiles_all.csv"
WTH_QUANTILE_CSV  = OUT_DIR / "ind_all_sim_quantiles_within.csv"
BY_YEAR_CSV       = OUT_DIR / "ind_all_sim_quantiles_by_year.csv"
HP_THRESH_CSV     = OUT_DIR / "ind_all_sim_threshold_hp.csv"
PAIRS_WITHIN_CSV  = OUT_DIR / "ind_all_sim_pairs_within.csv"


# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading company-year info …")
info = pd.read_csv(INFO_CSV, dtype={"Symbol": str, "Year": int, "csrc3": str})
info["Symbol"] = info["Symbol"].str.zfill(6)
print(f"  {len(info):,} firm-year observations")

print("Loading embeddings …")
with open(EMBEDDINGS_PKL, "rb") as f:
    embeddings: dict[str, np.ndarray] = pickle.load(f)
print(f"  {len(embeddings):,} cached embeddings")


# ── Step 1: CSRC coverage rate by year ───────────────────────────────────────
print("\nComputing CSRC coverage rates by year …")
coverage_rows = []

for year, yr_grp in info.groupby("Year", sort=True):
    N = len(yr_grp)
    all_pairs  = N * (N - 1) // 2
    csrc_pairs = int(yr_grp.groupby("csrc3").size()
                     .apply(lambda n: n * (n - 1) // 2).sum())
    rate = csrc_pairs / all_pairs if all_pairs > 0 else np.nan
    coverage_rows.append({"year": year, "n_firms": N,
                           "all_pairs": all_pairs, "csrc_pairs": csrc_pairs,
                           "csrc_rate": rate})

coverage_df = pd.DataFrame(coverage_rows)
coverage_overall_rate = (
    coverage_df["csrc_pairs"].sum() / coverage_df["all_pairs"].sum()
)
coverage_df.to_csv(COVERAGE_CSV, index=False, encoding="utf-8-sig", float_format="%.6f")
print(f"Saved → {COVERAGE_CSV.name}")
print(coverage_df.to_string(index=False))
print(f"\nOverall (pooled) CSRC coverage rate : {coverage_overall_rate:.4%}")
if TARGET_CSRC_RATE is not None:
    print(f"Paper's reference rate              : {TARGET_CSRC_RATE:.4%}")
target_rate = TARGET_CSRC_RATE if TARGET_CSRC_RATE is not None else coverage_overall_rate
print(f"Calibration target rate             : {target_rate:.4%}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    return mat / norms


def upper_tri_values(sim_mat: np.ndarray):
    """Return (values, row_idx, col_idx) for the strict upper triangle."""
    n = sim_mat.shape[0]
    ui, uj = np.triu_indices(n, k=1)
    return sim_mat[ui, uj], ui, uj


# ── Step 2: Compute similarities for ALL pairs, per year ──────────────────────
# For EACH year:
#   - Build the N×D embedding matrix for all N firms in that year
#   - Compute the full N×N cosine similarity matrix (vectorised: mat_norm @ mat_norm.T)
#   - Extract upper triangle → N*(N-1)/2 values for ALL possible pairs
#   - Label which pairs share the same 3-digit CSRC code (is_within flag)
#
# Accumulators:
#   all_sims_global[dim]   — ALL pairwise sim values pooled across years
#                            → used to derive the H&P threshold
#   all_sims_within[dim]   — within-CSRC sim values only
#                            → used for the descriptive quantile table
#   year_sims_all[dim]     — per-year: ALL pairs (for by-year quantile table)

all_sims_global = {d: [] for d in DIMENSIONS}
all_sims_within = {d: [] for d in DIMENSIONS}
year_sims_all   = {d: {} for d in DIMENSIONS}

pairs_within_rows = []

years = sorted(info["Year"].unique())
print(f"\nProcessing {len(years)} years (all-pairs similarity) …")

for year in years:
    yr = info[info["Year"] == year].reset_index(drop=True)
    N  = len(yr)
    print(f"  year={year}  N={N:,}  all_pairs={N*(N-1)//2:,}", flush=True)

    csrc_arr = yr["csrc3"].to_numpy()
    sym_arr  = yr["Symbol"].to_numpy()

    dim_data = {}  # dim -> dict with idx_ok, sim_mat, etc.

    for dim in DIMENSIONS:
        texts   = yr[dim].tolist()
        vecs    = [embeddings.get(t) for t in texts]
        has_emb = np.array([v is not None for v in vecs], dtype=bool)
        idx_ok  = np.where(has_emb)[0]

        if len(idx_ok) < 2:
            dim_data[dim] = None
            continue

        mat      = np.vstack([vecs[i] for i in idx_ok]).astype(np.float32)
        mat_norm = normalise(mat)
        sim_mat  = mat_norm @ mat_norm.T   # (M, M)

        sims_ut, ui, uj = upper_tri_values(sim_mat)

        csrc_sub  = csrc_arr[idx_ok]
        is_within = (csrc_sub[ui] == csrc_sub[uj])

        dim_data[dim] = {
            "idx_ok":    idx_ok,
            "sim_mat":   sim_mat,
            "sims_ut":   sims_ut,
            "ui": ui, "uj": uj,
            "is_within": is_within,
            "csrc_sub":  csrc_sub,
        }

        all_sims_global[dim].extend(sims_ut.tolist())
        year_sims_all[dim].setdefault(year, []).extend(sims_ut.tolist())
        all_sims_within[dim].extend(sims_ut[is_within].tolist())

    # ── Collect within-CSRC pair rows ────────────────────────────────────────
    if SAVE_PAIRS_WITHIN:
        ref = dim_data.get("combined") or dim_data.get("scope")
        if ref is not None:
            idx_ok_ref = ref["idx_ok"]
            ui_ref, uj_ref = ref["ui"], ref["uj"]
            is_within_ref  = ref["is_within"]
            sym_sub  = sym_arr[idx_ok_ref]
            csrc_sub = ref["csrc_sub"]

            for k in range(len(ui_ref)):
                if not is_within_ref[k]:
                    continue
                i_pos  = int(ui_ref[k])
                j_pos  = int(uj_ref[k])
                orig_i = idx_ok_ref[i_pos]
                orig_j = idx_ok_ref[j_pos]

                row = {
                    "stkcd_i": sym_sub[i_pos],
                    "stkcd_j": sym_sub[j_pos],
                    "year":    year,
                    "csrc3":   csrc_sub[i_pos],
                }
                for dim in DIMENSIONS:
                    dd = dim_data.get(dim)
                    if dd is None:
                        row[f"sim_{dim}"] = np.nan
                        continue
                    pi = np.where(dd["idx_ok"] == orig_i)[0]
                    pj = np.where(dd["idx_ok"] == orig_j)[0]
                    row[f"sim_{dim}"] = (
                        float(dd["sim_mat"][pi[0], pj[0]])
                        if len(pi) and len(pj) else np.nan
                    )
                pairs_within_rows.append(row)

print(f"\nAll pairs collected   (scope): {len(all_sims_global['scope']):,}")
print(f"Within-CSRC collected (scope): {len(all_sims_within['scope']):,}")


# ── Step 3: Quantile tables ───────────────────────────────────────────────────
pct_labels = [f"p{q:02d}" for q in QUANTILES]


def compute_quantiles(values: list) -> dict:
    arr = np.array(values, dtype=np.float32)
    if len(arr) == 0:
        return {lbl: np.nan for lbl in pct_labels}
    return {lbl: float(np.percentile(arr, q)) for lbl, q in zip(pct_labels, QUANTILES)}


# All-pairs quantile table
print("\nComputing all-pairs quantile table …")
all_q_rows = []
for dim in DIMENSIONS:
    row = {"dimension": dim, "n_pairs": len(all_sims_global[dim])}
    row.update(compute_quantiles(all_sims_global[dim]))
    all_q_rows.append(row)
all_q_df = pd.DataFrame(all_q_rows, columns=["dimension", "n_pairs"] + pct_labels)
all_q_df.to_csv(ALL_QUANTILE_CSV, index=False, encoding="utf-8-sig", float_format="%.6f")
print(f"Saved → {ALL_QUANTILE_CSV.name}")
print(all_q_df.to_string(index=False))

# Within-CSRC quantile table
print("\nComputing within-CSRC quantile table …")
wth_q_rows = []
for dim in DIMENSIONS:
    row = {"dimension": dim, "n_pairs": len(all_sims_within[dim])}
    row.update(compute_quantiles(all_sims_within[dim]))
    wth_q_rows.append(row)
wth_q_df = pd.DataFrame(wth_q_rows, columns=["dimension", "n_pairs"] + pct_labels)
wth_q_df.to_csv(WTH_QUANTILE_CSV, index=False, encoding="utf-8-sig", float_format="%.6f")
print(f"Saved → {WTH_QUANTILE_CSV.name}")
print(wth_q_df.to_string(index=False))

# By-year quantile table (all pairs)
print("\nComputing by-year quantile table (all pairs) …")
by_year_rows = []
for year in years:
    for dim in DIMENSIONS:
        vals = year_sims_all[dim].get(year, [])
        row  = {"year": year, "dimension": dim, "n_pairs": len(vals)}
        row.update(compute_quantiles(vals))
        by_year_rows.append(row)
by_year_df = pd.DataFrame(by_year_rows, columns=["year", "dimension", "n_pairs"] + pct_labels)
by_year_df.to_csv(BY_YEAR_CSV, index=False, encoding="utf-8-sig", float_format="%.6f")
print(f"Saved → {BY_YEAR_CSV.name}")

# ── Step 4: Hoberg-Phillips calibrated thresholds ────────────────────────────
# T = (1 − target_rate)-th percentile of ALL pairwise similarities.
# The top target_rate fraction of all pairs (by sim) become the rival set.

print("\nComputing Hoberg-Phillips calibrated thresholds …")

target_grid = sorted(set(
    [round(r, 4) for r in np.arange(0.005, 0.10, 0.005)] +
    [TARGET_CSRC_RATE, coverage_overall_rate]
) - {None})

hp_rows = []
for dim in DIMENSIONS:
    sims_arr = np.array(all_sims_global[dim], dtype=np.float32)
    n_all    = len(sims_arr)

    for tgt in target_grid:
        q            = (1.0 - tgt) * 100.0
        threshold    = float(np.percentile(sims_arr, q)) if n_all > 0 else np.nan
        n_surviving  = int((sims_arr >= threshold).sum()) if n_all > 0 else 0
        actual_rate  = n_surviving / n_all if n_all > 0 else np.nan
        hp_rows.append({
            "dimension":    dim,
            "target_rate":  tgt,
            "threshold_T":  threshold,
            "n_all_pairs":  n_all,
            "n_surviving":  n_surviving,
            "actual_rate":  actual_rate,
        })

hp_df = pd.DataFrame(hp_rows)
hp_df.to_csv(HP_THRESH_CSV, index=False, encoding="utf-8-sig", float_format="%.6f")
print(f"Saved → {HP_THRESH_CSV.name}")

print("\n── Calibrated thresholds (paper 2.89% vs sample rate) ──")
for dim in DIMENSIONS:
    sub = hp_df[hp_df["dimension"] == dim]
    for tgt, label in [(TARGET_CSRC_RATE,      "paper 2.89%"),
                       (coverage_overall_rate,  f"sample {coverage_overall_rate:.4%}")]:
        if tgt is None:
            continue
        row = sub[(sub["target_rate"] - tgt).abs() < 1e-9]
        if not row.empty:
            r = row.iloc[0]
            print(f"  [{dim:8s}]  target={tgt:.4%}  ({label})"
                  f"  T={r['threshold_T']:.4f}"
                  f"  surviving {r['n_surviving']:,}/{r['n_all_pairs']:,}"
                  f"  actual_rate={r['actual_rate']:.4%}")


# ── Save within-CSRC pairs CSV ────────────────────────────────────────────────
if SAVE_PAIRS_WITHIN and pairs_within_rows:
    print(f"\nSaving {len(pairs_within_rows):,} within-CSRC pair rows …")
    pairs_df = pd.DataFrame(pairs_within_rows)
    pairs_df = pairs_df.sort_values(["year", "csrc3", "stkcd_i", "stkcd_j"]).reset_index(drop=True)
    pairs_df.to_csv(PAIRS_WITHIN_CSV, index=False, encoding="utf-8-sig", float_format="%.6f")
    print(f"Saved → {PAIRS_WITHIN_CSV.name}")

print("\nDone.")
