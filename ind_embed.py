"""
Step 1 — Build membership pairs and compute embeddings.

Outputs (both saved to the same directory as this script):
  pairs.csv        — all (IPO firm, rival) pairs with their business descriptions
  embeddings.pkl   — dict mapping business-scope text → np.ndarray embedding vector

Run this script once (or re-run to extend the cache with new data).
Step 2 (step2_filter.py) reads these outputs and applies the similarity threshold.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow importing embed_utils from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from ind_embed_utils import make_client, batch_embed

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(r"D:\科研\IPO\PythonProject\basicInfos")
OUT_DIR  = Path(__file__).parent          # anns/

ANL_FILE = DATA_DIR / "STK_LISTEDCOINFOANL.xlsx"
IPO_FILE = DATA_DIR / "IPO_Ipobasic.xlsx"

PAIRS_CSV      = OUT_DIR / "pairs.csv"
EMBEDDINGS_PKL = OUT_DIR / "embeddings.pkl"

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading STK_LISTEDCOINFOANL.xlsx …")
anl = pd.read_excel(ANL_FILE, dtype={"Symbol": str})

print("Loading IPO_Ipobasic.xlsx …")
ipo = pd.read_excel(IPO_FILE, dtype={"Stkcd": str})

# ── Normalise stock codes to 6-digit strings ─────────────────────────────────
anl["Symbol"] = anl["Symbol"].str.strip().str.zfill(6)
ipo["Stkcd"]  = ipo["Stkcd"].str.strip().str.zfill(6)

# ── Extract year ─────────────────────────────────────────────────────────────
# ANL: EndDate is YYYY-MM-DD (e.g. 2015-12-31)
anl["Year"] = pd.to_datetime(anl["EndDate"], errors="coerce").dt.year

# IPO: Listdt may contain "00" day/month (e.g. "1993-12-00") — just take the year part
ipo["ListYear"] = ipo["Listdt"].astype(str).str[:4].apply(
    lambda s: int(s) if s.isdigit() else np.nan
)

# ── 3-digit CSRC code ─────────────────────────────────────────────────────────
# IndustryCode in ANL already encodes the appropriate CSRC version per year
# (CSRC 2001 pre-2012, CSRC 2012 for 2012-2022, association code 2023+).
# Taking the first 3 characters gives the sub-sector level used in the paper.
# For 2023+ IPO firms you may want to use IndustryCodeC instead — see note below.
anl["csrc3"] = anl["IndustryCode"].astype(str).str[:3]

# Restrict ANL to IPO listing years only (speeds up the pair-building loop)
valid_years = set(ipo["ListYear"].dropna().astype(int))
anl_filtered = anl[anl["Year"].isin(valid_years)].copy()

# ── Build membership pairs ────────────────────────────────────────────────────
print(f"\nBuilding membership pairs for {len(ipo)} IPO firms …")
pairs_list = []

for idx, ipo_row in ipo.iterrows():
    stkcd = ipo_row["Stkcd"]
    year  = ipo_row["ListYear"]

    if pd.isna(year):
        continue
    year = int(year)

    # IPO firm info in its listing year
    ipo_yr = anl_filtered[(anl_filtered["Symbol"] == stkcd) & (anl_filtered["Year"] == year)]
    if ipo_yr.empty:
        continue

    ipo_csrc3 = ipo_yr.iloc[0]["csrc3"]
    ipo_scope = str(ipo_yr.iloc[0]["BusinessScope"] or "").strip()
    if not ipo_scope or ipo_scope.lower() == "nan":
        continue

    # All listed firms in the same 3-digit CSRC code in the same year (excluding the IPO firm itself)
    rivals = anl_filtered[
        (anl_filtered["Year"]   == year) &
        (anl_filtered["csrc3"]  == ipo_csrc3) &
        (anl_filtered["Symbol"] != stkcd)
    ]

    for _, rival_row in rivals.iterrows():
        rival_scope = str(rival_row["BusinessScope"] or "").strip()
        if not rival_scope or rival_scope.lower() == "nan":
            continue

        pairs_list.append({
            "ipo_stkcd":   stkcd,
            "rival_stkcd": rival_row["Symbol"],
            "year":        year,
            "csrc3_code":  ipo_csrc3,
            "ipo_scope":   ipo_scope,
            "rival_scope": rival_scope,
        })

pairs_df = pd.DataFrame(pairs_list)
print(f"  → {len(pairs_df):,} pairs built")

# ── Save pairs ────────────────────────────────────────────────────────────────
pairs_df.to_csv(PAIRS_CSV, index=False, encoding="utf-8-sig")
print(f"Saved pairs to {PAIRS_CSV.name}")

# ── Collect unique texts and embed ───────────────────────────────────────────
all_texts = list(
    set(pairs_df["ipo_scope"].tolist()) | set(pairs_df["rival_scope"].tolist())
)
print(f"\n{len(all_texts):,} unique business-scope texts to embed")

client     = make_client()
embeddings = batch_embed(all_texts, client, cache_path=EMBEDDINGS_PKL)

print(f"\nAll done. Embeddings cache: {EMBEDDINGS_PKL.name} ({len(embeddings):,} entries)")
print("Now run step2_filter.py to apply the similarity threshold.")
