"""
Step 1 — Build membership pairs and compute embeddings.

Outputs (both saved to the same directory as this script):
  ind_pairs.csv        — all (IPO firm, rival) pairs with their business descriptions
  ind_embeddings.pkl   — dict mapping business-scope text → np.ndarray embedding vector

Run this script once (or re-run to extend the cache with new data).
Step 2 (ind_filter.py) reads these outputs and applies the similarity threshold.
"""

import sys
from pathlib import Path
import os
import numpy as np
import pandas as pd

import re

# Allow importing embed_utils from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from ind_embed_utils import make_client, batch_embed


def clean_text(s) -> str:
    """Remove newlines and spaces from raw field text."""
    s = str(s or "")
    s = s.replace("\r\n", "").replace("\n", "").replace("\r", "").replace(" ", "")
    return s.strip()

# ── Paths ────────────────────────────────────────────────────────────────────
if os.name == "nt":   # Windows
    INFO_DIR = Path(r"D:\科研\IPO\PythonProject\basicInfos")
    OUT_DIR  = Path(__file__).parent 
    ANN_DIR = Path(r"D:\科研\IPO\PythonProject\anns")    
else:
    INFO_DIR = Path("./")
    OUT_DIR  = Path("./")
    ANN_DIR = Path("./")

ANL_FILE = INFO_DIR / "STK_LISTEDCOINFOANL.xlsx"
IPO_FILE = ANN_DIR / "IPO_roadshow_index.xlsx"
FN_FILE  = INFO_DIR / "FN_Fn001.xlsx"

PAIRS_CSV      = OUT_DIR / "ind_pairs.csv"
EMBEDDINGS_PKL = OUT_DIR / "ind_embeddings.pkl"

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading STK_LISTEDCOINFOANL.xlsx …")
anl = pd.read_excel(ANL_FILE, dtype=str)

print("Loading IPO_roadshow_index.xlsx …")
ipo = pd.read_excel(IPO_FILE, dtype=str)

# print("Loading FN_Fn001.xlsx …")
# fn = pd.read_excel(FN_FILE, dtype=str)


# ── Normalise stock codes to 6-digit strings ─────────────────────────────────
anl["Symbol"] = anl["Symbol"].str.strip().str.zfill(6)
ipo["Stkcd"]  = ipo["Stkcd"].str.strip().str.zfill(6)
# fn["Stkcd"] = fn["Stkcd"].str.strip().str.zfill(6)

# ── Extract year ─────────────────────────────────────────────────────────────
# ANL: EndDate is YYYY-MM-DD (e.g. 2015-12-31)
anl["Year"] = pd.to_datetime(anl["EndDate"], errors="coerce").dt.year

# IPO: Listdt may contain "00" day/month (e.g. "1993-12-00") — just take the year part
ipo["ListYear"] = ipo["Listdt"].astype(str).str[:4].apply(
    lambda s: int(s) if s.isdigit() else np.nan
)
# fn["Year"]  = pd.to_datetime(fn["Accper"], errors="coerce").dt.year

# fn["fn_text"] = (
#     fn["Fn00101"].fillna("").apply(clean_text) +
#     fn["Fnother"].fillna("").apply(clean_text)
# )

# ── 3-digit CSRC code ─────────────────────────────────────────────────────────
# IndustryCode in ANL already encodes the appropriate CSRC version per year
# (CSRC 2001 pre-2012, CSRC 2012 for 2012-2022, association code 2023+).
# Taking the first 3 characters gives the sub-sector level used in the paper.
# For 2023+ IPO firms you may want to use IndustryCodeC instead — see note below.
# 统一使用中上协行业分类
anl["csrc3"] = anl["IndustryCodeD"].astype(str)

# Restrict ANL/FN to IPO listing years only (speeds up the pair-building loop)
valid_years = set(ipo["ListYear"].dropna().astype(int))
anl_filtered = anl[anl["Year"].isin(valid_years)].copy()
# fn_filtered  = fn[fn["Year"].isin(valid_years)].copy()

# ── Build membership pairs ────────────────────────────────────────────────────
print(f"\nBuilding membership pairs for {len(ipo)} IPO firms …")
pairs_list = []


for idx, ipo_row in ipo.iterrows():
    if (idx+1) % 100 == 0:
        print(f"  Processing IPO {idx+1:,} / {len(ipo):,} …")
    #     break  # TEMP: limit to first 100 IPOs for testing
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
    ipo_scope = clean_text(ipo_yr.iloc[0]["BusinessScope"])
    ipo_main  = clean_text(ipo_yr.iloc[0].get("MAINBUSSINESS", ""))
    if not ipo_scope or ipo_scope.lower() == "nan":
        continue
    ipo_combined = f"经营范围：{ipo_scope}，主营业务：{ipo_main}"
    # ipo_fn_row = fn_filtered[(fn_filtered["Stkcd"] == stkcd) & (fn_filtered["Year"] == year)]
    # ipo_fn = ipo_fn_row.iloc[0]["fn_text"] if not ipo_fn_row.empty else ""

    # All listed firms in the same 3-digit CSRC code in the same year (excluding the IPO firm itself)
    rivals = anl_filtered[
        (anl_filtered["Year"]   == year) &
        (anl_filtered["csrc3"]  == ipo_csrc3) &
        (anl_filtered["Symbol"] != stkcd)
    ]

    for _, rival_row in rivals.iterrows():
        rival_scope = clean_text(rival_row["BusinessScope"])
        if not rival_scope or rival_scope.lower() == "nan":
            continue
        rival_main     = clean_text(rival_row.get("MAINBUSSINESS", ""))
        rival_combined = f"经营范围：{rival_scope}，主营业务：{rival_main}"
        # rival_fn_row = fn_filtered[(fn_filtered["Stkcd"] == rival_row["Symbol"]) & (fn_filtered["Year"] == year)]
        # rival_fn = rival_fn_row.iloc[0]["fn_text"] if not rival_fn_row.empty else ""

        pairs_list.append({
            "ipo_stkcd":      stkcd,
            "rival_stkcd":    rival_row["Symbol"],
            "year":           year,
            "csrc3_code":     ipo_csrc3,
            "ipo_scope":      ipo_scope,
            "ipo_main":       ipo_main,
            "ipo_combined":   ipo_combined,
            # "ipo_fn":         ipo_fn,
            "rival_scope":    rival_scope,
            "rival_main":     rival_main,
            "rival_combined": rival_combined,
            # "rival_fn":       rival_fn,
        })

pairs_df = pd.DataFrame(pairs_list)
print(f"  → {len(pairs_df):,} pairs built")

# ── Save pairs ────────────────────────────────────────────────────────────────
pairs_df.to_csv(PAIRS_CSV, index=False, encoding="utf-8-sig")
print(f"Saved pairs to {PAIRS_CSV.name}")

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
