"""
Step 1a — Build membership pairs.

Outputs (saved to the same directory as this script):
  ind_pairs.csv   — all (IPO firm, rival) pairs with their business descriptions

Run this script first, then run ind_ipo_embed.py to compute embeddings.
"""

import sys
from pathlib import Path
import os
import numpy as np
import pandas as pd

import re

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
IPO_FILE = ANN_DIR / "IPO_index.xlsx"
FN_FILE  = INFO_DIR / "FN_Fn001.xlsx"

PAIRS_CSV = OUT_DIR / "ind_pairs.csv"

IDNEX_COL = "INDEX2009"  # column in IPO_roadshow_index_2009.xlsx that uniquely identifies each IPO firm (used for debugging)

SZSH_ONLY= True  # whether to restrict to Shanghai and Shenzhen stock exchanges (exclude NYSE, HKEX, etc.)

def clean_text(s) -> str:
    """Remove newlines and spaces from raw field text."""
    s = str(s or "")
    s = s.replace("\r\n", "").replace("\n", "").replace("\r", "").replace(" ", "")
    return s.strip()


# ── Load data ────────────────────────────────────────────────────────────────
print("Loading STK_LISTEDCOINFOANL.xlsx …")
anl = pd.read_excel(ANL_FILE, dtype=str)

print("Loading IPO_roadshow_index_2009.xlsx …")
ipo = pd.read_excel(IPO_FILE, dtype=str)


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
# Select only the firms in SHanghai and Shenzhen stock exchanges (exclude NYSE, HKEX, etc.)
if SZSH_ONLY:
    ipo=ipo[ipo["Symbol"].str.startswith(["0", "3", "6"])].copy()

# ── 3-digit CSRC code ─────────────────────────────────────────────────────────
# IndustryCode in ANL already encodes the appropriate CSRC version per year
# (CSRC 2001 pre-2012, CSRC 2012 for 2012-2022, association code 2023+).
# Taking the first 3 characters gives the sub-sector level used in the paper.
# For 2023+ IPO firms you may want to use IndustryCodeC instead — see note below.
# 统一使用中上协行业分类
anl["csrc3"] = anl["IndustryCodeD"].astype(str)

# Restrict ANL to IPO listing years only (speeds up the pair-building loop)
valid_years = set(ipo["ListYear"].dropna().astype(int))
anl_filtered = anl[anl["Year"].isin(valid_years)].copy()

# ── Build membership pairs ────────────────────────────────────────────────────
print(f"\nBuilding membership pairs for {len(ipo)} IPO firms …")
pairs_list = []

for idx, ipo_row in ipo.iterrows():
    if (idx + 1) % 100 == 0:
        print(f"  Processing IPO {idx+1:,} / {len(ipo):,} …")
    stkcd = ipo_row["Stkcd"]
    year  = ipo_row["ListYear"]
    id    = ipo_row[IDNEX_COL]
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

        pairs_list.append({
            "ipo_id":         id,
            "ipo_stkcd":      stkcd,
            "rival_stkcd":    rival_row["Symbol"],
            "year":           year,
            "csrc3_code":     ipo_csrc3,
            "ipo_scope":      ipo_scope,
            "ipo_main":       ipo_main,
            "ipo_combined":   ipo_combined,
            "rival_scope":    rival_scope,
            "rival_main":     rival_main,
            "rival_combined": rival_combined,
        })

pairs_df = pd.DataFrame(pairs_list)
print(f"  → {len(pairs_df):,} pairs built")

# ── Save pairs ────────────────────────────────────────────────────────────────
pairs_df.to_csv(PAIRS_CSV, index=False, encoding="utf-8-sig")
print(f"Saved pairs to {PAIRS_CSV.name}")
print("Now run ind_ipo_embed.py to compute embeddings.")
