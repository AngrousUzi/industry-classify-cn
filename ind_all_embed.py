"""
Embed ALL listed companies for every year 2006-2024 using BAAI/bge-m3.

Unlike ind_ipo_embed.py (which only embeds IPO-firm / rival pairs), this script
covers every firm-year observation in the annual report data so the resulting
embeddings can be used for any cross-sectional similarity analysis — including
computing industry-wide quantile thresholds.

Outputs (saved to the same directory as this script):
  ind_all_info.csv        — one row per (Symbol, Year) with
                            Symbol, Year, csrc3, scope, main, combined
  ind_all_embeddings.pkl  — dict mapping text → np.ndarray (float32)
                            Shares the same cache format as ind_embeddings.pkl.
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ── Allow importing from the same directory ───────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from ind_embed_utils import make_client, batch_embed


def clean_text(s) -> str:
    """Remove newlines and extra whitespace from raw field text."""
    s = str(s or "")
    s = s.replace("\r\n", "").replace("\n", "").replace("\r", "").replace(" ", "")
    return s.strip()


# ── Paths ─────────────────────────────────────────────────────────────────────
if os.name == "nt":
    INFO_DIR = Path(r"D:\科研\IPO\PythonProject\basicInfos")
    OUT_DIR  = Path(__file__).parent
else:
    INFO_DIR = Path("./")
    OUT_DIR  = Path("./")

ANL_FILE = INFO_DIR / "STK_LISTEDCOINFOANL.xlsx"

INFO_CSV       = OUT_DIR / "ind_all_info.csv"
EMBEDDINGS_PKL = OUT_DIR / "ind_all_embeddings.pkl"

YEAR_MIN = 2006
YEAR_MAX = 2025

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading {ANL_FILE.name} …")
anl = pd.read_excel(ANL_FILE, dtype=str)
print(f"  {len(anl):,} rows loaded")

# ── Normalise ─────────────────────────────────────────────────────────────────
anl["Symbol"] = anl["Symbol"].str.strip().str.zfill(6)
anl["Year"]   = pd.to_datetime(anl["EndDate"], errors="coerce").dt.year
anl["csrc3"]  = anl["IndustryCode"].astype(str).str.strip()

# ── Filter to 2006-2024 ───────────────────────────────────────────────────────
anl = anl[anl["Year"].between(YEAR_MIN, YEAR_MAX)].copy()
print(f"  {len(anl):,} rows after filtering for {YEAR_MIN}–{YEAR_MAX}")

# ── Build text fields ─────────────────────────────────────────────────────────
anl["scope"]    = anl["BusinessScope"].apply(clean_text)
anl["main"]     = anl["MAINBUSSINESS"].apply(clean_text)
anl["combined"] = "经营范围：" + anl["scope"] + "，主营业务：" + anl["main"]

# Drop rows where scope is empty / nan (can't embed them usefully)
anl = anl[anl["scope"].str.len() > 0].copy()
anl = anl[anl["scope"].str.lower() != "nan"].copy()
print(f"  {len(anl):,} rows with non-empty scope")

# ── Keep relevant columns and deduplicate by (Symbol, Year) ──────────────────
info = (
    anl[["Symbol", "Year", "csrc3", "scope", "main", "combined"]]
    .drop_duplicates(subset=["Symbol", "Year"])
    .reset_index(drop=True)
)
print(f"  {len(info):,} unique (Symbol, Year) observations")
print(f"  Years covered: {sorted(info['Year'].unique())}")

# ── Save info CSV ─────────────────────────────────────────────────────────────
info.to_csv(INFO_CSV, index=False, encoding="utf-8-sig")
print(f"Saved company-year info → {INFO_CSV.name}")

# ── Collect all unique texts across the three dimensions ───────────────────────
all_texts = list(
    set(info["scope"].tolist()) |
    set(info["main"].tolist())  |
    set(info["combined"].tolist())
)
# Remove any residual empty strings
all_texts = [t for t in all_texts if t and t.lower() != "nan"]
print(f"\n{len(all_texts):,} unique texts to embed (scope / main / combined)")

# ── Embed ─────────────────────────────────────────────────────────────────────
client     = make_client()
embeddings = batch_embed(all_texts, client, cache_path=EMBEDDINGS_PKL)

print(f"\nAll done.")
print(f"  Info CSV : {INFO_CSV}  ({len(info):,} rows)")
print(f"  Embeddings cache: {EMBEDDINGS_PKL}  ({len(embeddings):,} entries)")
print("Now run ind_all_similarity_quantile.py to compute quantile thresholds.")
