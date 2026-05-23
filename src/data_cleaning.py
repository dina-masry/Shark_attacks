# Data cleaning 
# 7 Stages : Load -> Shape -> Cloumns -> Nulls -> Invalid -> Validate

import logging
import pandas as pd
import numpy as np

#------ Logging setup ------------------------------------
logging.basicConfig(
level= logging.INFO ,
format="%(asctime)s | %(levelname)s | %(message)s",
datefmt="%H:%M:%S"
)

log = logging.getLogger(__name__)

 #____________________________________________________________
 # Stage 1 : Load
 #____________________________________________________________
def load(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, encoding ='ISO-8859-1')
    log.info(f"[1-Load] Loaded {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df
# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Shape
# ══════════════════════════════════════════════════════════════════════════════
USEFUL_COLS = [
    "Case Number", "Year", "Type", "Country", "Area", "Location",
    "Activity", "Name", "Sex ", "Age", "Injury", "Fatal (Y/N)",
    "Time", "Species ", "Investigator or Source",
]
 
def fix_shape(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
 
    # Drop rows that are completely empty or just separators
    df = df.dropna(how="all")
 
    # Keep only useful columns
    cols = [c for c in USEFUL_COLS if c in df.columns]
    df = df[cols]
 
    dropped = before - len(df)
    log.info(f"[2-Shape] Dropped {dropped:,} filler rows → {len(df):,} rows, {len(df.columns)} cols kept")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — COLUMN NAMES
# ══════════════════════════════════════════════════════════════════════════════
def fix_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.replace(r"_+$", "", regex=True)   # trailing underscores
        .str.replace(r"^_+", "", regex=True)   # leading underscores
    )
    log.info(f"[3-Columns] Renamed columns: {list(df.columns)}")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — TYPES
# ══════════════════════════════════════════════════════════════════════════════
def fix_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
 
    # Year → int (coerce bad values to NaN)
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        # Filter out clearly wrong years
        df = df[df["year"].isna() | df["year"].between(1500, 2025)]
 
    # Age → float (many entries are ranges like "20-30", we take first number)
    if "age" in df.columns:
        df["age"] = (
            df["age"]
            .astype(str)
            .str.extract(r"(\d+)")[0]
            .pipe(pd.to_numeric, errors="coerce")
        )
 
    # Fatal → bool-like category
    if "fatal_y_n" in df.columns:
        df["fatal_y_n"] = df["fatal_y_n"].str.strip().str.upper()
        df["fatal_y_n"] = df["fatal_y_n"].where(df["fatal_y_n"].isin(["Y", "N"]))
 
    # Sex → category
    if "sex" in df.columns:
        df["sex"] = df["sex"].str.strip().str.upper()
        df["sex"] = df["sex"].where(df["sex"].isin(["M", "F"])).astype("category")
 
    log.info("[4-Types] Cast: year→numeric, age→float, fatal→Y/N, sex→category")
    return df
 # ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — TYPES
# ══════════════════════════════════════════════════════════════════════════════
def fix_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to correct dtypes."""
    df = df.copy()
 
    # Year → int (coerce bad values to NaN)
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        # Filter out clearly wrong years
        df = df[df["year"].isna() | df["year"].between(1500, 2025)]
 
    # Age → float (many entries are ranges like "20-30", we take first number)
    if "age" in df.columns:
        df["age"] = (
            df["age"]
            .astype(str)
            .str.extract(r"(\d+)")[0]
            .pipe(pd.to_numeric, errors="coerce")
        )
 
    # Fatal → bool-like category
    if "fatal_y_n" in df.columns:
        df["fatal_y_n"] = df["fatal_y_n"].str.strip().str.upper()
        df["fatal_y_n"] = df["fatal_y_n"].where(df["fatal_y_n"].isin(["Y", "N"]))
 
    # Sex → category
    if "sex" in df.columns:
        df["sex"] = df["sex"].str.strip().str.upper()
        df["sex"] = df["sex"].where(df["sex"].isin(["M", "F"])).astype("category")
 
    log.info("[4-Types] Cast: year→numeric, age→float, fatal→Y/N, sex→category")
    return df
 
 
# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — NULLS
# ══════════════════════════════════════════════════════════════════════════════
def fix_nulls(df: pd.DataFrame) -> pd.DataFrame:
    # Decide: drop, fill, or keep nulls per column.
    df = df.copy()
    before = len(df)
 
    # Drop rows missing both year and country (unusable for analysis)
    df = df.dropna(subset=["year", "country"], how="all")
 
    # Fill: activity unknown
    if "activity" in df.columns:
        df["activity"] = df["activity"].fillna("Unknown")
 
    # Fill: type unknown
    if "type" in df.columns:
        df["type"] = df["type"].fillna("Unknown")
 
    # Keep: age, time, species → missing is valid info (don't impute)
 
    dropped = before - len(df)
    null_summary = df.isnull().sum()
    null_summary = null_summary[null_summary > 0]
    log.info(f"[5-Nulls] Dropped {dropped:,} rows with no year+country")
    log.info(f"[5-Nulls] Remaining nulls:\n{null_summary.to_string()}")
    return df