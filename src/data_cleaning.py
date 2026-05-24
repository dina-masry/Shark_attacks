# Data cleaning 
# 7 Stages : Load -> Shape -> Cloumns -> Nulls -> Invalid -> Validate

import logging
import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# Logging Setup
# ══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
level= logging.INFO ,
format="%(asctime)s | %(levelname)s | %(message)s",
datefmt="%H:%M:%S"
)

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Load
# ══════════════════════════════════════════════════════════════════════════════
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
] # Without metadata columns
 
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
# Stage 3 — Columns Names
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
# Stage 4 — Types
# ══════════════════════════════════════════════════════════════════════════════
def fix_types(df: pd.DataFrame) -> pd.DataFrame:
    # Cast columns to correct dtypes
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
# Stage 5 — Nulls
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

# ══════════════════════════════════════════════════════════════════════════════
# Stage 6 — Invalid Values
# ══════════════════════════════════════════════════════════════════════════════
def fix_invalid(df: pd.DataFrame) -> pd.DataFrame:
    """Fix dirty categories, outliers, and obvious errors."""
    df = df.copy()
 
    # Age: remove impossible values
    if "age" in df.columns:
        invalid_age = df["age"].notna() & ~df["age"].between(1, 100)
        log.info(f"[6-Invalid] Nullifying {invalid_age.sum()} impossible age values")
        df.loc[invalid_age, "age"] = np.nan
 
    # Type: normalize messy categories
    if "type" in df.columns:
        type_map = {
            "Boating": "Watercraft",
            "Sea Disaster": "Watercraft",
            "Boat": "Watercraft",
            "Invalid": "Unknown",
            "Questionable": "Unknown",
        }
        df["type"] = df["type"].replace(type_map)
 
    # Country: strip and title-case
    if "country" in df.columns:
        df["country"] = df["country"].str.strip().str.title()
 
    # Deduplicate
    before = len(df)
    df = df.drop_duplicates()
    log.info(f"[6-Invalid] Removed {before - len(df)} duplicate rows")
 
    return df

# ══════════════════════════════════════════════════════════════════════════════
# Stage 7 — Validate
# ══════════════════════════════════════════════════════════════════════════════
def validate(df: pd.DataFrame) -> pd.DataFrame:
    # Assert shape, types, and value ranges hold.
 
    # Expected columns exist
    expected_cols = {"year", "country", "type", "fatal_y_n"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"[7-Validate] Missing expected columns: {missing}")
 
    # Year range makes sense
    if "year" in df.columns:
        bad_years = df["year"].dropna()
        assert bad_years.between(1500, 2025).all(), \
            "[7-Validate] Year values out of valid range!"
 
    # Age range makes sense
    if "age" in df.columns:
        bad_age = df["age"].dropna()
        assert bad_age.between(1, 100).all(), \
            "[7-Validate] Age values out of valid range!"
 
    # Fatal only Y or N (or NaN)
    if "fatal_y_n" in df.columns:
        valid_fatal = df["fatal_y_n"].dropna().isin(["Y", "N"])
        assert valid_fatal.all(), \
            "[7-Validate] Unexpected values in fatal_y_n!"
 
    # No complete duplicate rows
    assert df.duplicated().sum() == 0, \
        "[7-Validate] Duplicate rows remain!"
 
    # Row count sanity check (should have kept most data)
    assert len(df) > 1000, \
        f"[7-Validate] Too few rows after cleaning: {len(df)}"
 
    log.info(f"[7-Validate] All checks passed — {len(df):,} rows, {df.shape[1]} cols")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# Main Pipeline — .pipe() chain
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(filepath: str, output_path: str = None) -> pd.DataFrame:
    
    original_len = len(pd.read_csv(filepath, encoding="latin-1"))
 
    df = (
        load(filepath)
        .pipe(fix_shape)
        .pipe(fix_column_names)
        .pipe(fix_types)
        .pipe(fix_nulls)
        .pipe(fix_invalid)
        .pipe(validate)
    )
 
    log.info(f"Pipeline complete: {original_len:,} → {len(df):,} rows "
             f"({original_len - len(df):,} removed)")
 
    if output_path:
        df.to_csv(output_path, index=False)
        log.info(f"Saved cleaned data → {output_path}")
 
    return df
 