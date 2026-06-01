"""
utils.py — Personal AI Data Analyst
Utility functions: data loading, profiling, anomaly detection,
export helpers, session state, and shared constants.
"""

import io
import os
import json
import base64
import hashlib
import warnings
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── CONSTANTS ────────────────────────────────────────────
SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".xls", ".json"]
MAX_UPLOAD_SIZE_MB    = 200
DATE_FORMAT           = "%Y-%m-%d %H:%M:%S"
NUMERIC_THRESHOLD     = 0.85   # fraction of column values that must be numeric
CATEGORICAL_THRESHOLD = 30     # max unique values for a column to be "categorical"

# Plotly dark theme config
PLOTLY_THEME = "plotly_dark"
PLOTLY_PAPER = "#0A1628"
PLOTLY_PLOT  = "#0F1F35"
PLOTLY_FONT  = "#E8F4F8"
PLOTLY_CYAN  = "#00F5FF"
PLOTLY_VIOLET= "#7B61FF"
PLOTLY_GREEN = "#00FFA3"
PLOTLY_PALETTE = [
    "#00F5FF", "#7B61FF", "#00FFA3", "#FF6B35",
    "#FF2D78", "#FFD700", "#A78BFA", "#34D399",
    "#F472B6", "#60A5FA", "#FBBF24", "#E879F9"
]


# ─── DATA LOADING ─────────────────────────────────────────

def load_data(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load CSV, Excel, or JSON file into a DataFrame.
    Auto-detects file type from extension.
    Returns None on failure.
    """
    if uploaded_file is None:
        return None

    fname = uploaded_file.name.lower()
    try:
        if fname.endswith(".csv"):
            # Try common encodings
            for enc in ["utf-8", "latin-1", "cp1252"]:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=enc, low_memory=False)
                    break
                except UnicodeDecodeError:
                    continue

        elif fname.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)

        elif fname.endswith(".json"):
            uploaded_file.seek(0)
            content = uploaded_file.read()
            data = json.loads(content)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame.from_dict(data, orient="index").reset_index()
            else:
                return None
        else:
            return None

        # Basic cleaning after load
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # remove unnamed cols
        df.columns = df.columns.astype(str).str.strip()
        return df

    except Exception as e:
        print(f"[load_data error] {e}")
        return None


# ─── COLUMN TYPE INFERENCE ────────────────────────────────

def infer_column_types(df: pd.DataFrame) -> dict:
    """
    Returns a dict classifying each column:
    'numeric', 'categorical', 'datetime', 'text', 'boolean'
    """
    types = {}
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            types[col] = "empty"
            continue

        dtype = df[col].dtype

        if pd.api.types.is_bool_dtype(dtype):
            types[col] = "boolean"
        elif pd.api.types.is_numeric_dtype(dtype):
            types[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            types[col] = "datetime"
        else:
            # Try datetime parse
            try:
                pd.to_datetime(series.head(50), infer_datetime_format=True)
                types[col] = "datetime"
                continue
            except Exception:
                pass

            n_unique = series.nunique()
            n_total  = len(series)
            if n_unique <= 2:
                types[col] = "boolean"
            elif n_unique <= CATEGORICAL_THRESHOLD or (n_unique / n_total) < 0.5:
                types[col] = "categorical"
            else:
                types[col] = "text"

    return types


def get_numeric_cols(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_cols(df: pd.DataFrame) -> list:
    col_types = infer_column_types(df)
    return [c for c, t in col_types.items() if t in ("categorical", "boolean")]


def get_datetime_cols(df: pd.DataFrame) -> list:
    col_types = infer_column_types(df)
    return [c for c, t in col_types.items() if t == "datetime"]


# ─── DATASET PROFILING ────────────────────────────────────

def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Returns a comprehensive summary dictionary of the dataset.
    """
    n_rows, n_cols = df.shape
    numeric_cols   = get_numeric_cols(df)
    cat_cols       = get_categorical_cols(df)
    dt_cols        = get_datetime_cols(df)

    missing_total  = int(df.isnull().sum().sum())
    missing_pct    = round(missing_total / (n_rows * n_cols) * 100, 2) if n_rows * n_cols > 0 else 0
    duplicates     = int(df.duplicated().sum())
    memory_mb      = round(df.memory_usage(deep=True).sum() / 1_048_576, 3)

    col_missing    = df.isnull().sum()
    col_missing_pct= (col_missing / n_rows * 100).round(2)

    # Per numeric column stats
    num_stats = {}
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr     = q3 - q1
        outliers= int(((s < (q1 - 1.5 * iqr)) | (s > (q3 + 1.5 * iqr))).sum())
        num_stats[col] = {
            "mean":    round(float(s.mean()), 4),
            "median":  round(float(s.median()), 4),
            "std":     round(float(s.std()), 4),
            "min":     round(float(s.min()), 4),
            "max":     round(float(s.max()), 4),
            "q1":      round(float(q1), 4),
            "q3":      round(float(q3), 4),
            "skew":    round(float(s.skew()), 4),
            "outliers": outliers,
        }

    # Correlation matrix (numeric only)
    corr_matrix = None
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().round(3)

    return {
        "n_rows":       n_rows,
        "n_cols":       n_cols,
        "numeric_cols": numeric_cols,
        "cat_cols":     cat_cols,
        "dt_cols":      dt_cols,
        "missing_total": missing_total,
        "missing_pct":  missing_pct,
        "duplicates":   duplicates,
        "memory_mb":    memory_mb,
        "col_missing":  col_missing.to_dict(),
        "col_missing_pct": col_missing_pct.to_dict(),
        "num_stats":    num_stats,
        "corr_matrix":  corr_matrix,
        "dtypes":       {c: str(t) for c, t in df.dtypes.items()},
    }


# ─── ANOMALY / OUTLIER DETECTION ──────────────────────────

def detect_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """
    Returns a DataFrame with z-scores and an 'is_outlier' column for each numeric column.
    """
    numeric_cols = get_numeric_cols(df)
    result = pd.DataFrame(index=df.index)
    for col in numeric_cols:
        s = df[col].fillna(df[col].median())
        z = (s - s.mean()) / (s.std() + 1e-9)
        result[f"{col}_zscore"] = z.round(3)
    result["outlier_score"] = result.abs().max(axis=1)
    result["is_outlier"]    = result["outlier_score"] > threshold
    return result


def detect_outliers_iqr(series: pd.Series) -> pd.Series:
    """Boolean mask: True if value is an IQR outlier."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return (series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))


# ─── DATA CLEANING ────────────────────────────────────────

def clean_dataframe(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """
    Fill missing values using a chosen strategy.
    strategy: 'median', 'mean', 'mode', 'ffill', 'bfill', 'drop'
    """
    df = df.copy()
    numeric_cols = get_numeric_cols(df)
    cat_cols     = get_categorical_cols(df)

    if strategy == "drop":
        return df.dropna()

    for col in numeric_cols:
        if df[col].isnull().sum() == 0:
            continue
        if strategy == "median":
            df[col].fillna(df[col].median(), inplace=True)
        elif strategy == "mean":
            df[col].fillna(df[col].mean(), inplace=True)
        elif strategy in ("ffill", "bfill"):
            df[col].fillna(method=strategy, inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    for col in cat_cols:
        if df[col].isnull().sum() == 0:
            continue
        if strategy == "mode":
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)
        else:
            df[col].fillna("Unknown", inplace=True)

    return df


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Remove duplicate rows. Returns (cleaned_df, n_removed)."""
    original_len = len(df)
    df = df.drop_duplicates()
    return df, original_len - len(df)


# ─── EXPORT HELPERS ───────────────────────────────────────

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to UTF-8 CSV bytes for download."""
    return df.to_csv(index=False).encode("utf-8")


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to Excel bytes for download."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buf.getvalue()


def fig_to_png_bytes(fig) -> bytes:
    """Convert a Plotly figure to PNG bytes."""
    try:
        return fig.to_image(format="png", scale=2)
    except Exception:
        return b""


def df_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Convert DataFrame head to markdown string."""
    return df.head(max_rows).to_markdown(index=False)


# ─── REPORT GENERATION ────────────────────────────────────

def generate_text_report(df: pd.DataFrame, profile: dict, insights: list) -> str:
    """
    Generate a plain-text analytics report.
    """
    now  = datetime.now().strftime(DATE_FORMAT)
    lines = [
        "=" * 60,
        "        Personal AI Data Analyst — Analytics Report",
        f"        Generated: {now}",
        "=" * 60,
        "",
        "DATASET OVERVIEW",
        "-" * 40,
        f"  Rows:            {profile['n_rows']:,}",
        f"  Columns:         {profile['n_cols']}",
        f"  Numeric Cols:    {len(profile['numeric_cols'])}",
        f"  Categorical Cols:{len(profile['cat_cols'])}",
        f"  Missing Values:  {profile['missing_total']:,} ({profile['missing_pct']}%)",
        f"  Duplicate Rows:  {profile['duplicates']}",
        f"  Memory Usage:    {profile['memory_mb']} MB",
        "",
        "NUMERIC COLUMN STATISTICS",
        "-" * 40,
    ]

    for col, stats in profile["num_stats"].items():
        lines.append(f"\n  [{col}]")
        for k, v in stats.items():
            lines.append(f"    {k:12}: {v}")

    lines += ["", "AI-GENERATED INSIGHTS", "-" * 40]
    for i, insight in enumerate(insights, 1):
        lines.append(f"  {i}. {insight}")

    lines += ["", "=" * 60, "  End of Report — Personal AI Data Analyst", "=" * 60]
    return "\n".join(lines)


# ─── SESSION HELPERS ──────────────────────────────────────

def get_file_hash(uploaded_file) -> str:
    """Return a short MD5 hash of the file content."""
    uploaded_file.seek(0)
    content = uploaded_file.read()
    uploaded_file.seek(0)
    return hashlib.md5(content).hexdigest()[:10]


def format_number(n) -> str:
    """Format large numbers with commas."""
    try:
        if float(n) >= 1_000_000:
            return f"{n/1_000_000:.2f}M"
        elif float(n) >= 1_000:
            return f"{n:,.0f}"
        else:
            return f"{n:.4g}"
    except Exception:
        return str(n)


def safe_percentage(part: float, total: float) -> str:
    """Return 'XX.X%' or '0.0%' safely."""
    if total == 0:
        return "0.0%"
    return f"{(part / total * 100):.1f}%"


def truncate_text(text: str, max_len: int = 120) -> str:
    return text if len(text) <= max_len else text[:max_len] + "…"


# ─── SMART CHART RECOMMENDATION ───────────────────────────

def recommend_charts(df: pd.DataFrame) -> list:
    """
    Return a list of (chart_type, description, reason) tuples
    based on data characteristics.
    """
    numeric_cols = get_numeric_cols(df)
    cat_cols     = get_categorical_cols(df)
    dt_cols      = get_datetime_cols(df)
    recs = []

    if len(numeric_cols) >= 1:
        recs.append(("histogram",    numeric_cols[0], "Understand distribution of numeric data"))
        recs.append(("box_plot",     numeric_cols[0], "Detect outliers and spread"))

    if len(numeric_cols) >= 2:
        recs.append(("scatter",  f"{numeric_cols[0]} vs {numeric_cols[1]}", "Explore correlation"))
        recs.append(("heatmap",  "all numeric cols", "Visualize correlation matrix"))

    if len(cat_cols) >= 1 and len(numeric_cols) >= 1:
        recs.append(("bar_chart", f"{cat_cols[0]} by {numeric_cols[0]}", "Compare groups"))
        recs.append(("pie_chart", cat_cols[0], "See distribution of categories"))

    if len(dt_cols) >= 1 and len(numeric_cols) >= 1:
        recs.append(("time_series", f"{dt_cols[0]} vs {numeric_cols[0]}", "Spot trends over time"))

    if len(numeric_cols) >= 2 and len(cat_cols) >= 1:
        recs.append(("violin", numeric_cols[0], "Distribution by category with density"))

    return recs


# ─── CORRELATION FINDER ───────────────────────────────────

def find_high_correlations(df: pd.DataFrame, threshold: float = 0.7) -> list:
    """
    Return pairs of highly correlated columns.
    """
    numeric_cols = get_numeric_cols(df)
    if len(numeric_cols) < 2:
        return []

    corr = df[numeric_cols].corr().abs()
    pairs = []
    cols  = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if val >= threshold:
                pairs.append((cols[i], cols[j], round(float(val), 3)))

    return sorted(pairs, key=lambda x: -x[2])


# ─── SUGGESTED PROMPTS ────────────────────────────────────

def suggest_prompts(df: pd.DataFrame) -> list:
    """
    Dynamically generate context-aware prompt suggestions based on column names.
    """
    numeric_cols = get_numeric_cols(df)
    cat_cols     = get_categorical_cols(df)
    dt_cols      = get_datetime_cols(df)

    prompts = [
        "Show me a summary of all columns",
        "What are the top 10 rows by highest values?",
        "Count missing values in each column",
    ]

    if numeric_cols:
        col = numeric_cols[0]
        prompts += [
            f"Show distribution of {col}",
            f"Find outliers in {col}",
            f"What is the average {col}?",
        ]

    if len(numeric_cols) >= 2:
        prompts.append(f"Correlation between {numeric_cols[0]} and {numeric_cols[1]}")

    if cat_cols:
        col = cat_cols[0]
        prompts += [
            f"Group data by {col} and show average values",
            f"Show count of each {col}",
            f"Which {col} has the highest total {numeric_cols[0] if numeric_cols else 'count'}?",
        ]

    if dt_cols:
        col = dt_cols[0]
        prompts.append(f"Show trend of {numeric_cols[0] if numeric_cols else 'count'} over {col}")

    prompts += [
        "Show me duplicate rows",
        "Describe the dataset statistics",
        "Find the top 5 rows",
        "Show value counts for all categorical columns",
    ]

    return prompts[:20]  # Return up to 20 prompts
