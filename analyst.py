"""
analyst.py — Personal AI Data Analyst
LLM integration (Ollama), prompt-to-code pipeline, and safe code execution.
"""

import re
import sys
import time
import json
import textwrap
import traceback
import contextlib
import threading
from io import StringIO
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests

# ─── OLLAMA CONFIG ────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"
LLM_TIMEOUT     = 120     # seconds
CODE_TIMEOUT    = 30      # seconds for user code execution

# ─── SAFE EXECUTION SANDBOX ──────────────────────────────
# Whitelist of modules allowed inside user-generated code
ALLOWED_MODULES = {
    "pandas", "pd", "numpy", "np", "math", "statistics",
    "datetime", "re", "json", "collections", "itertools",
    "functools", "string", "random", "scipy", "sklearn",
    "statsmodels", "plotly", "px", "go", "matplotlib",
    "seaborn", "io", "copy", "operator", "typing",
}

# Dangerous builtins/keywords to block
BLOCKED_PATTERNS = [
    r"\bos\b", r"\bsubprocess\b", r"\beval\b", r"\bexec\b",
    r"\bopen\b", r"\bcompile\b",
    r"\bgetattr\b.*__", r"\bsetattr\b", r"\bdelattr\b",
    r"shutil", r"socket", r"urllib", r"http\.client",
    r"ftplib", r"smtplib", r"pickle", r"marshal",
    r"importlib", r"builtins\.__", r"sys\.exit",
    r"sys\.modules", r"globals\(\)", r"locals\(\)",
    r"__builtins__", r"ctypes", r"cffi",
]


# ─── LLM AVAILABILITY ─────────────────────────────────────

def check_ollama_status() -> Tuple[bool, str]:
    """
    Check if Ollama is running and the model is available.
    Returns (is_available: bool, message: str)
    """
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            if any(DEFAULT_MODEL in m for m in models):
                return True, f"✅ Ollama running | Model: {DEFAULT_MODEL} ready"
            else:
                avail = ", ".join(models) if models else "none"
                return False, f"⚠️ Ollama running but '{DEFAULT_MODEL}' not found. Available: {avail}"
        return False, "❌ Ollama server not responding correctly"
    except requests.exceptions.ConnectionError:
        return False, "❌ Ollama not running. Start with: ollama serve"
    except Exception as e:
        return False, f"❌ Error checking Ollama: {str(e)}"


def get_available_models() -> list:
    """Return list of locally available Ollama model names."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        pass
    return []


# ─── ASK LLM ──────────────────────────────────────────────

def ask_llm(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system_prompt: str = "",
    temperature: float = 0.1,
    stream: bool = False,
) -> Tuple[bool, str]:
    """
    Send a prompt to Ollama and return (success, response_text).
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "num_predict": 180,
            "top_p": 0.9,
        },
    }
    if system_prompt:
        payload["system"] = system_prompt

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=LLM_TIMEOUT,
        )
        if resp.status_code == 200:
            if not stream:
                data = resp.json()
                return True, data.get("response", "").strip()
            else:
                # Streaming: concatenate chunks
                full = []
                for line in resp.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        full.append(chunk.get("response", ""))
                        if chunk.get("done"):
                            break
                return True, "".join(full).strip()
        else:
            return False, f"LLM API error: HTTP {resp.status_code}"
    except requests.exceptions.Timeout:
        return False, "LLM request timed out. Try a shorter prompt or simpler question."
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to Ollama. Ensure `ollama serve` is running."
    except Exception as e:
        return False, f"LLM error: {str(e)}"


# ─── PROMPT → CODE ────────────────────────────────────────
def build_system_prompt(df: pd.DataFrame) -> str:
    """
    Create a system prompt giving the LLM full context about the dataset.
    """

    n_rows, n_cols = df.shape

    cols_info = []

    for col in df.columns:
        dtype = str(df[col].dtype)
        sample = df[col].dropna().head(1).tolist()
        sample_str = ", ".join([str(s) for s in sample])

        cols_info.append(
            f"  - {col} ({dtype}): e.g. {sample_str}"
        )

    cols_block = "\n".join(cols_info)

    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    return f"""
You are an expert Python data analyst AI assistant.

The user has uploaded a dataset.

Your ONLY job is to write Python code to answer their question.

DATASET INFO:
- Shape: {n_rows} rows × {n_cols} columns

- Columns:
{cols_block}

- Numeric columns:
{numeric_cols}

RULES:

1. Return ONLY executable Python code.

2. The dataframe already exists as:
   df

NEVER create, reload, overwrite, or redefine df.

NEVER write:
df = ...
pd.read_csv(...)

3. Use print() ONLY for text outputs.

NEVER use:
print(fig)
fig.show()
plt.show()

4. For charts:

* use Plotly Express (px) whenever possible
* store chart in variable:
  fig

5. Always handle missing values safely.

Use:
df = df.ffill()
or
df = df.bfill()
or
dropna()

NEVER use:
fillna(method='ffill')
fillna(method='bfill')

6. For multiple columns ALWAYS use:
   df[['col1', 'col2']]

NEVER use:
df['col1', 'col2']

7. For groupby multiple columns ALWAYS use:
   df.groupby('column')[['col1', 'col2']]

NEVER use:
df.groupby('column')['col1', 'col2']

8. Always generate COMPLETE executable code.

NEVER generate:

* partial code
* pseudo code
* placeholder code
* comments-only code

9. Always close:

* brackets
* quotes
* f-strings
* parentheses

10. ALWAYS use:
    template='plotly_dark'

11. NEVER use:
    os
    subprocess
    open()
    eval()
    exec()
    network access
    file access

12. For correlation analysis:

* If both columns are numeric:

  * calculate correlation coefficient
  * optionally create scatter plot

* If one column is binary/categorical:

  * use:

    * boxplot
    * violin plot
    * bar chart

NEVER create meaningless line charts for binary target variables.

13. ALWAYS prefer SIMPLE and MINIMAL pandas code.

Prefer:

* groupby()
* mean()
* sum()
* value_counts()
* reset_index()

Avoid:

* complex pivot tables
* unnecessary reshaping
* unnecessary dummy variables
* overly complex transformations

14. NEVER invent variable names.

Only use variables explicitly created earlier in the code.

BAD:
y=survival0

GOOD:
y=survival_counts[0]

15. For visualizations:
    Prefer:

* px.bar()
* px.histogram()
* px.box()
* px.violin()
* px.scatter()

Avoid complex graph_objects unless necessary.

16. NEVER use unsupported or undefined variables.

17. ALWAYS produce production-ready Python code that executes successfully on the first run.

18. Before using any variable:
- ensure it was created earlier in the code

NEVER invent variable names.

BAD:
print(duplicate0)

GOOD:
print(duplicate_rows)

19. Keep solutions SIMPLE.

For duplicate rows:
Use:
duplicate_rows = df[df.duplicated()]

Do NOT create unnecessary extra variables.

20. NEVER remove columns that are needed later for grouping, plotting, or visualization.

BAD:
df[['Age', 'Fare']]

then using:
x='Sex'

GOOD:
df[['Sex', 'Age', 'Fare']]

EXAMPLE:

import plotly.express as px

summary = df.groupby('Region')[['Sales']].sum().reset_index()

fig = px.bar(
summary,
x='Region',
y='Sales',
title='Sales by Region'
)

fig.update_layout(
template='plotly_dark'
)

Output ONLY Python code.
"""

def prompt_to_code(
    user_prompt: str,
    df: pd.DataFrame,
    model: str = DEFAULT_MODEL,
    conversation_history: list = None,
) -> Tuple[bool, str]:
    """
    Convert a natural-language question about the dataframe into Python code.
    Returns (success: bool, code_or_error: str)
    """
    system_prompt = build_system_prompt(df)

    # Build conversation context if history provided
    history_context = ""
    if conversation_history:
        recent = conversation_history[-2:]  # last 2 exchanges only
        history_lines = []
        for entry in recent:
            role = entry.get("role", "user")
            msg  = entry.get("content", "")
            history_lines.append(f"[{role.upper()}]: {msg}")
        if history_lines:
            history_context = "Previous conversation:\n" + "\n".join(history_lines) + "\n\n"

    full_prompt = f"{history_context}User question: {user_prompt}"

    success, response = ask_llm(
        prompt=full_prompt,
        model=model,
        system_prompt=system_prompt,
        temperature=0.05,
    )

    if not success:
        return False, response

    code = extract_code(response)
    if not code.strip():
        # LLM might have returned code without fences
        code = response.strip()

    return True, code


# ─── CODE EXTRACTION ──────────────────────────────────────

def extract_code(text: str) -> str:
    """
    Extract Python code from LLM response.
    Handles: ```python ... ```, ``` ... ```, raw code.
    """
    # Try fenced python block
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try any fenced block
    match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Return raw (strip leading prose if code-like lines found)
    lines = text.strip().splitlines()
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("import ", "from ", "df", "fig", "print", "#",
                                 "result", "data", "plt", "px", "go", "pd", "np")):
            in_code = True
        if in_code:
            code_lines.append(line)
    return "\n".join(code_lines) if code_lines else text.strip()


# ─── SECURITY CHECK ───────────────────────────────────────

def check_code_safety(code: str) -> Tuple[bool, str]:
    """
    Scan code for dangerous patterns.
    Returns (is_safe: bool, reason_if_not: str)
    """
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, code):
            return False, f"Blocked pattern detected: `{pattern}`"

    # Check for suspicious import statements
    import_matches = re.findall(r"(?:import|from)\s+([\w\.]+)", code)
    for mod in import_matches:
        base_mod = mod.split(".")[0]
        if base_mod not in ALLOWED_MODULES and base_mod not in (
            "plotly", "sklearn", "scipy", "statsmodels",
            "matplotlib", "seaborn", "datetime", "collections",
        ):
            # Don't block, just warn — some valid modules might not be listed
            pass  # Could be strict: return False, f"Unauthorized module: {base_mod}"

    return True, ""


# ─── SAFE CODE RUNNER ─────────────────────────────────────

class TimeoutError(Exception):
    pass


def run_with_timeout(func, args=(), kwargs={}, timeout=CODE_TIMEOUT):
    """Run func in a thread with timeout. Returns result or raises TimeoutError."""
    result = [None]
    error  = [None]

    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)

    if t.is_alive():
        raise TimeoutError(f"Code execution timed out after {timeout}s")
    if error[0]:
        raise error[0]
    return result[0]


def run_code(
    code: str,
    df: pd.DataFrame,
    timeout: int = CODE_TIMEOUT,
) -> dict:
    """
    Safely execute Python code with df in scope.
    Returns a result dict with:
      - success: bool
      - output:  captured stdout text
      - figure:  Plotly fig object or None
      - result_df: DataFrame if code produced one, else None
      - error:   error message or None
      - exec_time: seconds
    """
    result = {
        "success":   False,
        "output":    "",
        "figure":    None,
        "result_df": None,
        "error":     None,
        "exec_time": 0.0,
    }

    # Security check
    is_safe, reason = check_code_safety(code)
    if not is_safe:
        result["error"] = f"⚠️ Security violation: {reason}"
        return result

    # Prepare execution namespace
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    import scipy.stats as stats

    namespace = {
"__builtins__": {
    "print": print,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "max": max,
    "min": min,
    "sum": sum,
    "abs": abs,
    "round": round,
    "type": type,
    "isinstance": isinstance,
    "hasattr": hasattr,
    "getattr": getattr,
    "setattr": setattr,
    "vars": vars,

    "__import__": __import__,
    "__build_class__": __build_class__,
    "object": object,

    "True": True,
    "False": False,
    "None": None,

    "ValueError": ValueError,
    "TypeError": TypeError,
    "Exception": Exception,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "ZeroDivisionError": ZeroDivisionError,
    "NotImplementedError": NotImplementedError,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
},
        "df":  df.copy(),
        "pd":  pd,
        "np":  np,
        "px":  px,
        "go":  go,
        "ff":  ff,
        "stats": stats,
    }

    # Try to import optional libraries
    try:
        import matplotlib.pyplot as plt
        namespace["plt"] = plt
        import seaborn as sns
        namespace["sns"] = sns
    except ImportError:
        pass

    try:
        from sklearn import preprocessing, linear_model, ensemble, cluster, metrics
        namespace["preprocessing"] = preprocessing
        namespace["linear_model"]  = linear_model
        namespace["ensemble"]      = ensemble
        namespace["cluster"]       = cluster
        namespace["metrics"]       = metrics
    except ImportError:
        pass

    try:
        import math, statistics, re, json, datetime, collections, string, random
        namespace.update({
            "math": math, "statistics": statistics, "re": re,
            "json": json, "datetime": datetime, "collections": collections,
            "string": string, "random": random,
        })
    except ImportError:
        pass

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()

    start_time = time.time()

    def _exec():

        # Auto-fix common LLM mistakes

        code_fixed = code

        # Remove unsupported figure rendering
        code_fixed = code_fixed.replace("fig.show()", "")
        code_fixed = code_fixed.replace("plt.show()", "")

        # Fix hallucinated duplicate variable
        code_fixed = code_fixed.replace("duplicate0", "duplicate_rows")
        
        # Fix wrong Plotly histogram function
        code_fixed = code_fixed.replace("px.hist(", "px.histogram(")

        # Execute cleaned code
        exec(compile(code_fixed, "<analyst_code>", "exec"), namespace)

    try:
        run_with_timeout(_exec, timeout=timeout)
        result["success"]   = True
        # Safely get figure if created
        result["figure"] = namespace.get("fig", None)
        # Safely get dataframe result
        result["result_df"] = (
            namespace.get("result")
            if isinstance(namespace.get("result"), pd.DataFrame)
            else None
        )

    except TimeoutError as e:
        result["error"] = f"⏱️ Timeout: {str(e)}"

    except SyntaxError as e:
        result["error"] = f"🔴 Syntax Error at line {e.lineno}: {e.msg}\n{e.text}"

    except NameError as e:
        result["error"] = f"🔴 Name Error: {str(e)}"

    except KeyError as e:
        result["error"] = f"🔴 Column Not Found: {str(e)}"

    except Exception as e:
        tb = traceback.format_exc()
        # Only show last few lines of traceback
        tb_lines = [l for l in tb.splitlines() if "analyst_code" in l or type(e).__name__ in l]
        result["error"] = f"🔴 {type(e).__name__}: {str(e)}"
        if tb_lines:
            result["error"] += "\n" + "\n".join(tb_lines[-3:])

    finally:
        sys.stdout = old_stdout
        result["exec_time"] = round(time.time() - start_time, 3)
        result["output"]    = captured.getvalue()

    return result


# ─── AI INSIGHTS GENERATION ──────────────────────────────

def generate_ai_insights(df: pd.DataFrame, model: str = DEFAULT_MODEL) -> Tuple[bool, list]:
    """
    Use LLM to generate natural-language insights about the dataset.
    Returns (success, list_of_insight_strings)
    """
    from utils import profile_dataset, find_high_correlations

    profile = profile_dataset(df)
    high_corr = find_high_correlations(df)

    # Build a compact data summary
    summary_lines = [
        f"Dataset: {profile['n_rows']} rows × {profile['n_cols']} columns",
        f"Numeric columns: {profile['numeric_cols']}",
        f"Categorical columns: {profile['cat_cols']}",
        f"Missing values: {profile['missing_total']} ({profile['missing_pct']}%)",
        f"Duplicate rows: {profile['duplicates']}",
    ]

    if profile["num_stats"]:
        summary_lines.append("Key statistics:")
        for col, stats in list(profile["num_stats"].items())[:4]:
            summary_lines.append(
                f"  {col}: mean={stats['mean']}, std={stats['std']}, "
                f"outliers={stats['outliers']}, skew={stats['skew']}"
            )

    if high_corr:
        corr_str = "; ".join([f"{a}↔{b}={v}" for a, b, v in high_corr[:3]])
        summary_lines.append(f"High correlations: {corr_str}")

    summary = "\n".join(summary_lines)

    prompt = f"""Analyze this dataset summary and provide exactly 8 concise, actionable insights.
Each insight should be a single sentence starting with an emoji.
Focus on: data quality, distributions, outliers, correlations, business implications.
Do NOT include headers, numbers, or explanations — just 8 insight sentences, one per line.

{summary}"""

    success, response = ask_llm(
        prompt=prompt,
        model=model,
        temperature=0.3,
    )

    if not success:
        return False, []

    # Parse response into list
    lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
    insights = [l for l in lines if len(l) > 10][:10]
    return True, insights


def generate_rule_based_insights(df: pd.DataFrame) -> list:
    """
    Generate data insights without LLM using pure statistical rules.
    Always available as fallback.
    """
    from utils import profile_dataset, find_high_correlations, detect_outliers_iqr

    profile   = profile_dataset(df)
    high_corr = find_high_correlations(df)
    insights  = []

    # Missing values
    if profile["missing_pct"] == 0:
        insights.append("✅ Dataset is complete — no missing values detected.")
    elif profile["missing_pct"] < 5:
        insights.append(f"⚠️ Low missing data ({profile['missing_pct']}%). Safe to impute.")
    else:
        worst = max(profile["col_missing_pct"], key=profile["col_missing_pct"].get)
        pct   = profile["col_missing_pct"][worst]
        insights.append(f"🔴 High missing data: '{worst}' has {pct}% missing values.")

    # Duplicates
    if profile["duplicates"] > 0:
        insights.append(f"♻️ {profile['duplicates']} duplicate rows found — consider deduplication.")
    else:
        insights.append("✅ No duplicate rows detected.")

    # Outliers
    for col, stats in list(profile["num_stats"].items())[:3]:
        if stats["outliers"] > 0:
            insights.append(
                f"📊 '{col}' has {stats['outliers']} IQR outliers "
                f"(skew={stats['skew']})."
            )

    # Skewness
    for col, stats in profile["num_stats"].items():
        if abs(stats["skew"]) > 1.5:
            direction = "right" if stats["skew"] > 0 else "left"
            insights.append(f"📈 '{col}' is strongly {direction}-skewed (skew={stats['skew']}).")
            break

    # Correlations
    if high_corr:
        a, b, v = high_corr[0]
        insights.append(f"🔗 Strong correlation between '{a}' and '{b}' (r={v}).")

    # Dataset size
    if profile["n_rows"] > 10_000:
        insights.append(f"🚀 Large dataset: {profile['n_rows']:,} rows — ML models will be reliable.")
    elif profile["n_rows"] < 100:
        insights.append(f"⚠️ Small dataset: only {profile['n_rows']} rows — ML results may vary.")

    # Memory
    if profile["memory_mb"] > 100:
        insights.append(f"💾 Large in-memory size: {profile['memory_mb']} MB — consider chunking.")

    # Column types
    if len(profile["numeric_cols"]) > len(profile["cat_cols"]):
        insights.append("📐 Mostly numeric data — great for regression and clustering models.")
    elif profile["cat_cols"]:
        insights.append(f"🏷️ {len(profile['cat_cols'])} categorical columns — suitable for classification tasks.")

    return insights[:10]
