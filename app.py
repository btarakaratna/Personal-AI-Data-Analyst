"""
app.py — Personal AI Data Analyst
Main Streamlit application entry point.
Run with: streamlit run app.py
"""

import os
import sys
import re
import time
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ─── PAGE CONFIG (must be first Streamlit call) ───────────
st.set_page_config(
    page_title      = "Personal AI Data Analyst",
    page_icon       = "🧠",
    layout          = "wide",
    initial_sidebar_state = "expanded",
    menu_items      = {
        "Get help": "https://github.com/",
        "Report a bug": "https://github.com/",
        "About": "Personal AI Data Analyst — AI-Powered Data Analytics and Analytics Platform",
    },
)

# ─── INJECT CSS ───────────────────────────────────────────
CSS_PATH = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─── LOCAL IMPORTS ────────────────────────────────────────
from utils import (
    load_data, profile_dataset, suggest_prompts, df_to_csv_bytes,
    generate_text_report, get_numeric_cols, get_categorical_cols,
    get_datetime_cols, detect_outliers_zscore, clean_dataframe,
    remove_duplicates, recommend_charts, find_high_correlations,
    format_number, safe_percentage,
)
from analyst import (
    check_ollama_status, get_available_models, prompt_to_code,
    run_code, generate_rule_based_insights, generate_ai_insights,
    extract_code, DEFAULT_MODEL,
)
from ml_engine import (
    run_classification, run_regression, run_clustering,
    detect_task_type, CLASSIFIERS, REGRESSORS, CLUSTERERS,
)
from ui_components import (
    render_hero, render_dataset_metrics, render_data_preview,
    render_column_profiler, render_visualizations, render_missing_heatmap,
    render_outlier_analysis, render_prompt_chips, render_chat_bubble,
    render_insights, render_export_buttons, render_sidebar_brand,
    render_ml_metrics, section_header, neon_divider, status_indicator,
    metric_card, DARK_LAYOUT,
)

# ─── SESSION STATE INIT ───────────────────────────────────
def init_session():
    defaults = {
        "df":                None,
        "df_original":       None,
        "file_name":         None,
        "profile":           None,
        "chat_history":      [],
        "query_history":     [],
        "insights":          [],
        "llm_enabled":       True,
        "selected_model":    DEFAULT_MODEL,
        "last_fig":          None,
        "last_result_df":    None,
        "active_page":       "🏠 Dashboard",
        "ml_results":        None,
        "df_cleaned":        False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ─── SIDEBAR ──────────────────────────────────────────────
with st.sidebar:
    render_sidebar_brand()

    try:
        from streamlit_option_menu import option_menu
        selected_page = option_menu(
            menu_title    = None,
            options       = [
                "🏠 Dashboard", "📁 Data Upload", "🤖 AI Analysis",
                "📊 Visualizations", "🧠 ML Studio", "💡 AI Insights",
                "📤 Export", "⚙️ Settings",
            ],
            icons         = [
                "house-fill", "cloud-upload-fill", "robot",
                "bar-chart-fill", "cpu-fill", "lightbulb-fill",
                "download", "gear-fill",
            ],
            default_index = 0,
            styles        = {
                "container":    {"background-color": "transparent", "padding": "0"},
                "icon":         {"color": "#00F5FF", "font-size": "14px"},
                "nav-link":     {
                    "font-size": "0.88rem", "text-align": "left",
                    "margin": "2px 0", "padding": "0.55rem 0.9rem",
                    "border-radius": "10px", "color": "#7FA8C8",
                    "--hover-color": "rgba(0,245,255,0.08)",
                },
                "nav-link-selected": {
                    "background": "linear-gradient(135deg,rgba(0,245,255,0.12),rgba(123,97,255,0.12))",
                    "color": "#00F5FF", "font-weight": "600",
                    "border": "1px solid rgba(0,245,255,0.25)",
                },
            },
        )
    except ImportError:
        selected_page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "📁 Data Upload", "🤖 AI Analysis",
             "📊 Visualizations", "🧠 ML Studio", "💡 AI Insights",
             "📤 Export", "⚙️ Settings"],
        )

    st.session_state["active_page"] = selected_page

    # ── Sidebar status panel
    st.markdown("<br>", unsafe_allow_html=True)
    neon_divider()

    # Dataset status
    if st.session_state["df"] is not None:
        df = st.session_state["df"]
        st.markdown(f"""
        <div style='background:rgba(0,255,163,0.06); border:1px solid rgba(0,255,163,0.2);
                    border-radius:10px; padding:0.8rem; margin-top:0.5rem;'>
            <div style='color:#00FFA3; font-size:0.8rem; font-weight:600; margin-bottom:0.4rem;'>
                📂 {st.session_state['file_name']}
            </div>
            <div style='color:#7FA8C8; font-size:0.72rem;'>
                {df.shape[0]:,} rows × {df.shape[1]} cols
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Ollama status
    ollama_ok, ollama_msg = check_ollama_status()
    st.markdown("<br>", unsafe_allow_html=True)
    status_indicator("Ollama LLM", ollama_ok, "llama3.2:3b" if ollama_ok else "offline")

    neon_divider()
    st.markdown(
        "<div style='color:#3D6B8C; font-size:0.65rem; text-align:center;'>"
        "Personal AI Data Analyst · v2.0<br>© 2026 Premium Edition</div>",
        unsafe_allow_html=True,
    )


# ─── HELPER: require dataset ──────────────────────────────
def require_dataset():
    if st.session_state["df"] is None:
        st.markdown("""
        <div style='text-align:center; padding:4rem 2rem;'>
            <div style='font-size:4rem;'>📁</div>
            <div style='font-size:1.4rem; color:#00F5FF; font-weight:700; margin:1rem 0 0.5rem;'>
                No Dataset Loaded
            </div>
            <div style='color:#7FA8C8; font-size:0.95rem;'>
                Go to <strong style='color:#7B61FF;'>📁 Data Upload</strong> to upload your CSV, Excel, or JSON file.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return False
    return True


# ─── FALLBACK CODE GENERATOR (no LLM) ────────────────────

def _generate_fallback_code(prompt, df):
    """Simple rule-based fallback when Ollama is unavailable."""
    p = prompt.lower()
    numeric_cols = get_numeric_cols(df)
    cat_cols = get_categorical_cols(df)

    if "summary" in p or "describe" in p or "overview" in p:
        return "print(df.describe())\nprint(df.shape)"
    if "missing" in p or "null" in p or "nan" in p:
        return "print(df.isnull().sum())\nprint('Total:', df.isnull().sum().sum())"
    if "duplicate" in p:
        return "print('Duplicates:', df.duplicated().sum())\nprint(df[df.duplicated()].head())"
    if "correlation" in p and len(numeric_cols) >= 2:
        cols = numeric_cols[:6]
        return (
            "import plotly.express as px\n"
            f"corr = df[{cols}].corr().round(3)\n"
            "fig = px.imshow(corr, text_auto=True, title='Correlation Matrix',\n"
            "                color_continuous_scale=['#FF2D78','#0F1F35','#00F5FF'],\n"
            "                template='plotly_dark')\n"
            "fig.update_layout(paper_bgcolor='#0A1628', plot_bgcolor='#0F1F35')"
        )
    if ("distribution" in p or "histogram" in p) and numeric_cols:
        col = numeric_cols[0]
        return (
            "import plotly.express as px\n"
            f"fig = px.histogram(df, x='{col}', title='Distribution of {col}',\n"
            "                   color_discrete_sequence=['#00F5FF'], template='plotly_dark',\n"
            "                   marginal='box')\n"
            "fig.update_layout(paper_bgcolor='#0A1628', plot_bgcolor='#0F1F35')"
        )
    if cat_cols and numeric_cols:
        cat, num = cat_cols[0], numeric_cols[0]
        return (
            "import plotly.express as px\n"
            f"grouped = df.groupby('{cat}')['{num}'].sum().reset_index().sort_values('{num}', ascending=False)\n"
            f"fig = px.bar(grouped, x='{cat}', y='{num}', title='{num} by {cat}',\n"
            f"             color='{num}', color_continuous_scale=['#7B61FF','#00F5FF'], template='plotly_dark')\n"
            "fig.update_layout(paper_bgcolor='#0A1628', plot_bgcolor='#0F1F35')"
        )
    return "print(df.head(10))\nprint(df.columns.tolist())"


# ══════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════
if selected_page == "🏠 Dashboard":
    render_hero()

    if st.session_state["df"] is None:
        # Welcome screen
        st.markdown("""
        <div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:1.2rem; margin:2rem 0;'>
            <div class='glass-container' style='text-align:center; padding:2rem;'>
                <div style='font-size:2.5rem; margin-bottom:0.75rem;'>📁</div>
                <div style='font-size:1rem; font-weight:600; color:#00F5FF; margin-bottom:0.4rem;'>Upload Data</div>
                <div style='font-size:0.82rem; color:#7FA8C8;'>CSV, Excel, JSON — up to 200MB</div>
            </div>
            <div class='glass-container' style='text-align:center; padding:2rem;'>
                <div style='font-size:2.5rem; margin-bottom:0.75rem;'>🤖</div>
                <div style='font-size:1rem; font-weight:600; color:#7B61FF; margin-bottom:0.4rem;'>AI Analysis</div>
                <div style='font-size:0.82rem; color:#7FA8C8;'>Ask questions in plain English</div>
            </div>
            <div class='glass-container' style='text-align:center; padding:2rem;'>
                <div style='font-size:2.5rem; margin-bottom:0.75rem;'>📊</div>
                <div style='font-size:1rem; font-weight:600; color:#00FFA3; margin-bottom:0.4rem;'>Visualize</div>
                <div style='font-size:0.82rem; color:#7FA8C8;'>10+ interactive chart types</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:1.2rem; margin-bottom:2rem;'>
            <div class='glass-container' style='text-align:center; padding:2rem;'>
                <div style='font-size:2.5rem; margin-bottom:0.75rem;'>🧠</div>
                <div style='font-size:1rem; font-weight:600; color:#FF6B35; margin-bottom:0.4rem;'>ML Studio</div>
                <div style='font-size:0.82rem; color:#7FA8C8;'>Regression, Classification, Clustering</div>
            </div>
            <div class='glass-container' style='text-align:center; padding:2rem;'>
                <div style='font-size:2.5rem; margin-bottom:0.75rem;'>💡</div>
                <div style='font-size:1rem; font-weight:600; color:#FFD700; margin-bottom:0.4rem;'>AI Insights</div>
                <div style='font-size:0.82rem; color:#7FA8C8;'>Auto-generated data intelligence</div>
            </div>
            <div class='glass-container' style='text-align:center; padding:2rem;'>
                <div style='font-size:2.5rem; margin-bottom:0.75rem;'>📤</div>
                <div style='font-size:1rem; font-weight:600; color:#A78BFA; margin-bottom:0.4rem;'>Export</div>
                <div style='font-size:0.82rem; color:#7FA8C8;'>CSV, JSON, reports & charts</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Load sample data shortcut
        sample_path = os.path.join(os.path.dirname(__file__), "sample_data", "sample.csv")
        if os.path.exists(sample_path):
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(
                    "<div style='text-align:center; color:#7FA8C8; font-size:0.9rem; margin-bottom:0.75rem;'>"
                    "🚀 Quick start with sample data</div>",
                    unsafe_allow_html=True
                )
                if st.button("⚡ Load Sample Business Dataset", use_container_width=True):
                    with st.spinner("Loading sample data..."):
                        df = pd.read_csv(sample_path)
                        st.session_state["df"]          = df
                        st.session_state["df_original"] = df.copy()
                        st.session_state["file_name"]   = "sample.csv"
                        st.session_state["profile"]     = profile_dataset(df)
                        st.session_state["insights"]    = generate_rule_based_insights(df)
                    st.success("✅ Sample dataset loaded! Explore the sidebar.")
                    st.rerun()
    else:
        df = st.session_state["df"]
        # Dashboard overview
        render_dataset_metrics(df)
        neon_divider()

        col1, col2 = st.columns([3, 2])

        with col1:
            render_data_preview(df)

        with col2:
            section_header("💡", "Quick Insights", "AUTO")
            if not st.session_state["insights"]:
                st.session_state["insights"] = generate_rule_based_insights(df)
            insights = st.session_state["insights"]
            for ins in insights[:6]:
                emoji = ins[:2] if len(ins) >= 2 else "ℹ️"
                color_map = {
                    "✅": "#00FFA3", "⚠️": "#FF6B35", "🔴": "#FF2D78",
                    "📊": "#00F5FF", "📈": "#00F5FF", "🔗": "#7B61FF",
                    "🚀": "#00FFA3", "💾": "#FFD700", "♻️": "#FF6B35",
                }
                c = color_map.get(emoji, "#7FA8C8")
                st.markdown(
                    f"<div style='padding:0.6rem 0.9rem; border-left:3px solid {c}; "
                    f"margin-bottom:0.5rem; border-radius:0 8px 8px 0; "
                    f"background:rgba(255,255,255,0.02); font-size:0.85rem; color:#C8DCE8;'>"
                    f"{ins}</div>",
                    unsafe_allow_html=True
                )

            neon_divider()

            # Quick chart recommendation
            recs = recommend_charts(df)
            if recs:
                section_header("🎯", "Recommended Charts", "SMART")
                for chart_type, desc, reason in recs[:3]:
                    st.markdown(
                        f"<div style='padding:0.5rem 0.8rem; margin-bottom:0.4rem; "
                        f"border:1px solid rgba(0,245,255,0.1); border-radius:8px; "
                        f"background:rgba(0,245,255,0.03);'>"
                        f"<span style='color:#00F5FF; font-size:0.8rem; font-weight:600;'>📈 {chart_type.title()}</span><br>"
                        f"<span style='color:#7FA8C8; font-size:0.75rem;'>{desc} — {reason}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )


# ══════════════════════════════════════════════════════════
# PAGE: DATA UPLOAD
# ══════════════════════════════════════════════════════════
elif selected_page == "📁 Data Upload":
    render_hero()
    section_header("📁", "Data Upload Center", "MULTI-FORMAT")

    st.markdown("""
    <div class='glass-container'>
        <div style='text-align:center; padding:0.5rem 0;'>
            <div style='font-size:0.9rem; color:#7FA8C8;'>
                Supported formats: <strong style='color:#00F5FF;'>CSV</strong> · 
                <strong style='color:#7B61FF;'>Excel (.xlsx/.xls)</strong> · 
                <strong style='color:#00FFA3;'>JSON</strong>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop your dataset here or click to browse",
        type=["csv", "xlsx", "xls", "json"],
        help="Maximum file size: 200MB",
    )

    if uploaded_file is not None:
        with st.spinner("📂 Uploading dataset... Please wait"):
            df = load_data(uploaded_file)

        if df is None:
            st.error("❌ Failed to load file. Please check format and try again.")
        else:
            st.session_state["df"]          = df
            st.session_state["df_original"] = df.copy()
            st.session_state["file_name"]   = uploaded_file.name
            st.session_state["profile"]     = profile_dataset(df)
            st.session_state["insights"]    = generate_rule_based_insights(df)
            st.session_state["ml_results"]  = None
            st.session_state["df_cleaned"]  = False
            st.toast("✅ Dataset loaded successfully!", icon="🎉")
            st.success(
                f"✅ **{uploaded_file.name}** loaded successfully — "
                f"{df.shape[0]:,} rows × {df.shape[1]} columns"
            )

    # Show data details if loaded
    if st.session_state["df"] is not None:
        df = st.session_state["df"]

        neon_divider()
        render_dataset_metrics(df)
        neon_divider()

        tab1, tab2, tab3, tab4 = st.tabs([
            "🗃️ Preview", "🔬 Column Types", "🕳️ Missing Values", "🧹 Data Cleaning"
        ])

        with tab1:
            render_data_preview(df)

        with tab2:
            render_column_profiler(df)

        with tab3:
            render_missing_heatmap(df)

        with tab4:
            section_header("🧹", "Data Cleaning Assistant", "AUTO")

            c1, c2 = st.columns(2)
            with c1:
                fill_strategy = st.selectbox(
                    "Missing Value Strategy",
                    ["median", "mean", "mode", "ffill", "bfill", "drop"],
                    help="How to handle missing values in numeric columns"
                )
            with c2:
                remove_dups = st.checkbox("Remove Duplicate Rows", value=True)

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if st.button("🧹 Apply Cleaning", use_container_width=True):
                    cleaned_df = clean_dataframe(df, strategy=fill_strategy)
                    n_removed  = 0
                    if remove_dups:
                        cleaned_df, n_removed = remove_duplicates(cleaned_df)
                    st.session_state["df"] = cleaned_df
                    st.session_state["df_cleaned"] = True
                    st.success(
                        f"✅ Cleaned! Filled missing values ({fill_strategy}). "
                        f"Removed {n_removed} duplicate rows. "
                        f"New shape: {cleaned_df.shape[0]:,} × {cleaned_df.shape[1]}"
                    )
                    st.rerun()

            with col_b:
                if st.button("↩️ Reset to Original", use_container_width=True):
                    if st.session_state["df_original"] is not None:
                        st.session_state["df"] = st.session_state["df_original"].copy()
                        st.session_state["df_cleaned"] = False
                        st.success("✅ Dataset reset to original.")
                        st.rerun()

            with col_c:
                render_export_buttons(df, "Raw Data")

            if st.session_state["df_cleaned"]:
                st.info("✅ Dataset has been cleaned. All analysis uses the cleaned version.")

            # Duplicate rows preview
            dupes = df[df.duplicated()]
            if len(dupes) > 0:
                with st.expander(f"👁️ Preview {len(dupes)} duplicate rows"):
                    st.dataframe(dupes, use_container_width=True)

    else:
        # Sample data shortcut
        st.markdown("<br>", unsafe_allow_html=True)
        sample_path = os.path.join(os.path.dirname(__file__), "sample_data", "sample.csv")
        if os.path.exists(sample_path):
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("⚡ Load Sample Business Dataset", use_container_width=True):
                    df = pd.read_csv(sample_path)
                    st.session_state["df"]          = df
                    st.session_state["df_original"] = df.copy()
                    st.session_state["file_name"]   = "sample.csv"
                    st.session_state["profile"]     = profile_dataset(df)
                    st.session_state["insights"]    = generate_rule_based_insights(df)
                    st.success("✅ Sample dataset loaded!")
                    st.rerun()


# ══════════════════════════════════════════════════════════
# PAGE: AI ANALYSIS
# ══════════════════════════════════════════════════════════
elif selected_page == "🤖 AI Analysis":
    render_hero()
    if not require_dataset():
        st.stop()

    df = st.session_state["df"]
    section_header("🤖", "AI Analysis Engine", "NL → CODE → RESULT")

    # LLM status row
    ollama_ok, ollama_msg = check_ollama_status()
    col_stat, col_toggle = st.columns([3, 1])
    with col_stat:
        status_indicator("Ollama LLM", ollama_ok, ollama_msg)
    with col_toggle:
        st.session_state["llm_enabled"] = st.toggle(
            "Use AI", value=st.session_state["llm_enabled"] and ollama_ok,
            disabled=not ollama_ok,
        )

    if not ollama_ok:
        st.warning(
            "⚠️ Ollama is not running. **Install Ollama** → run `ollama serve` → `ollama pull llama3.2:3b`. "
            "You can still use manual code execution below."
        )

    neon_divider()

    # Prompt suggestions
    prompts = suggest_prompts(df)
    selected_prompt = render_prompt_chips(prompts, prefix="analysis")

    # Chat-style input
    user_input = st.chat_input("Ask anything about your data... (e.g. Show sales by region)")
    if selected_prompt and not user_input:
        user_input = selected_prompt

    if user_input:
        # Clean broken HTML tags from user input
        clean_user_input = re.sub(r"<[^>]*>", "", str(user_input)).strip()
        # Add cleaned message to history
        st.session_state["chat_history"].append({
        "role": "user",
        "content": clean_user_input
        })
        st.session_state["query_history"].append(clean_user_input)
        
        with st.spinner("🧠 AI is analyzing your query..."):
            start = time.time()

            if st.session_state["llm_enabled"] and ollama_ok:
                success, code = prompt_to_code(
                    user_input, df,
                    model=st.session_state["selected_model"],
                    conversation_history=st.session_state["chat_history"][:-1],
                )
            else:
                # Fallback: simple rule-based code generation
                success = True
                code = _generate_fallback_code(user_input, df)

            elapsed = round(time.time() - start, 2)

        if not success:
            st.error(f"❌ LLM Error: {code}")
        else:
            # Show generated code
            with st.expander("🔍 Generated Python Code", expanded=False):
                st.code(code, language="python")

            # Execute code
            with st.spinner("⚡ Executing AI-generated code safely..."):
                result = run_code(code, df)

            elapsed_total = round(time.time() - start, 2)

            if result["success"]:
                response_parts = []

                # Text output
                if result["output"].strip():
                    st.markdown(
                        f"<div style='background:rgba(0,245,255,0.04); border:1px solid rgba(0,245,255,0.15); "
                        f"border-radius:12px; padding:1rem; font-family:Space Mono,monospace; "
                        f"font-size:0.82rem; white-space:pre-wrap; color:#C8DCE8;'>{result['output']}</div>",
                        unsafe_allow_html=True
                    )
                    response_parts.append(result["output"][:300])

                # Plotly figure
                if result["figure"] is not None:
                    st.plotly_chart(result["figure"], use_container_width=True)
                    st.session_state["last_fig"] = result["figure"]
                    response_parts.append("📊 Chart generated successfully")

                # Result DataFrame
                if result["result_df"] is not None:
                    st.dataframe(result["result_df"], use_container_width=True)
                    st.session_state["last_result_df"] = result["result_df"]

                
                ai_response = " | ".join(response_parts) if response_parts else "✅ Code executed successfully."
                # Remove ALL HTML tags completely
                clean_response = re.sub(r"<[^>]*>", "", str(ai_response))
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": clean_response.strip(),
                    "exec_time": result["exec_time"],
                })

                st.markdown(
                    f"<div style='color:#00FFA3; font-size:0.78rem; margin-top:0.5rem;'>"
                    f"⚡ Executed in {result['exec_time']}s · Total: {elapsed_total}s</div>",
                    unsafe_allow_html=True
                )

            else:
                st.error(f"{result['error']}")
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": f"Error: {result['error']}",
                })

    # ── Chat History
    neon_divider()
    if st.session_state["chat_history"]:
        section_header("💬", "Conversation History", f"{len(st.session_state['chat_history'])} messages")
        for msg in reversed(st.session_state["chat_history"][-10:]):
            # Clean old cached HTML from history
            clean_msg = re.sub(r"<[^>]*>", "", str(msg["content"])).strip()
            render_chat_bubble(
                msg["role"],
                msg["content"],
                msg.get("exec_time"),
            )

        if st.button("🗑️ Clear History"):
            st.session_state["chat_history"] = []
            st.session_state["query_history"] = []
            st.rerun()

    # ── Query History
    if st.session_state["query_history"]:
        neon_divider()
        with st.expander(f"📋 Query History ({len(st.session_state['query_history'])} queries)"):
            for i, q in enumerate(reversed(st.session_state["query_history"]), 1):
                st.markdown(
                    f"<div style='padding:0.4rem 0.7rem; margin:0.2rem 0; "
                    f"border-left:2px solid #7B61FF; color:#7FA8C8; font-size:0.83rem;'>"
                    f"<strong style='color:#A78BFA;'>#{i}</strong> {q}</div>",
                    unsafe_allow_html=True
                )

    # ── Manual Code Editor
    neon_divider()
    with st.expander("⌨️ Manual Code Editor (Advanced)", expanded=False):
        section_header("⌨️", "Code Sandbox", "SAFE EXEC")
        st.markdown(
            "<div style='color:#7FA8C8; font-size:0.82rem; margin-bottom:0.75rem;'>"
            "Write Python code directly. Variable <code style='color:#00F5FF;'>df</code> is your dataset. "
            "Assign a Plotly figure to <code style='color:#00F5FF;'>fig</code> to render it.</div>",
            unsafe_allow_html=True
        )
        manual_code = st.text_area(
            "Python Code",
            height=200,
            placeholder="# Example:\nresult = df.groupby('Region')['Sales'].sum()\nprint(result)\n\nfig = px.bar(result.reset_index(), x='Region', y='Sales', template='plotly_dark')",
        )
        if st.button("▶️ Run Code", use_container_width=True):
            if manual_code.strip():
                with st.spinner("Executing..."):
                    res = run_code(manual_code, df)
                if res["success"]:
                    if res["output"]:
                        st.code(res["output"], language="text")
                    if res["figure"]:
                        st.plotly_chart(res["figure"], use_container_width=True)
                    st.success(f"✅ Executed in {res['exec_time']}s")
                else:
                    st.error(res["error"])


# ══════════════════════════════════════════════════════════
# PAGE: VISUALIZATIONS
# ══════════════════════════════════════════════════════════
elif selected_page == "📊 Visualizations":
    render_hero()
    if not require_dataset():
        st.stop()

    df = st.session_state["df"]
    render_visualizations(df)

    neon_divider()
    render_outlier_analysis(df)

    neon_divider()
    render_missing_heatmap(df)


# ══════════════════════════════════════════════════════════
# PAGE: ML STUDIO
# ══════════════════════════════════════════════════════════
elif selected_page == "🧠 ML Studio":
    render_hero()
    if not require_dataset():
        st.stop()

    df = st.session_state["df"]
    section_header("🧠", "ML Studio", "AUTO ML")

    numeric_cols = get_numeric_cols(df)
    cat_cols     = get_categorical_cols(df)
    all_cols     = df.columns.tolist()

    tab_clf, tab_reg, tab_clust = st.tabs([
        "🎯 Classification", "📈 Regression", "🔵 Clustering"
    ])

    # ── CLASSIFICATION
    with tab_clf:
        section_header("🎯", "Classification", "SUPERVISED")
        if len(all_cols) < 2:
            st.warning("Need at least 2 columns.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                clf_target = st.selectbox("Target Column", all_cols, key="clf_target")
            with c2:
                clf_model_name = st.selectbox("Model", list(CLASSIFIERS.keys()), key="clf_model")
            with c3:
                clf_test_size = st.slider("Test Split %", 10, 40, 20, key="clf_split")

            feat_cols_available = [c for c in all_cols if c != clf_target]
            clf_features = st.multiselect(
                "Feature Columns (leave blank = auto-select all)",
                feat_cols_available,
                key="clf_features"
            )
            clf_features = clf_features if clf_features else None

            adv_col1, adv_col2 = st.columns(2)
            with adv_col1:
                n_estimators = st.slider("N Estimators", 10, 300, 100, key="clf_nest")
            with adv_col2:
                max_depth = st.slider("Max Depth (0 = unlimited)", 0, 20, 0, key="clf_depth")
            max_depth = None if max_depth == 0 else max_depth

            if st.button("🚀 Train Classification Model", use_container_width=True, key="run_clf"):
                model_key = CLASSIFIERS[clf_model_name]
                with st.spinner(f"🧠 Training {clf_model_name} model..."):
                    results = run_classification(
                        df, clf_target, model_key=model_key,
                        test_size=clf_test_size / 100,
                        feature_cols=clf_features,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                    )
                st.session_state["ml_results"] = results

            if st.session_state["ml_results"] and st.session_state["ml_results"].get("task") == "classification":
                r = st.session_state["ml_results"]
                if not r["success"]:
                    st.error(f"❌ ML Error: {r['error']}")
                else:
                    st.toast("🎯 Classification model trained successfully!", icon="🔥")
                    st.success(f"✅ Model trained! Accuracy: **{r['metrics']['Accuracy']:.2f}%**")
                    import pickle

                    if "model" in r:
                        model_bytes = pickle.dumps(r["model"])

                        st.download_button(
                            label="⬇ Download Trained Model",
                            data=model_bytes,
                            file_name="trained_model.pkl",
                            mime="application/octet-stream",
                            use_container_width=True
                        )
                    neon_divider()
                    render_ml_metrics(r["metrics"], "classification")
                    neon_divider()

                    viz1, viz2, viz3 = st.columns(3)
                    if r["fig_confusion"]:
                        with viz1:
                            st.plotly_chart(r["fig_confusion"], use_container_width=True)
                    if r["fig_importance"]:
                        with viz2:
                            st.plotly_chart(r["fig_importance"], use_container_width=True)
                    if r["fig_metrics"]:
                        with viz3:
                            st.plotly_chart(r["fig_metrics"], use_container_width=True)

                    if r["predictions_df"] is not None:
                        neon_divider()
                        with st.expander("📋 Predictions Table"):
                            st.dataframe(r["predictions_df"], use_container_width=True)
                            render_export_buttons(r["predictions_df"], "Predictions")

                    if r["class_report"]:
                        with st.expander("📊 Classification Report"):
                            st.code(r["class_report"])

    # ── REGRESSION
    with tab_reg:
        section_header("📈", "Regression", "SUPERVISED")
        if not numeric_cols:
            st.warning("No numeric columns found for regression.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                reg_target = st.selectbox("Target Column", numeric_cols, key="reg_target")
            with c2:
                reg_model_name = st.selectbox("Model", list(REGRESSORS.keys()), key="reg_model")
            with c3:
                reg_test_size = st.slider("Test Split %", 10, 40, 20, key="reg_split")

            feat_cols_available_reg = [c for c in all_cols if c != reg_target]
            reg_features = st.multiselect(
                "Feature Columns (leave blank = auto-select all)",
                feat_cols_available_reg,
                key="reg_features"
            )
            reg_features = reg_features if reg_features else None

            adv_r1, adv_r2 = st.columns(2)
            with adv_r1:
                r_estimators = st.slider("N Estimators", 10, 300, 100, key="reg_nest")
            with adv_r2:
                r_depth = st.slider("Max Depth (0 = unlimited)", 0, 20, 0, key="reg_depth")
            r_depth = None if r_depth == 0 else r_depth

            if st.button("🚀 Train Regression Model", use_container_width=True, key="run_reg"):
                model_key = REGRESSORS[reg_model_name]
                with st.spinner(f"📈 Training {reg_model_name} regression model..."):
                    results = run_regression(
                        df, reg_target, model_key=model_key,
                        test_size=reg_test_size / 100,
                        feature_cols=reg_features,
                        n_estimators=r_estimators,
                        max_depth=r_depth,
                    )
                st.session_state["ml_results"] = results

            if st.session_state["ml_results"] and st.session_state["ml_results"].get("task") == "regression":
                r = st.session_state["ml_results"]
                if not r["success"]:
                    st.error(f"❌ ML Error: {r['error']}")
                else:
                    st.toast("📈 Regression model trained successfully!", icon="🚀")
                    st.success(f"✅ Model trained! R² = **{r['metrics']['R² Score']:.4f}** | RMSE = **{r['metrics']['RMSE']:.4f}**")
                    import pickle

                    if "model" in r:
                        model_bytes = pickle.dumps(r["model"])

                        st.download_button(
                            label="⬇ Download Regression Model",
                            data=model_bytes,
                            file_name="regression_model.pkl",
                            mime="application/octet-stream",
                            use_container_width=True
                        )
                    neon_divider()
                    render_ml_metrics(r["metrics"], "regression")
                    neon_divider()

                    if r["fig_actual_vs_pred"] and r["fig_residuals"]:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(r["fig_actual_vs_pred"], use_container_width=True)
                        with col2:
                            st.plotly_chart(r["fig_residuals"], use_container_width=True)

                    if r["fig_importance"]:
                        st.plotly_chart(r["fig_importance"], use_container_width=True)

                    if r["predictions_df"] is not None:
                        with st.expander("📋 Predictions Table"):
                            st.dataframe(r["predictions_df"], use_container_width=True)
                            render_export_buttons(r["predictions_df"], "Predictions")

    # ── CLUSTERING
    with tab_clust:
        section_header("🔵", "KMeans Clustering", "UNSUPERVISED")
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for clustering.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                n_clusters = st.slider("Number of Clusters (K)", 2, 10, 3, key="n_clusters")
            with c2:
                clust_features = st.multiselect(
                    "Select Feature Columns",
                    numeric_cols,
                    default=numeric_cols[:min(4, len(numeric_cols))],
                    key="clust_features"
                )

            if st.button("🚀 Run Clustering", use_container_width=True, key="run_clust"):
                if len(clust_features) < 2:
                    st.warning("Select at least 2 features.")
                else:
                    with st.spinner(f"🔵 Running KMeans clustering with K={n_clusters}..."):
                        results = run_clustering(df, feature_cols=clust_features, n_clusters=n_clusters)
                    st.session_state["ml_results"] = results

            if st.session_state["ml_results"] and st.session_state["ml_results"].get("task") == "clustering":
                r = st.session_state["ml_results"]
                if not r["success"]:
                    st.error(f"❌ Clustering Error: {r['error']}")
                else:
                    sil = r["metrics"]["Silhouette Score"]
                    st.toast("🔵 Clustering completed successfully!", icon="✨")
                    st.success(f"✅ Clustering complete! Silhouette Score: **{sil:.4f}**")
                    import pickle

                    if "model" in r:
                        model_bytes = pickle.dumps(r["model"])

                        st.download_button(
                            label="⬇ Download Clustering Model",
                            data=model_bytes,
                            file_name="clustering_model.pkl",
                            mime="application/octet-stream",
                            use_container_width=True
                        )
                    neon_divider()

                    # Metrics
                    m_cols = st.columns(4)
                    mets = [
                        ("🎯", f"{r['metrics']['Silhouette Score']:.4f}", "Silhouette"),
                        ("🔵", f"{r['metrics']['N Clusters']}", "Clusters"),
                        ("📍", f"{r['metrics']['N Points']:,}", "Points"),
                        ("📉", f"{r['metrics']['Inertia']:.0f}", "Inertia"),
                    ]
                    for col, (ico, val, lbl) in zip(m_cols, mets):
                        with col:
                            st.markdown(f"""
                            <div class="ml-metric">
                                <div style="font-size:1.5rem;">{ico}</div>
                                <div class="ml-metric-value">{val}</div>
                                <div class="ml-metric-label">{lbl}</div>
                            </div>
                            """, unsafe_allow_html=True)

                    neon_divider()
                    col1, col2 = st.columns(2)
                    if r["fig_clusters"]:
                        with col1:
                            st.plotly_chart(r["fig_clusters"], use_container_width=True)
                    if r["fig_elbow"]:
                        with col2:
                            st.plotly_chart(r["fig_elbow"], use_container_width=True)

                    if r["fig_distribution"]:
                        st.plotly_chart(r["fig_distribution"], use_container_width=True)

                    if r["cluster_df"] is not None:
                        with st.expander("📋 Cluster Assignments"):
                            st.dataframe(r["cluster_df"], use_container_width=True)
                            render_export_buttons(r["cluster_df"], "Cluster Assignments")


# ══════════════════════════════════════════════════════════
# PAGE: AI INSIGHTS
# ══════════════════════════════════════════════════════════
elif selected_page == "💡 AI Insights":
    render_hero()
    if not require_dataset():
        st.stop()

    df = st.session_state["df"]
    section_header("💡", "AI Insights Engine", "AUTO-GENERATED")

    col_regen, col_ai = st.columns([2, 1])
    with col_regen:
        if st.button("🔄 Regenerate Insights (Rule-Based)", use_container_width=True):
            st.session_state["insights"] = generate_rule_based_insights(df)
            st.rerun()

    with col_ai:
        ollama_ok, _ = check_ollama_status()
        if st.button(
            "🤖 Generate AI Insights (LLM)",
            use_container_width=True,
            disabled=not ollama_ok,
        ):
            with st.spinner("🧠 AI is analyzing your dataset..."):
                ok, ai_insights = generate_ai_insights(df, model=st.session_state["selected_model"])
            if ok and ai_insights:
                st.session_state["insights"] = ai_insights
                st.success("✅ AI insights generated!")
                st.rerun()
            else:
                st.warning("Could not generate AI insights. Showing rule-based insights.")

    neon_divider()
    insights = st.session_state.get("insights", [])
    if not insights:
        insights = generate_rule_based_insights(df)
        st.session_state["insights"] = insights

    render_insights(insights, "Dataset Intelligence Report")

    neon_divider()

    # ── Correlation Insights
    section_header("🔗", "Correlation Analysis", "PEARSON")
    numeric_cols = get_numeric_cols(df)
    if len(numeric_cols) >= 2:
        high_corr = find_high_correlations(df, threshold=0.5)
        if high_corr:
            for a, b, v in high_corr[:8]:
                icon = "🟢" if v >= 0.9 else ("🟡" if v >= 0.7 else "🔵")
                strength = "Very Strong" if v >= 0.9 else ("Strong" if v >= 0.7 else "Moderate")
                st.markdown(
                    f"<div style='padding:0.65rem 1rem; margin-bottom:0.5rem; "
                    f"border:1px solid rgba(123,97,255,0.2); border-radius:10px; "
                    f"background:rgba(123,97,255,0.05);'>"
                    f"{icon} <strong style='color:#A78BFA;'>{a}</strong> ↔ "
                    f"<strong style='color:#A78BFA;'>{b}</strong> — "
                    f"<span style='color:#00F5FF;'>r = {v}</span> "
                    f"<span style='color:#7FA8C8; font-size:0.8rem;'>({strength})</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.info("No high correlations detected (threshold: 0.5).")

        # Full correlation heatmap
        corr = df[numeric_cols[:10]].corr().round(3)
        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale=[[0, "#FF2D78"], [0.5, "#0F1F35"], [1, "#00F5FF"]],
            zmid=0, text=corr.values.round(2), texttemplate="%{text}",
            textfont={"size": 10}, showscale=True,
        ))
        fig.update_layout(title="Full Correlation Matrix", **DARK_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    neon_divider()

    # ── Statistical Summary
    section_header("📐", "Statistical Overview", "DESCRIPTIVE STATS")
    tab_num, tab_cat = st.tabs(["📊 Numeric", "🏷️ Categorical"])
    with tab_num:
        if numeric_cols:
            st.dataframe(df[numeric_cols].describe().round(3), use_container_width=True)
        else:
            st.info("No numeric columns.")
    with tab_cat:
        cat_cols = get_categorical_cols(df)
        if cat_cols:
            for col in cat_cols[:5]:
                with st.expander(f"📋 {col} — {df[col].nunique()} unique values"):
                    vc = df[col].value_counts().reset_index()
                    vc.columns = [col, "Count"]
                    vc["Percentage"] = (vc["Count"] / len(df) * 100).round(2)
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.dataframe(vc.head(15), use_container_width=True)
                    with col2:
                        fig = px.bar(
                            vc.head(10), x=col, y="Count",
                            title=f"Top 10: {col}",
                            color="Count",
                            color_continuous_scale=["#7B61FF", "#00F5FF"],
                        )
                        fig.update_layout(**DARK_LAYOUT)
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical columns.")


# ══════════════════════════════════════════════════════════
# PAGE: EXPORT
# ══════════════════════════════════════════════════════════
elif selected_page == "📤 Export":
    render_hero()
    if not require_dataset():
        st.stop()

    df = st.session_state["df"]
    section_header("📤", "Export Center", "DOWNLOAD")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='glass-container'>
            <div class='section-header'>
                <span class='section-icon'>🗂️</span>
                <span class='section-title'>Dataset Export</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        render_export_buttons(df, "Dataset")

        neon_divider()

        st.markdown("""
        <div class='glass-container'>
            <div class='section-header'>
                <span class='section-icon'>📋</span>
                <span class='section-title'>Analytics Report</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("📝 Generate Text Report", use_container_width=True):
            profile  = profile_dataset(df)
            insights = st.session_state.get("insights", generate_rule_based_insights(df))
            report   = generate_text_report(df, profile, insights)
            st.download_button(
                "⬇️ Download Report (.txt)",
                data=report.encode("utf-8"),
                file_name="datasense_report.txt",
                mime="text/plain",
                use_container_width=True,
            )

    with col2:
        st.markdown("""
        <div class='glass-container'>
            <div class='section-header'>
                <span class='section-icon'>🤖</span>
                <span class='section-title'>ML Predictions Export</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state["ml_results"] and st.session_state["ml_results"].get("success"):
            pred_df = st.session_state["ml_results"].get("predictions_df")
            if pred_df is not None:
                st.dataframe(pred_df.head(20), use_container_width=True)
                render_export_buttons(pred_df, "ML Predictions")
            else:
                st.info("No predictions table available for this model.")
        else:
            st.info("Train a model in 🧠 ML Studio first to export predictions.")

        neon_divider()

        st.markdown("""
        <div class='glass-container'>
            <div class='section-header'>
                <span class='section-icon'>📊</span>
                <span class='section-title'>Last Chart Export</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state["last_fig"] is not None:
            fig = st.session_state["last_fig"]
            st.plotly_chart(fig, use_container_width=True)
            # Export as HTML (interactive)
            html_str = fig.to_html(include_plotlyjs=True)
            st.download_button(
                "⬇️ Download Chart (Interactive HTML)",
                data=html_str.encode("utf-8"),
                file_name="datasense_chart.html",
                mime="text/html",
                use_container_width=True,
            )
        else:
            st.info("Generate a chart in 🤖 AI Analysis first.")

    neon_divider()

    # ── Dataset statistics export
    section_header("📐", "Statistics Export", "DESCRIPTIVE")
    numeric_cols = get_numeric_cols(df)
    if numeric_cols:
        stats_df = df[numeric_cols].describe().T.round(4)
        stats_df["missing"]  = df[numeric_cols].isnull().sum()
        stats_df["skewness"] = df[numeric_cols].skew().round(4)
        stats_df["kurtosis"] = df[numeric_cols].kurt().round(4)
        st.dataframe(stats_df, use_container_width=True)
        render_export_buttons(stats_df.reset_index(), "Statistics")


# ══════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ══════════════════════════════════════════════════════════
elif selected_page == "⚙️ Settings":
    render_hero()
    section_header("⚙️", "Settings & Configuration", "SYSTEM")

    tab_llm, tab_app, tab_info = st.tabs(["🤖 LLM Config", "🎨 App Config", "ℹ️ About"])

    with tab_llm:
        section_header("🤖", "LLM Configuration", "OLLAMA")

        ollama_ok, ollama_msg = check_ollama_status()
        status_indicator("Ollama Server", ollama_ok, ollama_msg)

        neon_divider()

        c1, c2 = st.columns(2)
        with c1:
            available_models = get_available_models()
            if available_models:
                selected = st.selectbox(
                    "Active Model",
                    available_models,
                    index=0,
                )
                st.session_state["selected_model"] = selected
            else:
                st.info("No Ollama models found. Run: `ollama pull llama3.2:3b`")

        with c2:
            st.session_state["llm_enabled"] = st.toggle(
                "Enable AI Analysis",
                value=st.session_state["llm_enabled"],
            )

        neon_divider()
        st.markdown("#### 📦 Setup Commands")
        setup_cmds = {
            "Install Ollama (Linux/Mac)":     "curl -fsSL https://ollama.ai/install.sh | sh",
            "Install Ollama (Windows)":        "winget install Ollama.Ollama",
            "Start Ollama Server":             "ollama serve",
            "Pull Llama 3.1 (8B)":            "ollama pull llama3.2:3b",
            "Pull Llama 3.1 (70B)":           "ollama pull llama3.2:3b:70b",
            "List Models":                     "ollama list",
        }
        for label, cmd in setup_cmds.items():
            col_l, col_c = st.columns([2, 3])
            with col_l:
                st.markdown(f"<span style='color:#7FA8C8; font-size:0.85rem;'>{label}</span>", unsafe_allow_html=True)
            with col_c:
                st.code(cmd, language="bash")

    with tab_app:
        section_header("🎨", "Application Settings", "UI/UX")

        st.markdown("#### Session Management")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state["chat_history"]  = []
                st.session_state["query_history"] = []
                st.success("Chat cleared!")

        with col2:
            if st.button("🗑️ Clear ML Results", use_container_width=True):
                st.session_state["ml_results"] = None
                st.success("ML results cleared!")

        with col3:
            if st.button("🔄 Full Reset", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("Full reset done!")
                st.rerun()

        neon_divider()
        st.markdown("#### Current Session State")
        state_info = {
            "Dataset Loaded":    st.session_state["df"] is not None,
            "File Name":         st.session_state["file_name"] or "N/A",
            "Dataset Cleaned":   st.session_state["df_cleaned"],
            "Chat Messages":     len(st.session_state["chat_history"]),
            "Queries Run":       len(st.session_state["query_history"]),
            "ML Model Trained":  st.session_state["ml_results"] is not None,
            "LLM Enabled":       st.session_state["llm_enabled"],
            "Active Model":      st.session_state["selected_model"],
        }
        for k, v in state_info.items():
            col_k, col_v = st.columns([2, 3])
            with col_k:
                st.markdown(f"<span style='color:#7FA8C8; font-size:0.85rem;'>{k}</span>", unsafe_allow_html=True)
            with col_v:
                color = "#00FFA3" if v is True else ("#FF2D78" if v is False else "#00F5FF")
                st.markdown(f"<span style='color:{color}; font-size:0.85rem; font-weight:500;'>{v}</span>", unsafe_allow_html=True)

    with tab_info:
        st.markdown("""
        <div class='glass-container' style='padding:2.5rem;'>
            <div style='text-align:center;'>
                <div style='font-size:3rem; margin-bottom:0.75rem;'>🧠</div>
                <div style='font-size:1.8rem; font-weight:800; background:linear-gradient(135deg,#00F5FF,#7B61FF,#00FFA3);
                            -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:0.4rem;'>
                    Personal AI Data Analyst
                </div>
                <div style='color:#7FA8C8; font-size:0.95rem; margin-bottom:1.5rem;'>
                    v2.0 · AI-Powered Data Analytics and Machine Learning Platform
                </div>
            </div>

            <div style='display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-top:1.5rem;'>
                <div>
                    <div style='color:#00F5FF; font-weight:600; margin-bottom:0.5rem;'>🛠️ Tech Stack</div>
                    <div style='color:#7FA8C8; font-size:0.85rem; line-height:2;'>
                        Python 3.10+ · Streamlit · Pandas · NumPy<br>
                        Plotly · Scikit-learn · SciPy · Statsmodels<br>
                        Ollama · Llama 3.1 · Custom CSS
                    </div>
                </div>
                <div>
                    <div style='color:#7B61FF; font-weight:600; margin-bottom:0.5rem;'>✨ Features</div>
                    <div style='color:#7FA8C8; font-size:0.85rem; line-height:2;'>
                        Multi-format upload · AI NL→Code<br>
                        10+ Chart Types · Auto ML Pipeline<br>
                        Outlier Detection · Export Reports
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
