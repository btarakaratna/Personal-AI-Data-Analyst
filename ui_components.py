"""
ui_components.py — Personal AI Data Analyst
Reusable Streamlit UI components with premium glassmorphism styling.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Tuple

from utils import (
    PLOTLY_THEME, PLOTLY_PAPER, PLOTLY_PLOT, PLOTLY_FONT,
    PLOTLY_CYAN, PLOTLY_VIOLET, PLOTLY_GREEN, PLOTLY_PALETTE,
    profile_dataset, get_numeric_cols, get_categorical_cols, get_datetime_cols,
    format_number, find_high_correlations, detect_outliers_iqr,
)

DARK_LAYOUT = dict(
    template      = "plotly_dark",
    paper_bgcolor = PLOTLY_PAPER,
    plot_bgcolor  = PLOTLY_PLOT,
    font          = dict(family="Poppins, sans-serif", color=PLOTLY_FONT),
    title_font    = dict(size=16, color=PLOTLY_CYAN),
    margin        = dict(l=10, r=10, t=50, b=10),
    colorway      = PLOTLY_PALETTE,
)


# ─── HERO BANNER ──────────────────────────────────────────

def render_hero():
    """Render the animated hero banner."""
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">⚡ POWERED BY AI · REAL-TIME ANALYTICS · ML READY</div>
        <div class="hero-title">Personal AI Data Analyst</div>
        <div class="hero-subtitle">
            Next-Generation AI-Powered Data Analytics Platform<br>
            <span style="color: #7B61FF; font-size: 0.9rem;">
                Upload → Analyze → Visualize → Predict → Export
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── METRIC CARDS ─────────────────────────────────────────

def metric_card(icon: str, value: str, label: str, gradient: str = "") -> str:
    style = f'style="--gradient: {gradient};"' if gradient else ""
    return f"""
    <div class="metric-card" {style}>
        <span class="metric-icon">{icon}</span>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def render_dataset_metrics(df: pd.DataFrame):
    """Render 6 premium metric cards for dataset overview."""
    profile = profile_dataset(df)

    cards_data = [
        ("🗂️",  f"{profile['n_rows']:,}",               "Total Rows",
         "linear-gradient(90deg, #00F5FF, #7B61FF)"),
        ("📐",  f"{profile['n_cols']}",                  "Total Columns",
         "linear-gradient(90deg, #7B61FF, #00FFA3)"),
        ("⚠️",  f"{profile['missing_total']:,}",          "Missing Values",
         "linear-gradient(90deg, #FF6B35, #FF2D78)"),
        ("📊",  f"{len(profile['numeric_cols'])}",        "Numeric Cols",
         "linear-gradient(90deg, #00FFA3, #00F5FF)"),
        ("🧪",  f"{profile['memory_mb']} MB",             "Memory Usage",
         "linear-gradient(90deg, #A78BFA, #7B61FF)"),
        ("♻️",  f"{profile['duplicates']}",               "Duplicate Rows",
         "linear-gradient(90deg, #FFD700, #FF6B35)"),
    ]

    cols = st.columns(6)
    for col, (icon, val, label, grad) in zip(cols, cards_data):
        with col:
            st.markdown(metric_card(icon, val, label, grad), unsafe_allow_html=True)


def render_ml_metrics(metrics: dict, task: str):
    """Render ML result metric cards."""
    if task == "classification":
        cards = [
            ("🎯", f"{metrics.get('Accuracy', 0):.1f}%",  "Accuracy"),
            ("⚖️", f"{metrics.get('F1 Score', 0):.1f}%",  "F1 Score"),
            ("🔍", f"{metrics.get('Precision', 0):.1f}%", "Precision"),
            ("📡", f"{metrics.get('Recall', 0):.1f}%",    "Recall"),
        ]
    else:
        cards = [
            ("📈", f"{metrics.get('R² Score', 0):.4f}",   "R² Score"),
            ("📉", f"{metrics.get('RMSE', 0):.4f}",       "RMSE"),
            ("📏", f"{metrics.get('MAE', 0):.4f}",        "MAE"),
            ("🎓", f"{metrics.get('Train Size', 0):,}",   "Train Samples"),
        ]

    cols = st.columns(len(cards))
    for col, (icon, val, label) in zip(cols, cards):
        with col:
            st.markdown(f"""
            <div class="ml-metric">
                <div style="font-size:1.5rem; margin-bottom:0.4rem;">{icon}</div>
                <div class="ml-metric-value">{val}</div>
                <div class="ml-metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


# ─── SECTION HEADERS ──────────────────────────────────────

def section_header(icon: str, title: str, badge: str = ""):
    badge_html = f'<span class="section-badge">{badge}</span>' if badge else ""
    st.markdown(f"""
    <div class="section-header">
        <span class="section-icon">{icon}</span>
        <span class="section-title">{title}</span>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)


def neon_divider():
    st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)


# ─── DATA PREVIEW ─────────────────────────────────────────

def render_data_preview(df: pd.DataFrame):
    """Interactive, searchable, paginated dataframe preview."""
    section_header("🗃️", "Data Preview", "INTERACTIVE")

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        search = st.text_input("🔍 Search in data", placeholder="Type to filter rows...")
    with col2:
        page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=0)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        show_dtypes = st.checkbox("Show types", value=False)

    # Filter
    display_df = df.copy()
    if search:
        mask = display_df.astype(str).apply(
            lambda col: col.str.contains(search, case=False, na=False)
        ).any(axis=1)
        display_df = display_df[mask]

    total_rows = len(display_df)
    total_pages = max(1, (total_rows - 1) // page_size + 1)

    col_page, col_info = st.columns([1, 3])
    with col_page:
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    with col_info:
        st.markdown(
            f"<br><span style='color:#7FA8C8; font-size:0.85rem;'>"
            f"Showing {min(page_size, total_rows)} of {total_rows:,} rows (filtered)</span>",
            unsafe_allow_html=True
        )

    start_idx = (page - 1) * page_size
    end_idx   = start_idx + page_size
    page_df   = display_df.iloc[start_idx:end_idx]

    if show_dtypes:
        dtype_row = pd.DataFrame(
            [df.dtypes.astype(str).to_dict()], index=["dtype"]
        )
        st.dataframe(dtype_row, use_container_width=True)

    st.dataframe(
        page_df,
        use_container_width=True,
        height=min(400, 45 * (len(page_df) + 1)),
    )


# ─── COLUMN PROFILER ──────────────────────────────────────

def render_column_profiler(df: pd.DataFrame):
    """Detailed per-column statistics."""
    section_header("🔬", "Column Profiler", "DEEP SCAN")

    numeric_cols = get_numeric_cols(df)
    cat_cols     = get_categorical_cols(df)

    tab1, tab2 = st.tabs(["📊 Numeric Columns", "🏷️ Categorical Columns"])

    with tab1:
        if not numeric_cols:
            st.info("No numeric columns found.")
        else:
            stats = df[numeric_cols].describe().T.round(3)
            stats["missing"] = df[numeric_cols].isnull().sum()
            stats["missing%"] = (df[numeric_cols].isnull().mean() * 100).round(2)
            stats["skewness"] = df[numeric_cols].skew().round(3)
            stats["kurtosis"] = df[numeric_cols].kurt().round(3)
            outlier_counts = {}
            for col in numeric_cols:
                mask = detect_outliers_iqr(df[col].dropna())
                outlier_counts[col] = int(mask.sum())
            stats["outliers"] = pd.Series(outlier_counts)
            st.dataframe(stats, use_container_width=True)

    with tab2:
        if not cat_cols:
            st.info("No categorical columns found.")
        else:
            cat_data = []
            for col in cat_cols:
                n_unique = df[col].nunique()
                missing  = df[col].isnull().sum()
                top_val  = df[col].value_counts().index[0] if not df[col].value_counts().empty else "N/A"
                top_freq = df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0
                cat_data.append({
                    "Column": col,
                    "Unique Values": n_unique,
                    "Missing": missing,
                    "Missing %": round(missing / len(df) * 100, 2),
                    "Top Value": str(top_val),
                    "Top Frequency": top_freq,
                })
            st.dataframe(pd.DataFrame(cat_data), use_container_width=True)


# ─── VISUALIZATION DASHBOARD ──────────────────────────────

def render_visualizations(df: pd.DataFrame):
    """Full interactive visualization dashboard with 10+ chart types."""
    section_header("📊", "Visualization Dashboard", "INTERACTIVE")

    numeric_cols = get_numeric_cols(df)
    cat_cols     = get_categorical_cols(df)
    dt_cols      = get_datetime_cols(df)

    chart_types = [
        "📊 Histogram", "🔵 Scatter Plot", "🌡️ Correlation Heatmap",
        "📦 Box Plot", "🥧 Pie Chart", "📈 Bar Chart",
        "🎻 Violin Plot", "📉 Time Series", "🌊 Distribution Plot",
    ]

    selected_chart = st.selectbox("Select Chart Type", chart_types)

    # ── Histogram
    if "Histogram" in selected_chart:
        if not numeric_cols:
            st.warning("No numeric columns for histogram.")
            return
        col = st.selectbox("Select Column", numeric_cols)
        nbins = st.slider("Bins", 10, 100, 30)

        fig = px.histogram(
            df, x=col, nbins=nbins,
            title=f"Distribution of {col}",
            color_discrete_sequence=[PLOTLY_CYAN],
            marginal="box",
        )
        fig.update_traces(marker_line_width=0.5, marker_line_color="rgba(255,255,255,0.1)")
        fig.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # ── Scatter
    elif "Scatter" in selected_chart:
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns.")
            return
        c1, c2, c3 = st.columns(3)
        with c1: x_col = st.selectbox("X Axis", numeric_cols)
        with c2: y_col = st.selectbox("Y Axis", numeric_cols, index=min(1, len(numeric_cols)-1))
        with c3: color_col = st.selectbox("Color By", ["None"] + cat_cols)

        color = None if color_col == "None" else color_col
        fig = px.scatter(
            df, x=x_col, y=y_col, color=color,
            title=f"{x_col} vs {y_col}",
            color_discrete_sequence=PLOTLY_PALETTE,
            trendline="ols" if not color else None,
            opacity=0.7,
        )
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # ── Correlation Heatmap
    elif "Heatmap" in selected_chart:
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns.")
            return
        selected_num = st.multiselect(
            "Select Columns", numeric_cols, default=numeric_cols[:min(8, len(numeric_cols))]
        )
        if len(selected_num) < 2:
            st.warning("Select at least 2 columns.")
            return

        corr = df[selected_num].corr().round(3)
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale=[
                [0,   "#FF2D78"],
                [0.5, "#0F1F35"],
                [1,   "#00F5FF"],
            ],
            zmid=0,
            text=corr.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 11},
            showscale=True,
        ))
        fig.update_layout(title="Correlation Heatmap", **DARK_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        # Show strong correlations
        high = find_high_correlations(df[selected_num])
        if high:
            st.markdown("**Strong Correlations Detected:**")
            for a, b, v in high[:5]:
                color = "🟢" if v > 0.9 else "🟡"
                st.markdown(f"{color} `{a}` ↔ `{b}` — **r = {v}**")

    # ── Box Plot
    elif "Box" in selected_chart:
        if not numeric_cols:
            st.warning("No numeric columns.")
            return
        c1, c2 = st.columns(2)
        with c1: y_col = st.selectbox("Value Column", numeric_cols)
        with c2: x_col = st.selectbox("Group By", ["None"] + cat_cols)

        x = None if x_col == "None" else x_col
        fig = px.box(
            df, y=y_col, x=x,
            title=f"Box Plot: {y_col}",
            color=x,
            color_discrete_sequence=PLOTLY_PALETTE,
            points="outliers",
            notched=True,
        )
        fig.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # ── Pie Chart
    elif "Pie" in selected_chart:
        if not cat_cols:
            st.warning("No categorical columns.")
            return
        c1, c2 = st.columns(2)
        with c1: cat_col = st.selectbox("Category Column", cat_cols)
        with c2:
            val_col = st.selectbox("Value Column", ["Count"] + numeric_cols)

        if val_col == "Count":
            pie_data = df[cat_col].value_counts().reset_index()
            pie_data.columns = [cat_col, "Count"]
            values = "Count"
        else:
            pie_data = df.groupby(cat_col)[val_col].sum().reset_index()
            values   = val_col

        top_n = st.slider("Top N categories", 3, 20, 8)
        pie_data = pie_data.head(top_n)

        fig = px.pie(
            pie_data, names=cat_col, values=values,
            title=f"{cat_col} Distribution",
            color_discrete_sequence=PLOTLY_PALETTE,
            hole=0.4,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # ── Bar Chart
    elif "Bar" in selected_chart:
        if not cat_cols or not numeric_cols:
            st.warning("Need categorical and numeric columns.")
            return
        c1, c2, c3 = st.columns(3)
        with c1: x_col  = st.selectbox("Category", cat_cols)
        with c2: y_col  = st.selectbox("Value", numeric_cols)
        with c3: agg    = st.selectbox("Aggregation", ["sum", "mean", "count", "max", "min"])

        grouped = df.groupby(x_col)[y_col].agg(agg).reset_index()
        grouped.columns = [x_col, y_col]
        grouped = grouped.sort_values(y_col, ascending=False).head(20)

        orientation = st.radio("Orientation", ["Vertical", "Horizontal"], horizontal=True)
        if orientation == "Horizontal":
            fig = px.bar(
                grouped, x=y_col, y=x_col, orientation="h",
                title=f"{agg.title()} of {y_col} by {x_col}",
                color=y_col, color_continuous_scale=[PLOTLY_VIOLET, PLOTLY_CYAN],
            )
        else:
            fig = px.bar(
                grouped, x=x_col, y=y_col,
                title=f"{agg.title()} of {y_col} by {x_col}",
                color=y_col, color_continuous_scale=[PLOTLY_VIOLET, PLOTLY_CYAN],
            )
        fig.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # ── Violin Plot
    elif "Violin" in selected_chart:
        if not numeric_cols:
            st.warning("No numeric columns.")
            return
        c1, c2 = st.columns(2)
        with c1: y_col = st.selectbox("Value", numeric_cols)
        with c2: x_col = st.selectbox("Group By", ["None"] + cat_cols)

        x = None if x_col == "None" else x_col
        fig = px.violin(
            df, y=y_col, x=x,
            color=x,
            color_discrete_sequence=PLOTLY_PALETTE,
            box=True, points="outliers",
            title=f"Violin Plot: {y_col}",
        )
        fig.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # ── Time Series
    elif "Time" in selected_chart:
        dt_candidates = dt_cols[:]
        # Try to find datetime-parseable columns
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower():
                if col not in dt_candidates:
                    dt_candidates.append(col)

        if not dt_candidates:
            st.warning("No datetime columns detected.")
            return
        if not numeric_cols:
            st.warning("No numeric columns.")
            return

        c1, c2, c3 = st.columns(3)
        with c1: date_col = st.selectbox("Date Column", dt_candidates)
        with c2: val_col  = st.selectbox("Value Column", numeric_cols)
        with c3: agg      = st.selectbox("Aggregation", ["sum", "mean", "max"], key="ts_agg")

        try:
            ts_df = df[[date_col, val_col]].copy()
            ts_df[date_col] = pd.to_datetime(ts_df[date_col])
            ts_df = ts_df.dropna()
            ts_df = ts_df.groupby(date_col)[val_col].agg(agg).reset_index()
            ts_df = ts_df.sort_values(date_col)

            fig = px.line(
                ts_df, x=date_col, y=val_col,
                title=f"{val_col} over Time ({agg})",
                color_discrete_sequence=[PLOTLY_CYAN],
            )
            fig.update_traces(line=dict(width=2))
            fig.add_trace(go.Scatter(
                x=ts_df[date_col], y=ts_df[val_col],
                fill="tozeroy",
                fillcolor="rgba(0,245,255,0.06)",
                line=dict(width=0),
                showlegend=False,
            ))
            fig.update_layout(**DARK_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Time series error: {e}")

    # ── Distribution Plot (KDE)
    elif "Distribution" in selected_chart:
        if not numeric_cols:
            st.warning("No numeric columns.")
            return
        cols_sel = st.multiselect(
            "Select Columns", numeric_cols,
            default=numeric_cols[:min(3, len(numeric_cols))]
        )
        if not cols_sel:
            st.warning("Select at least 1 column.")
            return

        fig = go.Figure()
        for i, col in enumerate(cols_sel):
            s = df[col].dropna()
            fig.add_trace(go.Histogram(
                x=s, name=col, histnorm="probability density",
                opacity=0.5,
                marker_color=PLOTLY_PALETTE[i % len(PLOTLY_PALETTE)],
                nbinsx=40,
            ))
        fig.update_layout(
            title="Overlapping Distributions",
            barmode="overlay",
            **DARK_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)


# ─── MISSING VALUE HEATMAP ────────────────────────────────

def render_missing_heatmap(df: pd.DataFrame):
    """Render a missing values heatmap."""
    section_header("🕳️", "Missing Value Analysis", "HEATMAP")

    missing = df.isnull()
    if missing.sum().sum() == 0:
        st.success("✅ No missing values in the dataset!")
        return

    cols_with_missing = missing.columns[missing.any()].tolist()
    if not cols_with_missing:
        st.success("No missing values.")
        return

    # Sample for performance
    sample_df = missing[cols_with_missing].head(200).astype(int)

    fig = px.imshow(
        sample_df.T,
        color_continuous_scale=[[0, "#0F1F35"], [1, "#FF2D78"]],
        title=f"Missing Value Pattern (first 200 rows, {len(cols_with_missing)} columns with missing)",
        aspect="auto",
        labels=dict(color="Missing"),
    )
    fig.update_layout(**DARK_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    # Bar chart of missing per column
    miss_counts = df.isnull().sum()[df.isnull().sum() > 0].reset_index()
    miss_counts.columns = ["Column", "Missing"]
    miss_counts["Missing %"] = (miss_counts["Missing"] / len(df) * 100).round(2)
    miss_counts = miss_counts.sort_values("Missing", ascending=True)

    fig2 = px.bar(
        miss_counts, x="Missing", y="Column", orientation="h",
        title="Missing Values per Column",
        color="Missing %",
        color_continuous_scale=["#7B61FF", "#FF2D78"],
        text="Missing %",
    )
    fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig2.update_layout(**DARK_LAYOUT)
    st.plotly_chart(fig2, use_container_width=True)


# ─── OUTLIER VISUALIZATION ────────────────────────────────

def render_outlier_analysis(df: pd.DataFrame):
    """Render outlier detection results."""
    section_header("📡", "Outlier Detection", "Z-SCORE + IQR")

    numeric_cols = get_numeric_cols(df)
    if not numeric_cols:
        st.info("No numeric columns for outlier analysis.")
        return

    col_sel = st.selectbox("Select Column", numeric_cols, key="outlier_col")
    method  = st.radio("Method", ["IQR", "Z-Score"], horizontal=True, key="outlier_method")

    s = df[col_sel].dropna()

    if method == "IQR":
        mask = detect_outliers_iqr(s)
        outliers     = s[mask]
        non_outliers = s[~mask]
        title_method = "IQR"
    else:
        z_scores = (s - s.mean()) / (s.std() + 1e-9)
        threshold = st.slider("Z-Score Threshold", 1.5, 5.0, 3.0, 0.1)
        mask = z_scores.abs() > threshold
        outliers     = s[mask]
        non_outliers = s[~mask]
        title_method = f"Z-Score (|z|>{threshold})"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=non_outliers.index, y=non_outliers.values,
        mode="markers",
        name="Normal",
        marker=dict(color=PLOTLY_CYAN, size=5, opacity=0.6),
    ))
    fig.add_trace(go.Scatter(
        x=outliers.index, y=outliers.values,
        mode="markers",
        name="Outliers",
        marker=dict(color="#FF2D78", size=9, symbol="x", opacity=0.9,
                    line=dict(width=2, color="#FF2D78")),
    ))
    fig.update_layout(
        title=f"Outliers in '{col_sel}' ({title_method}) — {len(outliers)} outliers",
        xaxis_title="Index",
        yaxis_title=col_sel,
        **DARK_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Outliers", len(outliers))
    c2.metric("Outlier %", f"{len(outliers)/len(s)*100:.2f}%")
    c3.metric("Clean Points", len(non_outliers))


# ─── AI PROMPT SECTION ────────────────────────────────────

def render_prompt_chips(prompts: list, prefix: str = "chip") -> Optional[str]:
    """Render clickable prompt suggestion chips. Returns selected prompt."""
    selected = None
    chip_html = "<div style='display:flex; flex-wrap:wrap; gap:0.4rem; margin-bottom:1rem;'>"
    for i, p in enumerate(prompts[:12]):
        chip_html += f'<span class="prompt-chip">💡 {p}</span>'
    chip_html += "</div>"
    st.markdown(chip_html, unsafe_allow_html=True)

    # Use buttons for actual selection (Streamlit limitation)
    with st.expander("📋 Click a prompt suggestion"):
        cols = st.columns(2)
        for i, p in enumerate(prompts[:12]):
            with cols[i % 2]:
                if st.button(f"💡 {p}", key=f"{prefix}_{i}", use_container_width=True):
                    selected = p
    return selected


def render_chat_bubble(role: str, content: str, exec_time: float = None):
    """
    Clean Streamlit native chat bubble.
    Completely removes HTML/div rendering issues.
    """

    import re
    import streamlit as st

    # Convert content to clean plain text
    clean_content = str(content)

    # Remove ALL html tags completely
    clean_content = re.sub(r"<[^>]*>", "", clean_content)

    # Remove extra spaces/newlines
    clean_content = clean_content.strip()

    # USER MESSAGE
    if role == "user":

        with st.chat_message("user"):
            st.write(clean_content)

    # AI MESSAGE
    else:

        with st.chat_message("assistant"):

            st.write(clean_content)

            if exec_time is not None:
                st.caption(f"⏱ {exec_time:.2f}s")


# ─── INSIGHTS PANEL ───────────────────────────────────────

def render_insights(insights: list, title: str = "AI Insights"):
    """Render insight cards with color-coded types."""
    section_header("💡", title, f"{len(insights)} insights")

    color_map = {
        "✅": "positive", "⚠️": "warning", "🔴": "danger",
        "📊": "info",     "📈": "info",     "📉": "info",
        "🔗": "purple",   "🚀": "positive", "💾": "warning",
        "🏷️": "info",     "📐": "info",     "♻️": "warning",
    }

    for insight in insights:
        emoji = insight[:2] if len(insight) >= 2 else "ℹ️"
        card_class = color_map.get(emoji, "info")
        st.markdown(f"""
        <div class="insight-card {card_class}">
            {insight}
        </div>
        """, unsafe_allow_html=True)


# ─── STATUS INDICATOR ─────────────────────────────────────

def status_indicator(label: str, status: bool, detail: str = ""):
    dot_class = "" if status else "offline"
    detail_html = f"<span style='color:#7FA8C8; font-size:0.8rem;'> — {detail}</span>" if detail else ""
    st.markdown(f"""
    <div style='display:flex; align-items:center; padding:0.5rem 0;'>
        <span class="status-dot {dot_class}"></span>
        <span style='color: {"#00FFA3" if status else "#FF2D78"}; font-weight:500;'>{label}</span>
        {detail_html}
    </div>
    """, unsafe_allow_html=True)


# ─── EXPORT PANEL ─────────────────────────────────────────

def render_export_buttons(df: pd.DataFrame, label: str = "Dataset"):
    """Render download buttons for CSV and JSON."""
    from utils import df_to_csv_bytes

    c1, c2 = st.columns(2)
    with c1:
        csv_bytes = df_to_csv_bytes(df)
        st.download_button(
            f"⬇️ Download {label} (CSV)",
            data=csv_bytes,
            file_name=f"datasense_{label.lower().replace(' ','_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        json_bytes = df.to_json(orient="records", date_format="iso").encode("utf-8")
        st.download_button(
            f"⬇️ Download {label} (JSON)",
            data=json_bytes,
            file_name=f"datasense_{label.lower().replace(' ','_')}.json",
            mime="application/json",
            use_container_width=True,
        )


# ─── SIDEBAR BRANDING ─────────────────────────────────────

def render_sidebar_brand():
    st.markdown("""
    <div class="sidebar-brand">
        <div style="font-size: 2rem; margin-bottom: 0.3rem;">🧠</div>
        <div class="sidebar-brand-name">Personal AI Data Analyst</div>
        <div class="sidebar-brand-version">v2.0 · PREMIUM EDITION</div>
    </div>
    """, unsafe_allow_html=True)
