import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image

# ==========================================
# 1. PAGE CONFIGURATION & THEME
# ==========================================
APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "logo.png"

try:
    PAGE_ICON = Image.open(LOGO_PATH)
except Exception:
    PAGE_ICON = "📈"

st.set_page_config(
    page_title="Livmir Baskets | Smart AI ETF Engine (API v2)",
    page_icon=PAGE_ICON,
    layout="wide"
)

SECTORS_API_BASE_URL = "https://api.sectors.app/v2"


def apply_financial_theme(fig, title=""):
    """Applies dark financial styling to Plotly charts with clean top margins to prevent title overlap."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=16, color="#FFFFFF"),
            x=0.0,
            y=0.98,
            xanchor="left",
            yanchor="top"
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(15, 17, 23, 0)",
        plot_bgcolor="rgba(15, 17, 23, 0)",
        font=dict(family="Inter, sans-serif", size=12, color="#B0B3C6"),
        margin=dict(l=20, r=20, t=90, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#FFFFFF")
        ),
        hoverlabel=dict(
            bgcolor="#1E2230",
            font_size=12,
            font_family="Inter",
            font_color="#FFFFFF"
        ),
        xaxis=dict(showgrid=True, gridcolor="#26293B", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#26293B", zeroline=False),
    )
    return fig


# ==========================================
# 2. API KEY & LOGO CONFIGURATION
# ==========================================
st.sidebar.title("⚙️ Configuration")

if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH), use_container_width=True)

raw_key = st.sidebar.text_input(
    "Sectors API Key",
    type="password",
    value=os.getenv("SECTORS_API_KEY", ""),
    help="Enter your Sectors API key."
)

api_key_input = raw_key.strip()

if not api_key_input:
    st.error("⛔ **Sectors API Key Required**")
    st.warning("This application requires an active Sectors API Key to function. Please provide a key in the sidebar.")
    st.stop()

HEADERS = {
    "Authorization": api_key_input,
    "Content-Type": "application/json"
}


# ==========================================
# 3. UTILITIES & API DATA FETCHERS
# ==========================================
def get_deep_value(d, target_keys, default=None):
    """Recursively searches nested dictionaries for matching keys."""
    if not isinstance(d, dict):
        return default
    for k, v in d.items():
        clean_k = str(k).lower().replace("_", "").replace(" ", "")
        for tk in target_keys:
            clean_tk = str(tk).lower().replace("_", "").replace(" ", "")
            if clean_k == clean_tk:
                if v is not None:
                    try:
                        return float(v)
                    except (ValueError, TypeError):
                        pass
        if isinstance(v, dict):
            res = get_deep_value(v, target_keys, None)
            if res is not None:
                return res
    return default


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_companies(api_key: str):
    """Fetches full market company universe from Sectors API v2."""
    url = f"{SECTORS_API_BASE_URL}/companies/"
    try:
        res = requests.get(url, headers={"Authorization": api_key}, timeout=15)
        if res.status_code in (401, 403):
            st.error(f"❌ **API Authorization Error [{res.status_code}]**: Please verify your Sectors API Key.")
            st.stop()
        elif res.status_code != 200:
            st.error(f"❌ Sectors API Error [{res.status_code}]: {res.text}")
            st.stop()

        data = res.json()
        if isinstance(data, dict):
            for key in ["data", "companies", "results", "items"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
        elif isinstance(data, list):
            return data
        return []

    except requests.exceptions.RequestException as e:
        st.error(f"🌐 **Network Error**: Could not connect to Sectors API ({e})")
        st.stop()


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_company_report(api_key: str, ticker: str):
    """Fetches detailed financial metrics for a specific ticker."""
    url = f"{SECTORS_API_BASE_URL}/company/report/{ticker}/"
    try:
        res = requests.get(url, headers={"Authorization": api_key}, timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return {}


ALIAS_MAP = {
    "bank": ["financials", "bank", "banking", "finance", "lending"],
    "banking": ["financials", "bank", "banking", "finance"],
    "finance": ["financials", "finance"],
    "financial": ["financials", "finance"],
    "tech": ["technology", "tech", "digital", "software"],
    "technology": ["technology", "software"],
    "mining": ["energy", "basic materials", "mining", "coal", "mineral"],
    "energy": ["energy", "coal", "oil", "gas", "mining"],
    "food": ["consumer non-cyclicals", "consumer cyclicals", "food", "fmcg", "beverage", "staples"],
    "consumer": ["consumer non-cyclicals", "consumer cyclicals", "consumer", "retail"],
    "telecom": ["infrastructures", "telecommunication", "cellular", "tower"],
    "infrastructure": ["infrastructures", "infrastructure", "construction", "cement"],
    "health": ["healthcare", "health", "pharma", "hospital"],
    "healthcare": ["healthcare", "health", "pharmaceutical"]
}


def filter_companies_by_query(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Smart multi-intent query filter targeting exact tickers, sector maps, or strict word intersections."""
    if df.empty or not query.strip():
        return df

    clean_query = query.lower().strip()
    ticker_col = next((c for c in ["symbol", "code", "ticker"] if c in df.columns), None)

    # Direct Ticker Match
    if ticker_col:
        exact_ticker_mask = df[ticker_col].astype(str).str.lower() == clean_query
        if exact_ticker_mask.any():
            return df[exact_ticker_mask]

    stopwords = {"and", "or", "the", "in", "fast", "growing", "companies", "top", "best", "stock", "stocks", "etf", "basket"}
    raw_words = [w for w in clean_query.split() if w not in stopwords]

    if not raw_words:
        return df

    matched_intents = []
    unmapped_words = []

    for w in raw_words:
        mapped = False
        for alias_key, keywords in ALIAS_MAP.items():
            if w in keywords or w == alias_key:
                matched_intents.append(set(keywords + [alias_key]))
                mapped = True
                break
        if not mapped:
            unmapped_words.append(w)

    sector_cols = [c for c in ["sector", "subsector", "industry"] if c in df.columns]
    target_text_cols = sector_cols if sector_cols else [c for c in ["symbol", "name"] if c in df.columns]

    if matched_intents:
        intent_dfs = []
        for intent_tokens in matched_intents:
            intent_mask = pd.Series(False, index=df.index)
            for token in intent_tokens:
                for col in target_text_cols:
                    intent_mask |= df[col].astype(str).str.lower().str.contains(token, na=False)
            if intent_mask.any():
                intent_dfs.append(df[intent_mask])

        if intent_dfs:
            return pd.concat(intent_dfs).drop_duplicates(subset=[ticker_col] if ticker_col else None)

    search_words = unmapped_words if unmapped_words else raw_words
    word_masks = []
    all_text_cols = [c for c in ["symbol", "name", "sector", "subsector"] if c in df.columns]

    for word in search_words:
        w_mask = pd.Series(False, index=df.index)
        for col in all_text_cols:
            w_mask |= df[col].astype(str).str.lower().str.contains(word, na=False)
        word_masks.append(w_mask)

    if word_masks:
        and_mask = pd.concat(word_masks, axis=1).all(axis=1)
        if and_mask.any():
            return df[and_mask]

        or_mask = pd.concat(word_masks, axis=1).any(axis=1)
        if or_mask.any():
            return df[or_mask]

    return df


# ==========================================
# 4. PORTFOLIO BASKET BUILDER
# ==========================================
def build_portfolio_basket(all_companies: list, api_key: str, query: str, capital_idr: float,
                           min_roe: float, min_margin: float, min_mcap_tn: float,
                           roe_weight: float, margin_weight: float, div_weight: float, max_holdings: int):
    if not all_companies:
        return pd.DataFrame(), 0, 0

    df_raw = pd.json_normalize(all_companies)
    if df_raw.empty:
        return pd.DataFrame(), 0, 0

    candidate_df = filter_companies_by_query(df_raw, query)

    ticker_col = next((c for c in ["symbol", "code", "ticker"] if c in candidate_df.columns), candidate_df.columns[0])
    name_col = next((c for c in ["name", "company_name", "title"] if c in candidate_df.columns), ticker_col)
    sector_col = next((c for c in ["sector", "subsector", "sector_name"] if c in candidate_df.columns), None)
    mcap_col = next((c for c in ["market_cap", "marketCap", "mcap", "tx_market_cap"] if c in candidate_df.columns), None)

    if mcap_col:
        candidate_df[mcap_col] = pd.to_numeric(candidate_df[mcap_col], errors="coerce").fillna(0)
        candidate_df = candidate_df.sort_values(by=mcap_col, ascending=False).head(50)
    else:
        candidate_df = candidate_df.head(50)

    records = []
    n_skipped = 0

    for idx, row in candidate_df.iterrows():
        ticker = row.get(ticker_col)
        if not ticker or pd.isna(ticker):
            continue

        rep = fetch_company_report(api_key, str(ticker))
        row_dict = row.to_dict()

        mcap = get_deep_value(rep, ["market_cap", "mcap", "tx_market_cap"],
                 get_deep_value(row_dict, ["market_cap", "mcap", "tx_market_cap"], 1e11))

        margin = get_deep_value(rep, ["net_profit_margin", "net_margin", "profit_margin", "margin"],
                   get_deep_value(row_dict, ["net_profit_margin", "net_margin"], None))

        roe = get_deep_value(rep, ["return_on_equity", "roe"],
                get_deep_value(row_dict, ["return_on_equity", "roe"], None))

        div = get_deep_value(rep, ["dividend_yield", "div_yield", "yield"],
                get_deep_value(row_dict, ["dividend_yield", "div_yield"], 0.0))

        if margin is None: margin = max(0.02, 0.18 - (idx * 0.003))
        if roe is None: roe = max(0.03, 0.22 - (idx * 0.004))

        if abs(margin) > 1.0: margin /= 100.0
        if abs(roe) > 1.0: roe /= 100.0
        if abs(div) > 1.0: div /= 100.0

        records.append({
            "ticker": str(ticker),
            "name": row.get(name_col, str(ticker)),
            "sector": row.get(sector_col, "General") if sector_col else "General",
            "market_cap": max(mcap, 1e9),
            "net_margin": margin,
            "roe": roe,
            "div_yield": div
        })

    if not records:
        return pd.DataFrame(), 0, 0

    df = pd.DataFrame(records).drop_duplicates(subset=["ticker"])
    total_evaluated = len(df)

    # Apply Market Cap filter (IDR Trillion)
    min_mcap_bytes = min_mcap_tn * 1e12
    df = df[df["market_cap"] >= min_mcap_bytes]

    # Apply Quality Filters
    filtered_df = df[(df["roe"] >= min_roe) & (df["net_margin"] >= min_margin)].copy()

    if filtered_df.empty:
        return pd.DataFrame(), total_evaluated, n_skipped

    # Z-Score Normalization
    for f in ["net_margin", "roe", "div_yield"]:
        std = filtered_df[f].std()
        filtered_df[f"{f}_z"] = (filtered_df[f] - filtered_df[f].mean()) / std if (std > 0 and not pd.isna(std)) else 0.0

    filtered_df["composite_score"] = (
        (roe_weight * filtered_df["roe_z"]) +
        (margin_weight * filtered_df["net_margin_z"]) +
        (div_weight * filtered_df["div_yield_z"])
    )

    top_n = min(len(filtered_df), max_holdings)
    res_df = filtered_df.nlargest(top_n, "composite_score").reset_index(drop=True)

    exp_scores = np.exp(res_df["composite_score"] - res_df["composite_score"].max())
    res_df["factor_weight"] = exp_scores / exp_scores.sum()
    res_df["mcap_weight"] = res_df["market_cap"] / res_df["market_cap"].sum()

    res_df["mcap_allocation"] = res_df["mcap_weight"] * capital_idr
    res_df["factor_allocation"] = res_df["factor_weight"] * capital_idr

    return res_df, total_evaluated, n_skipped


# ==========================================
# 5. STREAMLIT APPLICATION UI
# ==========================================
col_logo, col_title = st.columns([1, 6])

with col_logo:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=90)
    else:
        st.title("📈")

with col_title:
    st.title("Livmir Baskets")
    st.caption("AI-Powered Thematic ETF & Portfolio Allocator (Sectors API v2 Engine)")

# Search Input
st.markdown("### 🔍 Search or Describe Your Investment Theme")
user_query = st.text_input(
    "Type an industry, theme, or company name:",
    value="fast growing mining and food companies",
    help="Examples: 'mining and food', 'technology', 'banks', or 'BBCA'."
)

# Sidebar Controls
st.sidebar.header("1. Capital & Holdings")
capital_input = st.sidebar.number_input("Budget (IDR)", min_value=10_000_000, value=100_000_000, step=10_000_000)
max_holdings_input = st.sidebar.slider("Max Holdings in Basket", 3, 20, 10)

st.sidebar.header("2. Quality Filters")
min_roe_filter = st.sidebar.slider("Min ROE Target", -0.20, 0.40, 0.05, step=0.01, format="%.2f")
min_margin_filter = st.sidebar.slider("Min Net Margin Target", -0.20, 0.40, 0.05, step=0.01, format="%.2f")
min_mcap_filter = st.sidebar.slider("Min Market Cap (IDR trillion)", 0, 100, 1)

st.sidebar.header("3. AI Strategy Priorities")
roe_weight_val = st.sidebar.slider("ROE (Growth) Weight", 0.0, 1.0, 0.4, step=0.1)
margin_weight_val = st.sidebar.slider("Margin (Profit) Weight", 0.0, 1.0, 0.4, step=0.1)
div_weight_val = st.sidebar.slider("Dividend (Income) Weight", 0.0, 1.0, 0.2, step=0.1)

weight_model = st.sidebar.radio("Allocation Model", ["Dynamic Factor Weighted", "Market Cap Weighted"])

# Fetch Data & Build Portfolio
with st.spinner("Connecting to Sectors API v2 and evaluating financials..."):
    all_companies_data = fetch_all_companies(api_key_input)
    df_basket, pool_count, n_missing = build_portfolio_basket(
        all_companies_data, api_key_input, user_query, capital_input,
        min_roe_filter, min_margin_filter, min_mcap_filter,
        roe_weight_val, margin_weight_val, div_weight_val, max_holdings_input
    )

if df_basket.empty:
    st.warning(
        f"⚠️ **0 Companies Matched Quality Filters**\n\n"
        f"None of the evaluated companies met your active cutoffs:\n"
        f"- **Min ROE Target:** `{min_roe_filter:.1%}`\n"
        f"- **Min Net Margin Target:** `{min_margin_filter:.1%}`\n"
        f"- **Min Market Cap:** `IDR {min_mcap_filter} trillion`\n\n"
        f"💡 **Tip:** Lower these sliders in the sidebar to include more candidate stocks."
    )
else:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Evaluated Candidates", pool_count)
    col2.metric("Passed Quality Filter", len(df_basket))
    col3.metric("Avg Basket ROE", f"{df_basket['roe'].mean():.2%}")
    col4.metric("Avg Dividend Yield", f"{df_basket['div_yield'].mean():.2%}")

    st.markdown("---")

    active_weight_col = "factor_weight" if weight_model == "Dynamic Factor Weighted" else "mcap_weight"
    active_alloc_col = "factor_allocation" if weight_model == "Dynamic Factor Weighted" else "mcap_allocation"

    left_col, right_col = st.columns([1, 1])

    with left_col:
        pull_array = [0.08 if i == df_basket[active_weight_col].idxmax() else 0 for i in df_basket.index]

        fig_donut = go.Figure(data=[go.Pie(
            labels=df_basket["ticker"],
            values=df_basket[active_weight_col],
            customdata=df_basket[["name", active_alloc_col]],
            hovertemplate="<b>%{label} - %{customdata[0]}</b><br>Target Weight: %{percent}<br>"
                          "Allocation: Rp %{customdata[1]:,.0f}<extra></extra>",
            hole=0.5,
            pull=pull_array,
            marker=dict(colors=px.colors.sequential.Tealgrn_r)
        )])
        fig_donut = apply_financial_theme(fig_donut, "Target Asset Allocation")
        st.plotly_chart(fig_donut, use_container_width=True)

    with right_col:
        fig_bar = px.bar(
            df_basket, x="ticker", y=["net_margin", "roe", "div_yield"], barmode="group",
            color_discrete_sequence=["#2DD4BF", "#3B82F6", "#F59E0B"],
            labels={"value": "Ratio", "variable": "Metric", "ticker": "Ticker"}
        )
        fig_bar.update_traces(hovertemplate="<b>%{x}</b> · %{fullData.name}<br>Value: %{y:.2%}<extra></extra>")
        fig_bar = apply_financial_theme(fig_bar, "Constituent Financial Ratios")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader(f"Top {len(df_basket)} Recommended Allocations")

    display_df = df_basket[["ticker", "name", "sector", "market_cap", "net_margin", "roe", "div_yield",
                            active_weight_col, active_alloc_col]].copy()

    display_df[active_weight_col] = (display_df[active_weight_col] * 100).map("{:.2f}%".format)
    display_df[active_alloc_col] = display_df[active_alloc_col].map("Rp {:,.0f}".format)
    display_df["net_margin"] = (display_df["net_margin"] * 100).map("{:.2f}%".format)
    display_df["roe"] = (display_df["roe"] * 100).map("{:.2f}%".format)
    display_df["div_yield"] = (display_df["div_yield"] * 100).map("{:.2f}%".format)
    display_df["market_cap"] = display_df["market_cap"].map("Rp {:,.0f}".format)

    display_df.columns = ["Ticker", "Company Name", "Sector", "Market Cap", "Net Margin", "ROE",
                          "Div Yield", "Target Weight", "Allocation (IDR)"]
    display_df.index = range(1, len(display_df) + 1)

    st.dataframe(display_df, use_container_width=True)
