import os
import re
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

# ==========================================
# 1. PAGE CONFIGURATION & THEME
# ==========================================
APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "logo.png"
try:
    PAGE_ICON = Image.open(LOGO_PATH)   # your logo as the browser-tab icon
except Exception:
    PAGE_ICON = "📈"                    # fallback if logo.png is missing

st.set_page_config(
    page_title="Livmir Baskets | Smart AI ETF Engine (API v2)",
    page_icon=PAGE_ICON,
    layout="wide"
)

SECTORS_API_BASE_URL = "https://api.sectors.app/v2"
_today = date.today()
# Latest fully reported fiscal year (reports for year N usually land by ~April N+1)
DEFAULT_FY = _today.year - 1 if _today.month >= 5 else _today.year - 2


def apply_financial_theme(fig, title=""):
    """Applies dark financial styling to Plotly charts with clean top margins to prevent title overlap."""
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=16, color="#FFFFFF"),
                   x=0.0, y=0.98, xanchor="left", yanchor="top"),
        template="plotly_dark",
        paper_bgcolor="rgba(15, 17, 23, 0)",
        plot_bgcolor="rgba(15, 17, 23, 0)",
        font=dict(family="Inter, sans-serif", size=12, color="#B0B3C6"),
        margin=dict(l=20, r=20, t=90, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=11, color="#FFFFFF")),
        hoverlabel=dict(bgcolor="#1E2230", font_size=12, font_family="Inter", font_color="#FFFFFF"),
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

api_key_input = st.sidebar.text_input(
    "Sectors API Key",
    type="password",
    value=os.getenv("SECTORS_API_KEY", ""),
    help="Enter your Sectors API key."
)

if not api_key_input:
    st.error("⛔ **Sectors API Key Required**")
    st.warning("This application requires an active Sectors API Key to function. Please provide a key in the sidebar.")
    st.stop()


# ==========================================
# 3. SECTORS API LAYER (v2)
# ==========================================
class SectorsError(Exception):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def sectors_get(api_key: str, path: str, params: dict | None = None, retries: int = 3):
    r = None
    for attempt in range(retries):
        r = requests.get(f"{SECTORS_API_BASE_URL}{path}", params=params, timeout=20,
                         headers={"Authorization": api_key})
        if r.status_code == 429 and attempt < retries - 1:
            time.sleep(1.5 * (attempt + 1))
            continue
        break
    if r.status_code in (401, 403):
        raise SectorsError("Invalid or unauthorized API key (the API needs a Standard or Professional plan).", r.status_code)
    if r.status_code == 429:
        raise SectorsError("Rate limit or credit quota reached (HTTP 429).", 429)
    if r.status_code != 200:
        try:
            body = r.json()
            detail = body.get("message") or body.get("error") or r.text[:200]
        except Exception:
            detail = r.text[:200]
        raise SectorsError(f"HTTP {r.status_code} on {path}: {detail}", r.status_code)
    return r.json()


def to_float(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def to_ratio(x):
    """Normalise percentages (e.g. 18.5) into ratios (0.185)."""
    v = to_float(x)
    return v / 100.0 if (not np.isnan(v) and abs(v) > 1.5) else v


def clean_ticker(sym: str) -> str:
    return str(sym).replace(".JK", "").replace(".jk", "").strip().upper()


def qv_get(qv, field: str):
    """Read a field out of a screener row's query_values (tolerates 'field[2025]' style keys)."""
    if not isinstance(qv, dict):
        return None
    for k, v in qv.items():
        k = str(k).lower()
        if k == field or k.startswith(field + "["):
            if isinstance(v, dict):
                vals = [x for x in v.values() if x is not None]
                v = vals[-1] if vals else None
            return v
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def load_subsectors(api_key: str) -> list[dict]:
    data = sectors_get(api_key, "/subsectors/")
    if isinstance(data, dict):
        data = data.get("results") or data.get("data") or []
    return [{"subsector": d["subsector"], "sector": d.get("sector", "")}
            for d in data if isinstance(d, dict) and d.get("subsector")]


def screen(api_key: str, where: str, order_by: str = "-market_cap") -> list[dict]:
    data = sectors_get(api_key, "/companies/", {
        "where": where, "order_by": order_by, "limit": 200, "include_query_values": "true"})
    if isinstance(data, dict):
        return data.get("results") or []
    return data if isinstance(data, list) else []


# ==========================================
# 4. THEME INTERPRETATION (query -> subsectors / tickers / names)
# ==========================================
STOPWORDS = {
    "and", "or", "the", "in", "of", "for", "with", "top", "best", "good", "great", "high", "low",
    "fast", "growing", "growth", "companies", "company", "stock", "stocks", "etf", "basket",
    "portfolio", "indonesia", "indonesian", "idx", "leaders", "leading", "undervalued", "cheap",
    "dividend", "income", "yield", "quality", "profitable", "defensive", "strong", "sector", "sectors",
}

# (words in the user's theme) -> (fragments to look for inside the live sector/subsector slugs)
THEME_HINTS = [
    (("bank", "banks", "banking", "lending", "credit"), ("banks", "financing")),
    (("finance", "financial", "financials", "insurance", "fintech"), ("financ", "insurance", "investment", "banks")),
    (("mining", "miner", "miners", "coal", "metal", "metals", "mineral", "minerals", "gold", "nickel", "copper", "tin"),
     ("oil-gas-coal", "metals", "minerals")),
    (("oil", "gas", "petroleum", "energy"), ("oil-gas", "energy")),
    (("renewable", "solar", "green", "clean"), ("alternative-energy", "utilities")),
    (("food", "beverage", "beverages", "fmcg", "snack", "snacks", "staples", "grocery"), ("food", "beverage", "staples")),
    (("tobacco", "cigarette", "cigarettes"), ("tobacco",)),
    (("agri", "agriculture", "palm", "plantation", "plantations", "farm", "crop", "crops"), ("agricultural",)),
    (("consumer", "household", "apparel", "fashion", "leisure", "retail", "shopping"),
     ("consumer", "household", "apparel", "retail", "leisure")),
    (("tech", "technology", "software", "digital", "internet", "ecommerce", "e-commerce", "startup", "startups", "ai"),
     ("software", "technology", "media")),
    (("telecom", "telecoms", "telecommunication", "cellular", "tower", "towers"), ("telecommunication",)),
    (("health", "healthcare", "pharma", "pharmaceutical", "hospital", "hospitals", "medicine", "medical", "clinic"),
     ("health", "pharma")),
    (("property", "properties", "real estate", "housing", "developer", "developers"), ("properties", "real-estate")),
    (("infrastructure", "infrastructures", "construction", "toll", "cement", "contractor", "contractors", "building"),
     ("infrastructure", "construction", "heavy-construction")),
    (("transport", "transportation", "logistics", "shipping", "airline", "airlines", "port", "ports", "delivery"),
     ("transport", "logistic")),
    (("utility", "utilities", "electricity", "power"), ("utilities",)),
    (("chemical", "chemicals", "plastic", "packaging", "paper", "forestry"), ("chemicals", "containers", "forestry")),
    (("auto", "automotive", "car", "cars", "vehicle", "vehicles", "motor"), ("automobiles",)),
    (("media", "entertainment", "gaming", "game", "games"), ("media", "leisure")),
]


def interpret_query(query: str, taxonomy: list[dict]) -> dict:
    orig = re.findall(r"[A-Za-z0-9\-]+", query)
    lower = [t.lower() for t in orig]
    low_query = query.lower()
    slug_text = {t["subsector"]: f"{t['sector']} {t['subsector']}".lower() for t in taxonomy}
    subs, used = [], set()

    def add_frags(frags):
        for sub, txt in slug_text.items():
            if sub not in subs and any(f in txt for f in frags):
                subs.append(sub)

    for keys, frags in THEME_HINTS:
        hit = False
        for i, tok in enumerate(lower):
            if any(tok == k or (len(k) >= 6 and tok.startswith(k)) for k in keys if " " not in k):
                hit = True
                used.add(i)
        if any(" " in k and k in low_query for k in keys):
            hit = True
        if hit:
            add_frags(frags)

    # any remaining word that appears directly inside a live sector/subsector slug
    for i, tok in enumerate(lower):
        if i in used or len(tok) < 4 or tok in STOPWORDS:
            continue
        matches = [s for s, txt in slug_text.items() if tok in txt]
        if matches:
            used.add(i)
            subs.extend(s for s in matches if s not in subs)

    # tickers: unmatched ALL-CAPS 4-letter words (BBCA), or a lone 4-letter word with no theme match
    tickers = []
    for i, tok in enumerate(orig):
        if i not in used and re.fullmatch(r"[A-Z]{4}", tok):
            tickers.append(tok)
            used.add(i)
    if not subs and not tickers and len(orig) == 1 and re.fullmatch(r"[A-Za-z]{4}", orig[0]):
        tickers.append(orig[0].upper())
        used.add(0)

    # leftover words -> company-name search
    names = [tok for i, tok in enumerate(lower) if i not in used and len(tok) >= 4 and tok not in STOPWORDS]
    return {"subsectors": subs, "tickers": tickers[:5], "names": names[:2]}


def build_sources(subs, tickers, names) -> tuple:
    """Each source = (label, (alternative WHERE conditions))."""
    src = []
    for s in subs:
        src.append((s.replace("-", " ").title(),
                    (f"sub_sector = '{s}'", f"sub_sector like '%{s.replace('-', ' ')}%'")))
    for t in tickers:
        src.append((f"Ticker {t}", (f"symbol like '{t}%'",)))
    for n in names:
        src.append((f"Name: {n}", (f"company_name like '%{n}%'",)))
    return tuple(src)


# ==========================================
# 5. UNIVERSE BUILDER (Sectors screener = primary, per-company reports = fallback)
# ==========================================
def fetch_source(api_key: str, label: str, conds: tuple, fy: int):
    warns = []
    core = "market_cap > 0 and roe_ttm > -1000"
    cond, rows = None, []
    for c in conds:
        rows = screen(api_key, f"{c} and {core}")
        if rows:
            cond = c
            break
    if not rows:
        return pd.DataFrame(), [f"No companies found for '{label}'."]

    recs = {}

    def absorb(rs, fields, create=False):
        for pos, r in enumerate(rs):
            sym = r.get("symbol")
            if not sym:
                continue
            t = clean_ticker(sym)
            if t not in recs:
                if not create:
                    continue
                recs[t] = {"ticker": t, "name": r.get("company_name") or t, "sector": label, "rank": pos}
            qv = r.get("query_values")
            for f in fields:
                v = qv_get(qv, f)
                if v is not None:
                    recs[t][f] = v

    def safe_screen(where):
        try:
            return screen(api_key, where)
        except SectorsError as e:
            if e.status in (401, 403, 429):
                raise
            warns.append(f"{label}: a query failed ({e}).")
            return None

    absorb(rows, ["market_cap", "roe_ttm"], create=True)

    rs = safe_screen(f"{cond} and market_cap > 0 and yield_ttm >= 0")
    if rs:
        absorb(rs, ["yield_ttm"])

    margin_ok = False
    for yr in (fy, fy - 1):
        rs = safe_screen(f"{cond} and market_cap > 0 and net_profit_margin[{yr}] > -1000")
        if rs is None:
            break
        if rs:
            absorb(rs, ["net_profit_margin"])
            margin_ok = True
            break
    if not margin_ok:
        rs = safe_screen(f"{cond} and market_cap > 0 and total_revenue_mrq > 0 and earnings_mrq > -999999999999999")
        if rs:
            absorb(rs, ["total_revenue_mrq", "earnings_mrq"])
            warns.append(f"{label}: annual net margin unavailable, used latest-quarter margin instead.")

    d = pd.DataFrame(list(recs.values()))
    for c in ["market_cap", "roe_ttm", "yield_ttm", "net_profit_margin", "total_revenue_mrq", "earnings_mrq"]:
        if c not in d:
            d[c] = np.nan
    rev = d["total_revenue_mrq"].map(to_float)
    earn = d["earnings_mrq"].map(to_float)
    margin = d["net_profit_margin"].map(to_ratio).fillna(earn / rev.where(rev > 0))
    out = pd.DataFrame({
        "ticker": d["ticker"], "name": d["name"], "sector": d["sector"], "rank": d["rank"],
        "market_cap": d["market_cap"].map(to_float),
        "roe": d["roe_ttm"].map(to_ratio),
        "div_yield": d["yield_ttm"].map(to_ratio),
        "net_margin": margin,
    })
    return out, warns


def report_metrics(api_key: str, ticker: str) -> dict:
    """Fallback only: 3 credits per company (overview + financials + dividend)."""
    try:
        rep = sectors_get(api_key, f"/company/report/{ticker}/",
                          {"sections": "overview,financials,dividend"})
    except SectorsError as e:
        if e.status == 404:
            return {}
        raise
    ov = rep.get("overview") or {}
    ratios = (rep.get("financials") or {}).get("historical_financial_ratio") or []
    ratios = sorted([x for x in ratios if isinstance(x, dict)], key=lambda x: str(x.get("year")), reverse=True)
    roe = margin = np.nan
    for x in ratios:
        p = x.get("profitability") or {}
        if p.get("roe") is not None:
            roe, margin = to_ratio(p.get("roe")), to_ratio(p.get("net_profit_margin"))
            break
    return {"market_cap": to_float(ov.get("market_cap")), "roe": roe, "net_margin": margin,
            "div_yield": to_ratio((rep.get("dividend") or {}).get("yield_ttm"))}


@st.cache_data(ttl=3600, show_spinner=False)
def build_universe(api_key: str, sources: tuple, fy: int):
    with ThreadPoolExecutor(max_workers=min(6, max(1, len(sources)))) as pool:
        results = list(pool.map(lambda s: fetch_source(api_key, s[0], s[1], fy), sources))
    warnings = [w for _, ws in results for w in ws]
    frames = [df for df, _ in results if not df.empty]
    meta = {"mode": "screener", "warnings": warnings}
    if not frames:
        return pd.DataFrame(), meta

    df = pd.concat(frames, ignore_index=True).drop_duplicates("ticker").reset_index(drop=True)

    if df["roe"].notna().sum() == 0:   # screener gave symbols but no metric values
        meta["mode"] = "reports"
        warnings.append("The screener returned no metric values, so metrics were pulled from per-company reports "
                        "(3 credits each, limited to the 12 largest companies per source).")
        df = df.sort_values(["sector", "rank"]).groupby("sector", group_keys=False).head(12).reset_index(drop=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            mets = list(pool.map(lambda t: report_metrics(api_key, t), df["ticker"]))
        m = pd.DataFrame(mets, index=df.index)
        for col in ["market_cap", "roe", "net_margin", "div_yield"]:
            if col in m:
                df[col] = df[col].where(df[col].notna(), m[col])

    df = df.dropna(subset=["market_cap"])
    df = df[df["market_cap"] > 0]
    df["div_yield"] = df["div_yield"].fillna(0.0)
    return df.reset_index(drop=True), meta


# ==========================================
# 6. PORTFOLIO BASKET BUILDER
# ==========================================
def zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    if not std or pd.isna(std):
        return pd.Series(0.0, index=s.index)
    return ((s - s.mean()) / std).clip(-3, 3)   # winsorised so outliers can't dominate


def build_portfolio_basket(universe: pd.DataFrame, capital_idr: float, min_roe: float, min_margin: float,
                           min_mcap_tn: float, roe_weight: float, margin_weight: float,
                           div_weight: float, max_holdings: int):
    df = universe.copy()
    missing = df[["roe", "net_margin"]].isna().any(axis=1)
    n_missing = int(missing.sum())
    df = df[~missing]
    df = df[df["market_cap"] >= min_mcap_tn * 1e12]
    pool_count = len(df)

    df = df[(df["roe"] >= min_roe) & (df["net_margin"] >= min_margin)].copy()
    if df.empty:
        return pd.DataFrame(), pool_count, n_missing

    df["roe_z"], df["net_margin_z"], df["div_yield_z"] = zscore(df["roe"]), zscore(df["net_margin"]), zscore(df["div_yield"])
    df["composite_score"] = (roe_weight * df["roe_z"] + margin_weight * df["net_margin_z"]
                             + div_weight * df["div_yield_z"])

    res = df.nlargest(min(len(df), max_holdings), "composite_score").reset_index(drop=True)

    exp_scores = np.exp(res["composite_score"] - res["composite_score"].max())
    res["factor_weight"] = exp_scores / exp_scores.sum()
    res["mcap_weight"] = res["market_cap"] / res["market_cap"].sum()
    res["mcap_allocation"] = res["mcap_weight"] * capital_idr
    res["factor_allocation"] = res["factor_weight"] * capital_idr
    return res, pool_count, n_missing


# ==========================================
# 7. STREAMLIT APPLICATION UI
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

st.markdown("### 🔍 Search or Describe Your Investment Theme")
user_query = st.text_input(
    "Type an industry, theme, or company name:",
    value="fast growing mining and food companies",
    help="Examples: 'mining and food', 'technology', 'banks', 'BBCA', 'telkom'."
)

# Sector taxonomy (cached 24h, 1 credit)
try:
    with st.spinner("Connecting to Sectors API v2..."):
        taxonomy = load_subsectors(api_key_input)
except SectorsError as e:
    st.error(f"❌ {e}")
    st.stop()
except requests.exceptions.RequestException as e:
    st.error(f"🌐 **Network Error**: Could not connect to Sectors API ({e})")
    st.stop()

interp = interpret_query(user_query, taxonomy)
all_sub_names = [t["subsector"] for t in taxonomy]
chosen_subs = st.multiselect(
    "Subsectors being sourced (edit if the theme matcher missed something)",
    all_sub_names, default=interp["subsectors"]
)

parts = []
if chosen_subs:
    parts.append(", ".join(s.replace("-", " ").title() for s in chosen_subs))
if interp["tickers"]:
    parts.append("tickers " + ", ".join(interp["tickers"]))
if interp["names"]:
    parts.append("company names matching " + ", ".join(f"'{n}'" for n in interp["names"]))

sources = build_sources(chosen_subs, interp["tickers"], interp["names"])
if not sources:
    st.warning("No subsector, ticker or company matched your theme. Pick subsectors above or rephrase "
               "(e.g. 'banks', 'mining', 'technology', 'BBCA').")
    st.stop()

st.info(f"🤖 **AI Theme Matcher:** Sourcing companies from → **{' | '.join(parts)}**", icon="🎯")

# Sidebar controls
st.sidebar.header("1. Capital & Holdings")
capital_input = st.sidebar.number_input("Budget (IDR)", min_value=10_000_000, value=100_000_000, step=10_000_000)
max_holdings_input = st.sidebar.slider("Max Holdings in Basket", 3, 20, 10)

st.sidebar.header("2. Quality Filters")
min_roe_filter = st.sidebar.slider("Min ROE Target", -0.20, 0.40, 0.05, step=0.01, format="%.2f")
min_margin_filter = st.sidebar.slider("Min Net Margin Target", -0.20, 0.40, 0.05, step=0.01, format="%.2f")
min_mcap_filter = st.sidebar.slider("Min Market Cap (IDR trillion)", 0, 100, 1,
                                    help="Filters out illiquid micro-caps whose extreme ratios distort scoring.")

st.sidebar.header("3. AI Strategy Priorities")
roe_weight_val = st.sidebar.slider("ROE (Growth) Weight", 0.0, 1.0, 0.4, step=0.1)
margin_weight_val = st.sidebar.slider("Margin (Profit) Weight", 0.0, 1.0, 0.4, step=0.1)
div_weight_val = st.sidebar.slider("Dividend (Income) Weight", 0.0, 1.0, 0.2, step=0.1)

weight_model = st.sidebar.radio("Allocation Model", ["Dynamic Factor Weighted", "Market Cap Weighted"])

# Fetch universe (cached 1h: moving sliders never re-hits the API)
st.caption(f"Data fetch uses about {len(sources) * 3} Sectors API credits per new theme; repeats within an hour are cached.")
try:
    with st.spinner("Pulling fundamentals from Sectors API v2..."):
        universe, meta = build_universe(api_key_input, sources, DEFAULT_FY)
except SectorsError as e:
    st.error(f"❌ {e}")
    st.stop()
except requests.exceptions.RequestException as e:
    st.error(f"🌐 **Network Error**: {e}")
    st.stop()

if meta["warnings"]:
    with st.expander(f"ℹ️ Data notes ({len(meta['warnings'])})"):
        for w in meta["warnings"]:
            st.write("• " + w)

if universe.empty:
    st.warning("The Sectors API returned no usable company data for this theme. Try different subsectors.")
    st.stop()

df_basket, pool_count, n_missing = build_portfolio_basket(
    universe, capital_input, min_roe_filter, min_margin_filter, min_mcap_filter,
    roe_weight_val, margin_weight_val, div_weight_val, max_holdings_input
)

if df_basket.empty:
    st.warning(
        f"⚠️ **0 Companies Matched Quality Filters**\n\n"
        f"None of the {pool_count} evaluated companies met your active cutoffs:\n"
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
    if n_missing:
        st.caption(f"{n_missing} companies were skipped because Sectors has no ROE / net margin for them.")

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

# Data inspector: verify what Sectors actually returned
with st.expander("🛠 Data inspector (verify the values coming from Sectors)"):
    st.caption(f"Mode: **{meta['mode']}** · {len(universe)} companies loaded · "
               f"{int((universe['div_yield'] > 0).sum())} with a non-zero dividend yield.")
    st.dataframe(universe.drop(columns=["rank"], errors="ignore"), use_container_width=True)
    if st.button("Show raw screener response for the first source (1 credit)"):
        label0, conds0 = sources[0]
        st.json(screen(api_key_input, f"{conds0[0]} and market_cap > 0 and yield_ttm >= 0")[:3])
