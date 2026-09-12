"""
Equity — Risk & Momentum Analyzer (Streamlit)

Add any ticker or company you like; the EasyEquities list loads by default.
The app scores every selected name, suggests a portfolio allocation, and draws
the supporting charts — all designed to stay readable across a 100+ name list.
"""
from __future__ import annotations

import time
from typing import List

import numpy as np
import pandas as pd
import streamlit as st

import core
import charts
from core import BENCHMARK_TICKER, DEFAULT_TICKERS

st.set_page_config(page_title="Equity — Risk & Momentum Analyzer",
                   page_icon="📊", layout="wide")

# --- light styling ---------------------------------------------------------
st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1500px;}
  h1, h2, h3 {letter-spacing: -0.01em;}
  div[data-testid="stMetricValue"] {font-size: 1.4rem;}
  .stTabs [data-baseweb="tab-list"] {gap: 4px;}
  .stTabs [data-baseweb="tab"] {padding: 8px 16px;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Ticker parsing
# ---------------------------------------------------------------------------
def parse_universe(text: str) -> List[str]:
    """Accept commas, newlines, spaces, or tabs. Uppercase, dedupe, keep order.
    A pasted company name is left as-is (Yahoo resolves many)."""
    if not text:
        return []
    raw = text.replace(",", "\n").replace("\t", "\n")
    parts = []
    for line in raw.split("\n"):
        tok = line.strip().upper()
        if tok:
            parts.append(tok)
    return list(dict.fromkeys(parts))


@st.cache_data(show_spinner=False, ttl=core.PRICE_TTL)
def _run_pipeline(tickers_key: str, period: str, risk_free: float,
                  max_weight: float, zero_out: bool, force: bool):
    """Cached end-to-end run. tickers_key is the sorted comma list so the same
    universe reuses results. `force` busts by being part of the args."""
    tickers = tickers_key.split(",")
    requested = list(dict.fromkeys(tickers))
    if BENCHMARK_TICKER not in requested:
        requested = requested + [BENCHMARK_TICKER]

    price = core.fetch_price_data(requested, period, force=force)
    if price.empty or len(price) < 2:
        return {"error": (
            f"No price data came back for these tickers over {period}. "
            "Check the symbols are valid on Yahoo Finance, or hit Refresh data "
            "in a minute if Yahoo is throttling.")}

    live = list(price.columns)
    fundamentals = core.fetch_fundamentals(live, force=force)
    profiles = core.resolve_profiles(live, force=force)
    metrics = core.calculate_metrics(price, fundamentals, risk_free_rate=risk_free / 100.0)
    if metrics.empty:
        return {"error": "Prices loaded, but no ticker had enough history to score."}

    caps = (pd.to_numeric(fundamentals["Market Cap ($B)"], errors="coerce")
            if "Market Cap ($B)" in fundamentals.columns else None)
    allocation = core.compute_allocation(metrics, max_weight=max_weight / 100.0,
                                          zero_out=zero_out)

    return {
        "price": price, "fundamentals": fundamentals, "profiles": profiles,
        "metrics": metrics, "caps": caps, "allocation": allocation,
        "requested": requested, "live": live,
    }


# ---------------------------------------------------------------------------
# Sidebar — universe + assumptions
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Universe")
    st.caption("Pick from the default list, or add your own tickers. "
               "Benchmark **SPY** is always included.")

    # Custom tickers the user has added beyond the default list live here so
    # they survive selection changes and appear as options too.
    if "custom_tickers" not in st.session_state:
        st.session_state.custom_tickers = []
    if "selected_tickers" not in st.session_state:
        st.session_state.selected_tickers = list(DEFAULT_TICKERS)

    # --- add your own -------------------------------------------------------
    add_box = st.text_input("Add tickers", placeholder="e.g. TSLA, PLTR, ASML",
                            help="Type one or more symbols (or paste a list) and press Add. "
                                 "They join the selection and the option list below.")
    c_add, c_reset = st.columns(2)
    if c_add.button("➕ Add", use_container_width=True) and add_box.strip():
        new = parse_universe(add_box)
        # register as custom options and select them
        for t in new:
            if t not in DEFAULT_TICKERS and t not in st.session_state.custom_tickers:
                st.session_state.custom_tickers.append(t)
        merged = list(dict.fromkeys(st.session_state.selected_tickers + new))
        st.session_state.selected_tickers = merged
        st.rerun()
    if c_reset.button("↺ Reset", use_container_width=True,
                      help="Reselect the full default list and clear custom additions."):
        st.session_state.custom_tickers = []
        st.session_state.selected_tickers = list(DEFAULT_TICKERS)
        st.rerun()

    # --- the selectable universe -------------------------------------------
    options = list(dict.fromkeys(list(DEFAULT_TICKERS) + st.session_state.custom_tickers))
    # keep any previously-selected tickers valid as options
    for t in st.session_state.selected_tickers:
        if t not in options:
            options.append(t)

    c_all, c_none = st.columns(2)
    if c_all.button("Select all", use_container_width=True):
        st.session_state.selected_tickers = list(options)
        st.rerun()
    if c_none.button("Clear all", use_container_width=True):
        st.session_state.selected_tickers = []
        st.rerun()

    selected = st.multiselect(
        "Tickers", options=options,
        default=None, key="selected_tickers",
        help="The default EasyEquities list is pre-selected. Deselect any you "
             "don't want, or add more above.")

    tickers = list(dict.fromkeys(selected))
    st.caption(f"**{len(tickers)}** of {len(options)} selected · benchmark **SPY** added automatically.")

    st.divider()
    st.header("Assumptions")
    period = st.selectbox("Look-back period", core.PERIOD_CHOICES, index=4)  # 1y
    price_view = st.selectbox("Price chart view", core.PRICE_VIEW_CHOICES, index=0)
    risk_free = st.number_input("Risk-free rate (% annual)", value=core.DEFAULT_RISK_FREE,
                                step=0.25, format="%.2f",
                                help="Subtracted from returns before computing Sharpe.")
    max_weight = st.slider("Max position size (%)", 5, 100, int(core.DEFAULT_MAX_WEIGHT),
                           step=5, help="Caps concentration; the excess is redistributed pro-rata.")
    zero_out = not st.checkbox(
        "Allocate to every scored stock", value=True,
        help="On: every stock that could be scored gets a weight (best names "
             "biggest, weakest smallest) — the allocation chart covers the whole "
             "list. Off: only the strongest names are held; small and thin-history "
             "positions drop to zero for a tighter book.")

    st.divider()
    force = st.checkbox("Refresh data (bypass cache)", value=False,
                        help=f"Prices cache {core.PRICE_TTL // 60} min, "
                             f"fundamentals {core.FUNDAMENTAL_TTL // 3600} h.")
    if st.button("🧹 Clear cache", use_container_width=True):
        n = core.clear_caches()
        st.cache_data.clear()
        st.success(f"Cleared {n} cached file(s).")

    run = st.button("Analyse", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Equity — Risk & Momentum Analyzer")
st.markdown(
    "Score any set of stocks on risk-adjusted return and momentum, get a "
    "suggested portfolio allocation, and see the supporting charts. "
    "<span style='color:#8a94a3'>Not investment advice — a screening tool on free, best-effort data.</span>",
    unsafe_allow_html=True)

# Trigger a run on first load or when the button is pressed.
if "has_run" not in st.session_state:
    st.session_state.has_run = False
if run:
    st.session_state.has_run = True

if not st.session_state.has_run:
    st.info("Set your universe in the sidebar (the EasyEquities list is loaded by "
            "default), then press **Analyse**.")
    st.stop()

if not tickers:
    st.warning("No tickers selected — add at least one in the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
tickers_key = ",".join(sorted(dict.fromkeys(tickers)))
if force:
    # Bypass both Streamlit's result cache and the on-disk data cache.
    _run_pipeline.clear()
t0 = time.perf_counter()
with st.spinner(f"Fetching and scoring {len(tickers)} tickers over {period}…"):
    result = _run_pipeline(tickers_key, period, risk_free, max_weight, zero_out, force)
elapsed = time.perf_counter() - t0

if "error" in result:
    st.error(result["error"])
    st.stop()

price = result["price"]
metrics = result["metrics"]
fundamentals = result["fundamentals"]
profiles = result["profiles"]
caps = result["caps"]
allocation = result["allocation"]
requested = result["requested"]
live = result["live"]
names = core.name_map(profiles)

# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------
held = allocation[allocation > 0]
body = metrics.drop(index=[BENCHMARK_TICKER], errors="ignore")
best_ret = body["Total Return (%)"].idxmax() if not body.empty else None
top_pos = held.idxmax() if not held.empty else None

m1, m2, m3, m4 = st.columns(4)
m1.metric("Scored", f"{len(body)}", help="Tickers with enough history to score.")
m2.metric("In portfolio", f"{len(held)}",
          help="Names that passed coverage and got a non-zero weight.")
if top_pos is not None:
    m3.metric("Largest position", names.get(top_pos, top_pos), f"{held.max():.1f}%")
if best_ret is not None:
    m4.metric("Best return", names.get(best_ret, best_ret),
              f"{body.loc[best_ret, 'Total Return (%)']:+.0f}%")

# ---------------------------------------------------------------------------
# Data notes
# ---------------------------------------------------------------------------
notes = []
missing = [t for t in requested if t not in live]
if missing:
    notes.append(f"No price history for **{', '.join(missing)}** — symbol may be "
                 "delisted, pre-IPO, or spelled differently on Yahoo.")
late = [t for t in core.late_starters(price) if t != BENCHMARK_TICKER]
if late:
    notes.append(f"Partial history: **{', '.join(names.get(t, t) for t in late)}** — "
                 "their total return covers a shorter window; use *Excess vs Bench*.")
held_set = set(allocation[allocation > 0].index)
skipped = [t for t in metrics.index if t != BENCHMARK_TICKER and t not in held_set]
if skipped:
    if zero_out:
        notes.append(f"Excluded from allocation: **{', '.join(names.get(t, t) for t in skipped)}** "
                     f"(under {core.MIN_COVERAGE:.0f}% of benchmark sessions or missing risk metrics).")
    else:
        notes.append(f"Given only a floor weight (thin history / missing risk metrics): "
                     f"**{', '.join(names.get(t, t) for t in skipped) or '—'}**.")
pe_missing = [t for t in metrics.index if t != BENCHMARK_TICKER
              and (pd.isna(metrics.loc[t, "P/E Ratio"]) or metrics.loc[t, "P/E Ratio"] <= 0)]
if pe_missing:
    notes.append(f"P/E unresolved for **{', '.join(names.get(t, t) for t in pe_missing)}** — "
                 "usually negative earnings or Yahoo rate-limiting (retry in ~60s).")

if notes:
    with st.expander(f"⚠️ Data notes ({len(notes)})", expanded=False):
        for n in notes:
            st.markdown(f"- {n}")
st.caption(f"⏱ {elapsed:.1f}s · {len(live)} priced · Sharpe uses a {risk_free:.2f}% risk-free rate · "
           "cached runs are near-instant.")

# ===========================================================================
# The flow mirrors the original app: supporting / decision charts first, then
# the portfolio allocation as the culmination at the bottom.
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. Price performance
# ---------------------------------------------------------------------------
st.subheader("Price performance")
st.caption("Every name in one muted line; the benchmark and your largest allocations are "
           "highlighted. Hover any line to identify it, or box-zoom into a cluster.")
hl = held.sort_values(ascending=False).head(8).index.tolist() if not held.empty else []
perf = charts.price_performance(price, price_view, names, highlight=hl)
if perf is not None:
    st.plotly_chart(perf, use_container_width=True,
                    config={"displayModeBar": True, "displaylogo": False})

# ---------------------------------------------------------------------------
# 2. Decision / ranking charts  (P/E, momentum, volatility, drawdown, Sharpe,
#    total return) — same set the original app surfaced.
# ---------------------------------------------------------------------------
st.subheader("Decision charts")
st.caption("Each name on its own row, sorted, with the SPY benchmark marked. "
           "Long lists make these tall — scroll within a tab.")

tabs = st.tabs(["Valuation (P/E)", "Momentum (SMA)", "Volatility", "Max drawdown",
                "Sharpe", "Total return"])
chart_fns = [
    charts.pe_bars, charts.sma_bars, charts.volatility_bars,
    charts.drawdown_bars, charts.sharpe_bars, charts.total_return_bars,
]
for tab, fn in zip(tabs, chart_fns):
    with tab:
        fig = fn(metrics, names)
        if fig is None:
            st.info("Not enough data for this chart.")
        else:
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# 3. Risk vs return scatter
# ---------------------------------------------------------------------------
st.subheader("Risk vs return")
scatter = charts.risk_return_scatter(metrics, names, market_caps=caps, allocation=allocation)
if scatter is not None:
    st.plotly_chart(scatter, use_container_width=True,
                    config={"displayModeBar": True, "displaylogo": False})

# ---------------------------------------------------------------------------
# 4. Suggested portfolio allocation — the culmination
# ---------------------------------------------------------------------------
st.subheader("Suggested portfolio allocation")
if allocation.empty or held.empty:
    st.warning("No positions cleared the filters for this universe/period.")
else:
    st.caption(f"Every scored stock gets a weight — best names biggest — across all "
               f"**{len(held)}** positions." if not zero_out else
               f"Tighter book: only the strongest **{len(held)}** names are held.")
    st.plotly_chart(charts.allocation_bars(allocation, names),
                    use_container_width=True, config={"displayModeBar": False})
    st.plotly_chart(charts.allocation_donut(allocation, names),
                    use_container_width=True, config={"displayModeBar": False})

    alloc_table = (held.sort_values(ascending=False).rename("Weight (%)")
                   .reset_index().rename(columns={"index": "Ticker"}))
    alloc_table["Company"] = alloc_table["Ticker"].map(names).fillna(alloc_table["Ticker"])
    alloc_table = alloc_table[["Ticker", "Company", "Weight (%)"]]
    st.download_button("⬇ Download allocation (CSV)",
                       alloc_table.to_csv(index=False).encode(),
                       file_name="suggested_allocation.csv", mime="text/csv")

# ---------------------------------------------------------------------------
# Detailed table
# ---------------------------------------------------------------------------
st.subheader("Detailed metrics")

display = metrics.reset_index().rename(columns={"index": "Ticker"})
display["Company"] = display["Ticker"].map(names).fillna(display["Ticker"])
if "Country" in profiles.columns:
    display["Country"] = display["Ticker"].map(profiles["Country"].to_dict()).fillna("—")
if caps is not None:
    display["Market Cap ($B)"] = pd.to_numeric(
        display["Ticker"].map(caps.to_dict()), errors="coerce").round(1)
if "Currency" in fundamentals.columns:
    display["Currency"] = display["Ticker"].map(fundamentals["Currency"].to_dict()).fillna("—")
display["Allocation (%)"] = display["Ticker"].map(allocation.to_dict()).fillna(0.0)
display["P/E Ratio"] = pd.to_numeric(display["P/E Ratio"], errors="coerce")
display.loc[display["P/E Ratio"] <= 0, "P/E Ratio"] = np.nan

lead = ["Ticker", "Company", "Country", "Allocation (%)"]
lead = [c for c in lead if c in display.columns]
display = display[lead + [c for c in display.columns if c not in lead]]

is_bench = display["Ticker"] == BENCHMARK_TICKER
display = pd.concat([
    display[~is_bench].sort_values(["Allocation (%)", "Total Return (%)"],
                                   ascending=False, na_position="last"),
    display[is_bench],
]).reset_index(drop=True)

st.dataframe(
    display, use_container_width=True, hide_index=True, height=560,
    column_config={
        "Allocation (%)": st.column_config.ProgressColumn(
            "Allocation (%)", format="%.1f%%",
            min_value=0.0, max_value=float(max(allocation.max() if not allocation.empty else 1, 1))),
        "Total Return (%)": st.column_config.NumberColumn(format="%+.1f%%"),
        "Excess vs Bench (pp)": st.column_config.NumberColumn(format="%+.1f"),
        "Volatility (Ann %)": st.column_config.NumberColumn(format="%.1f%%"),
        "Max Drawdown (%)": st.column_config.NumberColumn(format="%.1f%%"),
        "Sharpe Ratio": st.column_config.NumberColumn(format="%.2f"),
        "Beta": st.column_config.NumberColumn(format="%.2f"),
        "P/E Ratio": st.column_config.NumberColumn(format="%.1f"),
        "Current Price": st.column_config.NumberColumn(format="%.2f"),
        "Coverage (%)": st.column_config.NumberColumn(format="%.0f%%"),
        "Market Cap ($B)": st.column_config.NumberColumn(format="%.1f"),
    })

st.download_button("⬇ Download full metrics (CSV)",
                   display.to_csv(index=False).encode(),
                   file_name="equity_metrics.csv", mime="text/csv")

# ---------------------------------------------------------------------------
# Methodology
# ---------------------------------------------------------------------------
with st.expander("📘 How the allocation is scored"):
    st.markdown("""
**Score = 0.40 × Sharpe + 0.30 × Momentum − 0.12 × Volatility − 0.08 × Max Drawdown − 0.10 × P/E**

| Component | Weight | Logic |
|---|---|---|
| Sharpe ratio | +40% | Reward per unit of risk — the core quality signal |
| Momentum (SMA deviation) | +30% | Price above its moving average = positive trend |
| Volatility | −12% | Penalises erratic names (counts up- and down-moves alike) |
| Max drawdown | −8% | Penalises the worst peak-to-trough loss actually endured |
| P/E ratio | −10% | Penalises expensive names (scored on log P/E) |

Each component is normalized 0→1 after **winsorizing at the 5th/95th percentile**, so a single
extreme name can't flatten the spread for everyone else. Positions below **2%** are zeroed,
anything above your **max position size** is capped with the excess redistributed pro-rata, and
the rest renormalizes to 100%. Missing P/E is imputed with the universe median. Names trading
fewer than **60%** of the benchmark's sessions sit the round out.

Metrics are computed on each ticker's *own* trading sessions (no forward-filling across foreign
market holidays), so volatility and Sharpe aren't distorted for non-US listings.

*Not investment advice.*
""")
