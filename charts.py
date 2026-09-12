"""
Plotly charts for the analyzer.

Design brief: the universe is long (100+ names). Everything here is built so a
long list stays legible.

  - Ranking charts are HORIZONTAL bars, not vertical. With 100 names, vertical
    bars force 45-degree labels that collide; horizontal bars put each label on
    its own row, read left-to-right, and the figure grows in height instead of
    cramping width. Height scales with the number of rows.
  - Bars are always sorted, and the benchmark (SPY) is drawn as a dashed
    reference line across the plot so "beat the market or not" is one glance.
  - The scatter labels only the names worth calling out (best/worst on each
    axis); everything else is a hover point, so 100 markers don't become 100
    overlapping text labels.
  - Interactive (zoom, pan, hover, box-select) because a static PNG of 100 rows
    is unreadable on a laptop screen — the user scrolls the plot, not the page.

Palette is a cool, restrained analyst scheme: a single ink for structure, a
teal/amber divergence for good/bad, muted slate for "no data".
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core import BENCHMARK_TICKER

# --- palette ---------------------------------------------------------------
INK = "#1a2028"          # near-black structural ink (not pure black)
GRID = "#e7e9ee"
POS = "#0e8f7e"          # teal — good / above
NEG = "#d9714e"          # clay-amber — bad / below
NEUTRAL = "#b9bfc9"      # muted slate — missing / zero
BENCH = "#22303f"        # benchmark line
ACCENT = "#3a6ea5"       # allocation / primary blue
BG = "rgba(0,0,0,0)"

FONT = dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif",
            size=13, color=INK)


def _row_height(n: int, per_row: int = 26, floor: int = 320, top_bottom: int = 90) -> int:
    """Height that gives every bar room. Grows with the list."""
    return max(floor, n * per_row + top_bottom)


def _base_layout(title: str, height: int, x_title: str = "", y_title: str = "") -> dict:
    return dict(
        title=dict(text=title, font=dict(size=17, color=INK), x=0, xanchor="left",
                   pad=dict(l=6, b=10)),
        height=height,
        margin=dict(l=8, r=28, t=58, b=44),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=FONT,
        xaxis=dict(title=x_title, gridcolor=GRID, zerolinecolor="#c4c9d2",
                   zerolinewidth=1.4, showline=False),
        yaxis=dict(title=y_title, gridcolor=BG, showline=False,
                   automargin=True, ticklabelposition="outside"),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID,
                        font=dict(color=INK, size=12)),
        showlegend=False,
        bargap=0.28,
    )


def _sorted_for_bars(series: pd.Series, names: Dict[str, str],
                     ascending: bool) -> pd.DataFrame:
    """Return a tidy frame (ticker, name, value) sorted, benchmark stripped out
    of the ranking (it's drawn as a reference line instead)."""
    s = pd.to_numeric(series, errors="coerce")
    s = s.drop(index=[BENCHMARK_TICKER], errors="ignore")
    s = s.sort_values(ascending=ascending, na_position="first")
    return pd.DataFrame({
        "ticker": s.index,
        "name": [names.get(t, t) for t in s.index],
        "value": s.values,
    })


def _hbar(frame: pd.DataFrame, title: str, x_title: str,
          color_fn, value_fmt: str = "{:.1f}",
          bench_value: Optional[float] = None,
          bench_label: str = "SPY") -> go.Figure:
    n = len(frame)
    heights = frame["value"].fillna(0).tolist()
    colors = [color_fn(v) for v in frame["value"].tolist()]
    labels = [frame["name"].iloc[i] for i in range(n)]
    text = [("" if pd.isna(v) else value_fmt.format(v)) for v in frame["value"].tolist()]

    fig = go.Figure()
    fig.add_bar(
        x=heights, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=text, textposition="outside",
        textfont=dict(size=11, color=INK),
        cliponaxis=False,
        customdata=frame["ticker"],
        hovertemplate="<b>%{y}</b> (%{customdata})<br>" + x_title + ": %{x:.2f}<extra></extra>",
    )
    layout = _base_layout(title, _row_height(n), x_title=x_title)
    fig.update_layout(layout)

    if bench_value is not None and not pd.isna(bench_value):
        fig.add_vline(x=float(bench_value), line=dict(color=BENCH, width=1.6, dash="dash"))
        fig.add_annotation(x=float(bench_value), y=1.0, yref="paper",
                           text=f"{bench_label} {value_fmt.format(bench_value)}",
                           showarrow=False, font=dict(size=11, color=BENCH),
                           xanchor="left", yanchor="bottom", bgcolor="rgba(255,255,255,0.7)")
    # keep highest at top
    fig.update_yaxes(autorange="reversed")
    return fig


# ---------------------------------------------------------------------------
# Price / performance line chart
# ---------------------------------------------------------------------------
def price_performance(price_df: pd.DataFrame, view: str, names: Dict[str, str],
                      highlight: Optional[List[str]] = None) -> Optional[go.Figure]:
    if price_df is None or price_df.empty:
        return None
    df = price_df.ffill()
    first = df.apply(lambda c: c[c > 0].iloc[0] if (c > 0).any() else np.nan)

    if view == "Cumulative Returns (%)":
        data = (df.divide(first) - 1) * 100
        y_title, title = "Return (%)", "Cumulative performance"
    elif view == "Normalized (Base 100)":
        data = df.divide(first) * 100
        y_title, title = "Indexed (base 100)", "Normalized price (base 100)"
    else:
        data = df
        y_title, title = "Price (local ccy)", "Absolute price"

    data = data.dropna(how="all")
    if data.empty:
        return None

    highlight = set(highlight or [])
    # With 100+ lines a legend and 100 hues are useless. Draw every line thin &
    # translucent in one muted colour, highlight the benchmark, and let hover
    # identify a line. Optionally spotlight the current top allocations.
    fig = go.Figure()
    ordered = [c for c in data.columns if c != BENCHMARK_TICKER] + \
              ([BENCHMARK_TICKER] if BENCHMARK_TICKER in data.columns else [])
    for col in ordered:
        is_bench = col == BENCHMARK_TICKER
        is_hi = col in highlight
        if is_bench:
            color, width, alpha = BENCH, 2.6, 1.0
        elif is_hi:
            color, width, alpha = ACCENT, 2.0, 0.95
        else:
            color, width, alpha = "#9aa3b0", 1.0, 0.5
        fig.add_scatter(
            x=data.index, y=data[col], mode="lines", name=names.get(col, col),
            line=dict(color=color, width=width),
            opacity=alpha,
            hovertemplate="<b>" + names.get(col, col) + f"</b> ({col})<br>%{{y:.1f}}<extra></extra>",
        )
    fig.update_layout(_base_layout(title, 560, y_title=y_title))
    fig.update_layout(hovermode="closest", xaxis=dict(gridcolor=GRID, showline=False))
    if view != "Absolute Price (local ccy)":
        fig.add_hline(y=0 if view == "Cumulative Returns (%)" else 100,
                      line=dict(color="#c4c9d2", width=1))
    return fig


# ---------------------------------------------------------------------------
# Ranking bar charts
# ---------------------------------------------------------------------------
def total_return_bars(metrics, names):
    if metrics.empty or "Total Return (%)" not in metrics.columns:
        return None
    bench = metrics.loc[BENCHMARK_TICKER, "Total Return (%)"] if BENCHMARK_TICKER in metrics.index else None
    frame = _sorted_for_bars(metrics["Total Return (%)"], names, ascending=True)
    return _hbar(frame, "Total return over period", "Total return (%)",
                 color_fn=lambda v: NEUTRAL if pd.isna(v) else (POS if v >= 0 else NEG),
                 value_fmt="{:+.0f}", bench_value=bench)


def sharpe_bars(metrics, names):
    if metrics.empty or "Sharpe Ratio" not in metrics.columns:
        return None
    bench = metrics.loc[BENCHMARK_TICKER, "Sharpe Ratio"] if BENCHMARK_TICKER in metrics.index else None
    frame = _sorted_for_bars(metrics["Sharpe Ratio"], names, ascending=True)
    return _hbar(frame, "Sharpe ratio — return per unit of risk", "Sharpe",
                 color_fn=lambda v: NEUTRAL if pd.isna(v) else (POS if v >= 0 else NEG),
                 value_fmt="{:.2f}", bench_value=bench)


def volatility_bars(metrics, names):
    if metrics.empty or "Volatility (Ann %)" not in metrics.columns:
        return None
    bench = metrics.loc[BENCHMARK_TICKER, "Volatility (Ann %)"] if BENCHMARK_TICKER in metrics.index else None
    frame = _sorted_for_bars(metrics["Volatility (Ann %)"], names, ascending=False)
    return _hbar(frame, "Annualized volatility — lower is calmer", "Volatility (%)",
                 color_fn=lambda v: NEUTRAL if pd.isna(v) else NEG,
                 value_fmt="{:.0f}", bench_value=bench)


def drawdown_bars(metrics, names):
    if metrics.empty or "Max Drawdown (%)" not in metrics.columns:
        return None
    bench = metrics.loc[BENCHMARK_TICKER, "Max Drawdown (%)"] if BENCHMARK_TICKER in metrics.index else None
    frame = _sorted_for_bars(metrics["Max Drawdown (%)"], names, ascending=True)
    return _hbar(frame, "Maximum drawdown — worst peak-to-trough", "Drawdown (%)",
                 color_fn=lambda v: NEUTRAL if pd.isna(v) else "#a8412a",
                 value_fmt="{:.0f}", bench_value=bench)


def pe_bars(metrics, names):
    if metrics.empty or "P/E Ratio" not in metrics.columns:
        return None
    pe = pd.to_numeric(metrics["P/E Ratio"], errors="coerce")
    pe = pe.where(pe > 0)
    if pe.dropna().empty:
        return None
    frame = _sorted_for_bars(pe, names, ascending=False)
    frame = frame.dropna(subset=["value"])
    med = float(np.median(frame["value"])) if not frame.empty else None
    fig = _hbar(frame, "Valuation — P/E ratio", "P/E",
                color_fn=lambda v: ACCENT, value_fmt="{:.0f}",
                bench_value=med, bench_label="median")
    return fig


def sma_bars(metrics, names):
    sma_cols = [c for c in metrics.columns if "Diff from SMA" in c]
    if metrics.empty or not sma_cols:
        return None
    col = sma_cols[0]
    frame = _sorted_for_bars(metrics[col], names, ascending=True)
    return _hbar(frame, "Momentum — distance from moving average", "% above / below SMA",
                 color_fn=lambda v: NEUTRAL if pd.isna(v) else (POS if v >= 0 else NEG),
                 value_fmt="{:+.0f}", bench_value=0.0, bench_label="SMA")


# ---------------------------------------------------------------------------
# Risk vs return scatter
# ---------------------------------------------------------------------------
def risk_return_scatter(metrics: pd.DataFrame, names: Dict[str, str],
                        market_caps: Optional[pd.Series] = None,
                        allocation: Optional[pd.Series] = None) -> Optional[go.Figure]:
    needed = {"Volatility (Ann %)", "Total Return (%)"}
    if metrics.empty or not needed.issubset(metrics.columns):
        return None
    df = metrics[["Volatility (Ann %)", "Total Return (%)"]].dropna()
    if df.empty:
        return None

    caps = None
    if market_caps is not None:
        caps = pd.to_numeric(market_caps, errors="coerce").reindex(df.index)
    max_cap = float(caps.max()) if caps is not None and caps.notna().any() else None

    alloc = None
    if allocation is not None:
        alloc = pd.to_numeric(allocation, errors="coerce").reindex(df.index).fillna(0)

    bx = by = None
    if BENCHMARK_TICKER in df.index:
        bx = float(df.loc[BENCHMARK_TICKER, "Volatility (Ann %)"])
        by = float(df.loc[BENCHMARK_TICKER, "Total Return (%)"])

    body = df.drop(index=[BENCHMARK_TICKER], errors="ignore")

    # size ~ sqrt(market cap); allocation names get a stronger colour
    sizes, colors, texts, hovers, xs, ys = [], [], [], [], [], []
    # decide which names get a printed label: extremes only, so text doesn't pile up
    to_label = set()
    if not body.empty:
        to_label |= set(body["Total Return (%)"].nlargest(6).index)
        to_label |= set(body["Total Return (%)"].nsmallest(4).index)
        to_label |= set(body["Volatility (Ann %)"].nlargest(3).index)
        to_label |= set(body["Volatility (Ann %)"].nsmallest(3).index)
    if alloc is not None:
        to_label |= set(alloc[alloc > 0].nlargest(8).index)

    for t in body.index:
        x = float(body.loc[t, "Volatility (Ann %)"])
        y = float(body.loc[t, "Total Return (%)"])
        xs.append(x); ys.append(y)
        size = 12.0
        if max_cap and caps is not None and pd.notna(caps.get(t)):
            size = 9 + 34 * float(np.sqrt(max(caps[t], 0) / max_cap))
        sizes.append(size)
        held = alloc is not None and float(alloc.get(t, 0)) > 0
        colors.append(ACCENT if held else "#c3cad4")
        texts.append(names.get(t, t) if t in to_label else "")
        cap_txt = f"<br>Mkt cap: ${caps[t]:.0f}B" if (caps is not None and pd.notna(caps.get(t))) else ""
        alloc_txt = f"<br>Allocation: {alloc[t]:.1f}%" if (held) else ""
        hovers.append(f"<b>{names.get(t, t)}</b> ({t})<br>Return: {y:+.1f}%<br>"
                      f"Volatility: {x:.1f}%{cap_txt}{alloc_txt}")

    fig = go.Figure()
    fig.add_scatter(
        x=xs, y=ys, mode="markers+text",
        marker=dict(size=sizes, color=colors, line=dict(color="white", width=1),
                    opacity=0.85),
        text=texts, textposition="top center", textfont=dict(size=10, color=INK),
        hovertext=hovers, hoverinfo="text",
    )
    if bx is not None:
        fig.add_vline(x=bx, line=dict(color=BENCH, width=1.2, dash="dash"))
        fig.add_hline(y=by, line=dict(color=BENCH, width=1.2, dash="dash"))
        fig.add_scatter(x=[bx], y=[by], mode="markers+text",
                        marker=dict(size=18, color=BENCH, symbol="star"),
                        text=["SPY"], textposition="bottom center",
                        textfont=dict(size=11, color=BENCH),
                        hovertext=[f"<b>S&P 500 ETF</b> (SPY)<br>Return: {by:+.1f}%<br>Volatility: {bx:.1f}%"],
                        hoverinfo="text")

    fig.update_layout(_base_layout(
        "Risk vs return — bubble size ≈ market cap · blue = in portfolio", 620,
        x_title="Annualized volatility (%)  →  more risk",
        y_title="Total return over period (%)"))
    fig.update_layout(xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID))
    if bx is not None:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=-0.13, showarrow=False,
            text="Above the horizontal line beat SPY · left of the vertical line did it with less risk",
            font=dict(size=11, color="#66707c"))
    return fig


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------
def allocation_bars(allocation: pd.Series, names: Dict[str, str]) -> Optional[go.Figure]:
    a = allocation[allocation.index.notna() & allocation.notna() & (allocation > 0)]
    if a.empty:
        return None
    a = a.sort_values(ascending=True)
    labels = [names.get(t, t) for t in a.index]
    fig = go.Figure()
    fig.add_bar(
        x=a.values, y=labels, orientation="h",
        marker=dict(color=ACCENT, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in a.values], textposition="outside",
        textfont=dict(size=11, color=INK), cliponaxis=False,
        customdata=list(a.index),
        hovertemplate="<b>%{y}</b> (%{customdata})<br>Weight: %{x:.2f}%<extra></extra>",
    )
    fig.update_layout(_base_layout(
        f"Suggested allocation — {len(a)} positions", _row_height(len(a)),
        x_title="Weight (%)"))
    fig.update_yaxes(autorange="reversed")
    return fig


def allocation_donut(allocation: pd.Series, names: Dict[str, str],
                     top: int = 12) -> Optional[go.Figure]:
    """A compact donut of the largest positions, everything else rolled into
    'Other'. Complements the full bar list for an at-a-glance concentration read."""
    a = allocation[allocation.notna() & (allocation > 0)].sort_values(ascending=False)
    if a.empty:
        return None
    if len(a) > top:
        head = a.iloc[:top]
        other = a.iloc[top:].sum()
        labels = [names.get(t, t) for t in head.index] + ["Other"]
        values = head.values.tolist() + [other]
    else:
        labels = [names.get(t, t) for t in a.index]
        values = a.values.tolist()

    palette = ["#3a6ea5", "#0e8f7e", "#d9714e", "#8a6fbf", "#c9a227", "#4c9f70",
               "#c05780", "#5b8bb0", "#e08a3c", "#6b7f8c", "#9f5b5b", "#7aa0c4",
               "#b9bfc9"]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.58,
        marker=dict(colors=palette[:len(labels)], line=dict(color="white", width=1.5)),
        sort=False, direction="clockwise",
        textposition="outside", textinfo="label+percent",
        textfont=dict(size=11, color=INK),
        hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Concentration — top positions", font=dict(size=17, color=INK),
                   x=0, xanchor="left"),
        height=460, margin=dict(l=10, r=10, t=58, b=10),
        paper_bgcolor=BG, plot_bgcolor=BG, font=FONT, showlegend=False)
    return fig
