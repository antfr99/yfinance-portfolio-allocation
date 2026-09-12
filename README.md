# Equity — Risk & Momentum Analyzer (Streamlit)

Score any set of stocks on risk-adjusted return and momentum, get a suggested
portfolio allocation, and see supporting charts. A Streamlit rebuild of the
Gradio/Hugging Face analyzer, with the ecosystem filters removed in favour of a
single **fully editable ticker universe**.

## What's different from the Gradio version

- **No ecosystem/sector dropdowns.** You edit one flat list of tickers. The
  uploaded EasyEquities list (118 names) loads by default; add, remove, paste,
  or reset it in the sidebar. `SPY` is always added as the benchmark.
- **Company names + countries are resolved live** from Yahoo (and cached a
  month), instead of a hand-maintained dictionary — so any ticker you add gets
  a proper name with no code change.
- **Charts are rebuilt for long lists.** Rankings are interactive horizontal
  bars (one name per row, sorted, benchmark marked) that grow in height instead
  of cramping; the risk/return scatter labels only the extremes and makes the
  rest hoverable; the price chart draws all lines muted with the benchmark and
  your top allocations highlighted.

The scoring, metrics, caching and allocation logic are ported unchanged, so the
numbers match the original.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then set your universe in the sidebar and press **Analyse**.

## Deploy

Works as-is on **Streamlit Community Cloud** (point it at `app.py`) or a
**Hugging Face Space** (SDK: Streamlit). The disk cache lives under
`~/.cache/equity_analyzer`; set `EQUITY_CACHE_DIR` to relocate it.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI — universe editor, run flow, layout |
| `core.py` | Data layer + metrics + allocation scoring |
| `charts.py` | Plotly charts designed for long lists |

## Allocation score

```
Score = 0.40 × Sharpe + 0.30 × Momentum − 0.12 × Volatility − 0.08 × Max Drawdown − 0.10 × P/E
```

Each term is winsorized (5th/95th pct) then normalized 0→1. Positions under 2%
are dropped, anything over the max position size is capped with the excess
redistributed pro-rata, and the book renormalizes to 100%. Names trading under
60% of the benchmark's sessions sit out.

*Not investment advice — a screening tool on free, best-effort market data.*
