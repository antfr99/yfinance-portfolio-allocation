"""
Core financial logic for the Equity Risk & Momentum Analyzer.

Ported from the original Gradio/Hugging Face app and adapted for Streamlit:
  - The hard-coded ECOSYSTEM registry is gone. The universe is now a flat,
    fully user-editable list of tickers (the EasyEquities CSV is the default).
  - Company short-names and countries are resolved live from Yahoo instead of
    a curated dict, and cached to disk.
  - Everything below the data layer (fundamentals, metrics, allocation scoring)
    is the same battle-tested logic, so the numbers match the original app.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BENCHMARK_TICKER = "SPY"
BENCHMARK_NAME = "S&P 500 ETF"

TRADING_DAYS = 252

PERIOD_CHOICES = ["1mo", "3mo", "6mo", "ytd", "1y", "2y"]
PRICE_VIEW_CHOICES = [
    "Cumulative Returns (%)",
    "Normalized (Base 100)",
    "Absolute Price (local ccy)",
]

CACHE_DIR = Path(os.environ.get("EQUITY_CACHE_DIR", Path.home() / ".cache" / "equity_analyzer"))
FUNDAMENTAL_TTL = 6 * 3600
PRICE_TTL = 10 * 60
FX_TTL = 6 * 3600
PROFILE_TTL = 30 * 24 * 3600   # names/countries barely change — cache for a month

MAX_WORKERS = 6
MIN_REQUEST_INTERVAL = 0.12
INFO_RETRIES = 2

# Allocation defaults
DEFAULT_RISK_FREE = 4.0
DEFAULT_MAX_WEIGHT = 25.0
MIN_WEIGHT = 0.02
MIN_COVERAGE = 60.0

WINSOR_QUANTILE: Optional[float] = None

try:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
except Exception as _exc:
    print(f"[WARN] disk cache disabled: {_exc}")
    CACHE_DIR = None


# ---------------------------------------------------------------------------
# Default universe — the uploaded EasyEquities list
# ---------------------------------------------------------------------------
DEFAULT_TICKERS: List[str] = [
    "MARA", "SPOT", "STX", "CRCL", "SMCI", "COIN", "NVDA", "SKHY", "RBLX", "MU",
    "MSTR", "SNDK", "LRCX", "KLAC", "CSCO", "AVGO", "QCOM", "IBM", "LITE", "INFQ",
    "NOK", "WDC", "UI", "RGTI", "TSM", "QBTS", "IONQ", "AMD", "TER", "COHR",
    "ASML", "ADI", "AMAT", "INTC", "NVTS", "GLW", "UMC", "CBRS", "CIEN", "MRVL",
    "WOLF", "Q", "NXPI", "CRDO", "ON", "DELL", "QRVO", "AEHR", "POET", "META",
    "SPCX", "HIMX", "HPE", "ENTG", "U", "AAPL", "UCTT", "TXN", "DOCN", "MCHP",
    "CEG", "GOOG", "HON", "QUBT", "TWLO", "SNOW", "ANET", "NET", "FTNT", "AI",
    "STM", "ESTC", "DT", "ORCL", "BOX", "CRM", "NEE", "TEAM", "CRWD", "PLTR",
    "DLR", "MSFT", "EQIX", "VEEV", "ETN", "WDAY", "INTU", "DUOL", "HOOD", "TEM",
    "IOT", "NOW", "INDI", "HUBS", "SNPS", "BBAI", "DDOG", "GEV", "PANW", "BABA",
    "ADBE", "VRT", "APPN", "MPWR", "ALAB", "CRWV", "SOUN", "APLD", "AMZN", "CDNS",
    "ADSK", "PATH", "AMBA", "NBIS", "MDB", "APH", "DRAM", "ARM",
]


# ---------------------------------------------------------------------------
# Rate limiter + disk cache
# ---------------------------------------------------------------------------
class _RateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            self._next = max(now, self._next) + self.min_interval
        if wait > 0:
            time.sleep(wait)


_LIMITER = _RateLimiter(MIN_REQUEST_INTERVAL)


def _cache_file(kind: str, key: str) -> Optional[Path]:
    if CACHE_DIR is None:
        return None
    digest = hashlib.md5(f"{kind}:{key}".encode()).hexdigest()[:20]
    return CACHE_DIR / f"{kind}_{digest}.pkl"


def _disk_get(kind: str, key: str, ttl: float):
    path = _cache_file(kind, key)
    if path is None or not path.exists():
        return None
    try:
        if time.time() - path.stat().st_mtime > ttl:
            return None
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def _disk_put(kind: str, key: str, obj) -> None:
    path = _cache_file(kind, key)
    if path is None:
        return
    try:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(obj, fh)
        tmp.replace(path)
    except Exception as exc:
        print(f"[WARN] cache write failed: {exc}")


def clear_caches() -> int:
    removed = 0
    if CACHE_DIR is not None:
        for f in CACHE_DIR.glob("*.pkl"):
            try:
                f.unlink()
                removed += 1
            except Exception:
                pass
    return removed


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------
def _build_session():
    try:
        from curl_cffi import requests as curl_requests
        return curl_requests.Session(impersonate="chrome")
    except Exception:
        return None


_SESSION = _build_session()


def _mk_ticker(ticker: str) -> "yf.Ticker":
    if _SESSION is not None:
        try:
            return yf.Ticker(ticker, session=_SESSION)
        except Exception:
            pass
    return yf.Ticker(ticker)


def _fi_get(fast_info, *keys):
    for k in keys:
        try:
            val = fast_info[k]
            if val is not None:
                return val
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Batch quote (Yahoo /v7) — one request for the whole universe
# ---------------------------------------------------------------------------
_QUOTE_URL = "https://query2.finance.yahoo.com/v7/finance/quote"
_QUOTE_FIELDS = ("symbol,shortName,longName,regularMarketPrice,marketCap,trailingPE,"
                 "forwardPE,epsTrailingTwelveMonths,epsForward,currency")


def _batch_quote(tickers: Sequence[str]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    tickers = list(tickers)
    if not tickers:
        return out
    try:
        from yfinance.data import YfData
    except Exception as exc:
        print(f"[INFO] batch quote unavailable ({exc})")
        return out
    try:
        yfd = YfData(session=_SESSION)
    except Exception:
        try:
            yfd = YfData()
        except Exception as exc:
            print(f"[INFO] YfData init failed ({exc})")
            return out

    consecutive_failures = 0
    for i in range(0, len(tickers), 40):
        chunk = tickers[i:i + 40]
        params = {"symbols": ",".join(chunk), "fields": _QUOTE_FIELDS}
        try:
            try:
                params["crumb"] = yfd._get_crumb()
            except Exception:
                pass
            js = yfd.get_raw_json(_QUOTE_URL, params=params)
            results = (js or {}).get("quoteResponse", {}).get("result") or []
            for q in results:
                sym = q.get("symbol")
                if sym:
                    out[sym] = q
            consecutive_failures = 0
        except Exception as exc:
            consecutive_failures += 1
            print(f"[INFO] batch quote chunk failed ({exc})")
            if consecutive_failures >= 2:
                break
    return out


# ---------------------------------------------------------------------------
# Company profile (name + country), resolved live and cached
# ---------------------------------------------------------------------------
_PROFILE_CACHE: Dict[str, Tuple[float, dict]] = {}


def _clean_name(raw: Optional[str], ticker: str) -> str:
    if not raw:
        return ticker
    name = str(raw).strip()
    for suffix in (", Inc.", " Inc.", " Incorporated", " Corporation", " Corp.",
                   " Limited", " Ltd.", " plc", " N.V.", " S.A.", " AG",
                   " Co., Ltd.", " Company", " Holdings"):
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()
    return name or ticker


def resolve_profiles(tickers: List[str], force: bool = False) -> pd.DataFrame:
    """ticker -> {Company, Country}. Cheap: names come from the batch quote
    where possible, falling back to .info only for what's missing. Cached for a
    month since these rarely change."""
    out: Dict[str, dict] = {}
    todo: List[str] = []
    for t in tickers:
        if not force:
            hit = _PROFILE_CACHE.get(t)
            if hit and time.time() - hit[0] < PROFILE_TTL:
                out[t] = hit[1]
                continue
            disk = _disk_get("profile", t, PROFILE_TTL)
            if disk is not None:
                _PROFILE_CACHE[t] = (time.time(), disk)
                out[t] = disk
                continue
        todo.append(t)

    if todo:
        quotes = _batch_quote(todo)
        still: List[str] = []
        for t in todo:
            q = quotes.get(t)
            if q and (q.get("shortName") or q.get("longName")):
                row = {"Company": _clean_name(q.get("shortName") or q.get("longName"), t),
                       "Country": ""}
                out[t] = row
                _PROFILE_CACHE[t] = (time.time(), row)
                _disk_put("profile", t, row)
            else:
                still.append(t)

        # Deep fallback for names the quote endpoint didn't carry
        def _one(tk: str):
            _LIMITER.acquire()
            try:
                info = _mk_ticker(tk).info or {}
            except Exception:
                info = {}
            row = {
                "Company": _clean_name(info.get("shortName") or info.get("longName"), tk),
                "Country": info.get("country") or "",
            }
            return tk, row

        if still:
            with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(still))) as pool:
                for tk, row in pool.map(_one, still):
                    out[tk] = row
                    _PROFILE_CACHE[tk] = (time.time(), row)
                    _disk_put("profile", tk, row)

    for t in tickers:
        out.setdefault(t, {"Company": t, "Country": ""})
    out.setdefault(BENCHMARK_TICKER, {"Company": BENCHMARK_NAME, "Country": "USA"})
    return pd.DataFrame.from_dict(out, orient="index")


def name_map(profiles: pd.DataFrame) -> Dict[str, str]:
    if profiles is None or profiles.empty or "Company" not in profiles.columns:
        return {BENCHMARK_TICKER: BENCHMARK_NAME}
    m = profiles["Company"].to_dict()
    m.setdefault(BENCHMARK_TICKER, BENCHMARK_NAME)
    return m


# ---------------------------------------------------------------------------
# Fundamentals (P/E + market cap in USD)
# ---------------------------------------------------------------------------
_FUNDAMENTAL_CACHE: Dict[str, Tuple[float, dict]] = {}


def _cache_get_fundamental(ticker: str) -> Optional[dict]:
    hit = _FUNDAMENTAL_CACHE.get(ticker)
    if hit and (time.time() - hit[0]) < FUNDAMENTAL_TTL:
        return hit[1]
    disk = _disk_get("fund2", ticker, FUNDAMENTAL_TTL)
    if disk is not None:
        _FUNDAMENTAL_CACHE[ticker] = (time.time(), disk)
        return disk
    return None


def _cache_put_fundamental(ticker: str, row: dict) -> None:
    _FUNDAMENTAL_CACHE[ticker] = (time.time(), row)
    _disk_put("fund2", ticker, row)


def _eps_from_statements(tk) -> Optional[float]:
    for attr, is_quarterly in (("quarterly_income_stmt", True), ("income_stmt", False)):
        try:
            df = getattr(tk, attr)
        except Exception:
            continue
        if df is None or getattr(df, "empty", True):
            continue
        for row in ("Diluted EPS", "Basic EPS"):
            if row not in df.index:
                continue
            try:
                series = df.loc[row]
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[0]
                vals = pd.to_numeric(series, errors="coerce").dropna()
                if vals.empty:
                    continue
                vals = vals.sort_index(ascending=False)
                eps = float(vals.iloc[:4].sum()) if (is_quarterly and len(vals) >= 4) else float(vals.iloc[0])
                if eps:
                    return eps
            except Exception:
                continue
    return None


def _fetch_single_fundamental(ticker: str, seed: Optional[dict] = None) -> Tuple[str, dict]:
    tk = _mk_ticker(ticker)
    seed = seed or {}
    price = seed.get("_price")
    market_cap_local = seed.get("_market_cap", np.nan)
    currency = seed.get("Currency") or None

    if not price:
        _LIMITER.acquire()
        try:
            fi = tk.fast_info
            p = _fi_get(fi, "last_price", "lastPrice", "regular_market_price")
            price = float(p) if p else None
            if pd.isna(market_cap_local):
                mc = _fi_get(fi, "market_cap", "marketCap")
                market_cap_local = float(mc) if mc else np.nan
            currency = currency or _fi_get(fi, "currency")
        except Exception as exc:
            print(f"[WARN] {ticker} fast_info failed: {exc}")

    info: dict = {}
    for attempt in range(INFO_RETRIES):
        _LIMITER.acquire()
        try:
            info = tk.info or {}
            if len(info) < 10:
                raise ValueError(f"incomplete response ({len(info)} keys)")
            break
        except Exception as exc:
            info = {}
            if attempt == INFO_RETRIES - 1:
                print(f"[WARN] {ticker} .info unavailable: {exc}")
                break
            time.sleep(1.0 * (2 ** attempt) + random.random() * 0.5)

    if not price:
        price = info.get("currentPrice") or info.get("regularMarketPrice")
    if pd.isna(market_cap_local) and info.get("marketCap"):
        market_cap_local = float(info["marketCap"])
    currency = currency or info.get("currency")

    pe, source = seed.get("P/E Ratio", np.nan), seed.get("P/E Source", "unavailable")
    if pd.isna(pe):
        eps = info.get("trailingEps")
        if price and eps and eps > 0:
            pe, source = round(price / eps, 2), "price/trailingEps"
    if pd.isna(pe) and (info.get("trailingPE") or 0) > 0:
        pe, source = round(float(info["trailingPE"]), 2), "trailingPE"
    if pd.isna(pe):
        f_eps = info.get("forwardEps")
        if price and f_eps and f_eps > 0:
            pe, source = round(price / f_eps, 2), "price/forwardEps (fwd)"
    if pd.isna(pe) and (info.get("forwardPE") or 0) > 0:
        pe, source = round(float(info["forwardPE"]), 2), "forwardPE (fwd)"
    if pd.isna(pe) and price:
        _LIMITER.acquire()
        stmt_eps = _eps_from_statements(tk)
        if stmt_eps and stmt_eps > 0:
            pe, source = round(price / stmt_eps, 2), "income statement EPS"
        elif stmt_eps is not None and stmt_eps <= 0:
            source = "negative earnings"

    return ticker, {
        "P/E Ratio": pe,
        "P/E Source": source,
        "_market_cap": market_cap_local,
        "Currency": currency or "",
    }


def _from_quote(q: dict) -> dict:
    price = q.get("regularMarketPrice")
    mc = q.get("marketCap")
    pe, source = np.nan, "unavailable"
    eps = q.get("epsTrailingTwelveMonths")
    if price and eps and eps > 0:
        pe, source = round(price / eps, 2), "price/trailingEps"
    elif (q.get("trailingPE") or 0) > 0:
        pe, source = round(float(q["trailingPE"]), 2), "trailingPE"
    elif price and (q.get("epsForward") or 0) > 0:
        pe, source = round(price / q["epsForward"], 2), "price/forwardEps (fwd)"
    elif (q.get("forwardPE") or 0) > 0:
        pe, source = round(float(q["forwardPE"]), 2), "forwardPE (fwd)"
    elif eps is not None and eps <= 0:
        source = "negative earnings"
    return {
        "P/E Ratio": pe, "P/E Source": source,
        "_market_cap": float(mc) if mc else np.nan,
        "Currency": q.get("currency") or "", "_price": price,
    }


_MINOR_UNITS = {"GBp": ("GBP", 0.01), "GBX": ("GBP", 0.01),
                "ILA": ("ILS", 0.01), "ZAc": ("ZAR", 0.01)}


def fx_to_usd(currencies: Set[str]) -> Dict[str, float]:
    rates: Dict[str, float] = {"USD": 1.0}
    needed: Set[str] = set()
    for c in currencies:
        base, _ = _MINOR_UNITS.get(c, (c, 1.0))
        if base and base != "USD":
            needed.add(base)
    for base in list(needed):
        hit = _disk_get("fx", base, FX_TTL)
        if hit:
            rates[base] = float(hit)
            needed.discard(base)
    if needed:
        symbols = {f"{b}USD=X": b for b in needed}
        quotes = _batch_quote(list(symbols))
        for sym, base in symbols.items():
            rate = (quotes.get(sym) or {}).get("regularMarketPrice")
            if not rate:
                try:
                    _LIMITER.acquire()
                    rate = _fi_get(_mk_ticker(sym).fast_info, "last_price", "lastPrice")
                except Exception as exc:
                    print(f"[WARN] FX {sym} failed: {exc}")
                    rate = None
            if rate:
                rates[base] = float(rate)
                _disk_put("fx", base, float(rate))
    out: Dict[str, float] = {}
    for c in currencies:
        base, mult = _MINOR_UNITS.get(c, (c, 1.0))
        rate = rates.get(base)
        out[c] = rate * mult if rate else np.nan
    out["USD"] = 1.0
    out.setdefault("", np.nan)
    return out


def fetch_fundamentals(tickers: List[str], force: bool = False) -> pd.DataFrame:
    out: Dict[str, dict] = {}
    todo: List[str] = []
    for t in tickers:
        hit = None if force else _cache_get_fundamental(t)
        if hit is not None:
            out[t] = hit
        else:
            todo.append(t)

    deep: Dict[str, dict] = {}
    if todo:
        quotes = _batch_quote(todo)
        for t in todo:
            q = quotes.get(t)
            if not q:
                deep[t] = {}
                continue
            row = _from_quote(q)
            unresolved = pd.isna(row["P/E Ratio"]) and row["P/E Source"] != "negative earnings"
            if unresolved or not row.get("_price"):
                deep[t] = row
            else:
                row.pop("_price", None)
                out[t] = row
                _cache_put_fundamental(t, row)

    if deep:
        print(f"[INFO] deep fetch for {len(deep)} ticker(s)")
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(deep))) as pool:
            futures = [pool.submit(_fetch_single_fundamental, t, seed) for t, seed in deep.items()]
            for fut in futures:
                try:
                    t, row = fut.result()
                except Exception as exc:
                    print(f"[WARN] deep fetch error: {exc}")
                    continue
                row.pop("_price", None)
                out[t] = row
                _cache_put_fundamental(t, row)

    for t in tickers:
        out.setdefault(t, {"P/E Ratio": np.nan, "P/E Source": "unavailable",
                           "_market_cap": np.nan, "Currency": ""})

    df = pd.DataFrame.from_dict(out, orient="index")
    if "_market_cap" not in df.columns:
        df["_market_cap"] = np.nan
    if "Currency" not in df.columns:
        df["Currency"] = ""

    ccys = {c for c in df["Currency"].fillna("").tolist() if c}
    fx = fx_to_usd(ccys) if ccys else {}
    mult = df["Currency"].map(lambda c: fx.get(c, 1.0 if c in ("", "USD") else np.nan))
    df["Market Cap ($B)"] = (pd.to_numeric(df["_market_cap"], errors="coerce")
                             * pd.to_numeric(mult, errors="coerce") / 1e9)
    return df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")


# ---------------------------------------------------------------------------
# Price data
# ---------------------------------------------------------------------------
_YF_FIELDS = {"Open", "High", "Low", "Close", "Adj Close", "Volume",
              "Dividends", "Stock Splits"}


def _extract_close(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    if not isinstance(raw.columns, pd.MultiIndex):
        col = "Adj Close" if "Adj Close" in raw.columns else "Close"
        if col not in raw.columns:
            return pd.DataFrame()
        return raw[[col]].rename(columns={col: tickers[0]})

    l0 = raw.columns.get_level_values(0).unique().tolist()
    l1 = raw.columns.get_level_values(1).unique().tolist()

    if l0[0] in _YF_FIELDS:
        col = "Adj Close" if "Adj Close" in l0 else "Close"
        price = raw[col] if col in raw.columns.get_level_values(0) else pd.DataFrame()
    elif l1[0] in _YF_FIELDS:
        col = "Adj Close" if "Adj Close" in l1 else "Close"
        price = raw.xs(col, axis=1, level=1, drop_level=True)
    else:
        price = pd.DataFrame()
        for level in (0, 1):
            for col in ("Adj Close", "Close"):
                try:
                    candidate = raw.xs(col, axis=1, level=level, drop_level=True)
                    if any(t in candidate.columns for t in tickers):
                        price = candidate
                        break
                except Exception:
                    continue
            else:
                continue
            break

    if isinstance(price, pd.Series):
        price = price.to_frame(name=tickers[0] if len(tickers) == 1 else "price")
    keep = [t for t in tickers if t in price.columns]
    return price[keep] if keep else pd.DataFrame()


def _download_multi(tickers: List[str], period: str) -> pd.DataFrame:
    base = dict(period=period, actions=False, progress=False, threads=True)
    for kwargs in [
        {**base, "auto_adjust": False},
        {**base, "auto_adjust": True},
        base,
    ]:
        try:
            raw = yf.download(tickers, **kwargs)
            if raw is not None and not raw.empty:
                return raw
        except Exception as exc:
            print(f"[WARN] download attempt failed ({exc})")
    return pd.DataFrame()


def _download_per_ticker(tickers: List[str], period: str) -> pd.DataFrame:
    frames = {}

    def _one(t):
        try:
            raw = yf.download(t, period=period, actions=False, progress=False, auto_adjust=False)
            if raw is None or raw.empty:
                raw = yf.download(t, period=period, actions=False, progress=False, auto_adjust=True)
            if raw is not None and not raw.empty:
                col = "Adj Close" if "Adj Close" in raw.columns else "Close"
                if col in raw.columns:
                    return t, raw[col]
        except Exception as exc:
            print(f"[WARN] per-ticker fetch failed {t}: {exc}")
        return t, None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for t, s in pool.map(_one, tickers):
            if s is not None:
                frames[t] = s
    return pd.DataFrame(frames) if frames else pd.DataFrame()


def fetch_price_data(tickers: List[str], period: str = "1y", force: bool = False) -> pd.DataFrame:
    cache_key = f"{period}|{','.join(sorted(tickers))}"
    if not force:
        cached = _disk_get("price", cache_key, PRICE_TTL)
        if cached is not None:
            return cached

    raw = _download_multi(tickers, period)
    price = _extract_close(raw, tickers) if not raw.empty else pd.DataFrame()
    if price.empty:
        price = _download_per_ticker(tickers, period)
    if price.empty:
        return pd.DataFrame()

    price = price.dropna(how="all")
    price = price.loc[:, price.notna().any()]
    if not price.empty:
        _disk_put("price", cache_key, price)
    return price


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _clean_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    s = s[s > 0]
    return s[~s.index.duplicated(keep="last")].sort_index()


def daily_returns(df: pd.DataFrame) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for col in df.columns:
        s = _clean_series(df, col)
        if len(s) < 3:
            out[col] = pd.Series(dtype=float)
            continue
        out[col] = (s.pct_change(fill_method=None)
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna())
    return out


def _max_drawdown(prices: pd.Series) -> float:
    if prices.empty:
        return np.nan
    running_max = prices.cummax()
    return float((prices / running_max - 1.0).min() * 100)


def late_starters(df: pd.DataFrame, tolerance: int = 5) -> List[str]:
    if df.empty:
        return []
    out = []
    for col in df.columns:
        s = _clean_series(df, col)
        if s.empty:
            continue
        try:
            pos = df.index.get_loc(s.index[0])
        except KeyError:
            continue
        if isinstance(pos, slice):
            pos = pos.start or 0
        if pos > tolerance:
            out.append(col)
    return out


def calculate_metrics(df: pd.DataFrame, fundamental_df: pd.DataFrame,
                      risk_free_rate: float = DEFAULT_RISK_FREE / 100) -> pd.DataFrame:
    if df.empty or len(df) < 2:
        return pd.DataFrame()

    rets = daily_returns(df)
    bench_rets = rets.get(BENCHMARK_TICKER, pd.Series(dtype=float))
    bench_px = _clean_series(df, BENCHMARK_TICKER) if BENCHMARK_TICKER in df.columns else pd.Series(dtype=float)

    if not bench_px.empty:
        session_count = len(bench_px)
    else:
        session_count = max((len(_clean_series(df, c)) for c in df.columns), default=1)
    session_count = max(session_count, 1)

    rf_daily = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    window = min(50, len(df))

    metrics: Dict[str, dict] = {}
    for col in df.columns:
        s = _clean_series(df, col)
        if s.empty:
            continue
        r = rets.get(col, pd.Series(dtype=float))
        price = float(s.iloc[-1])
        total_return = (price / float(s.iloc[0]) - 1.0) * 100

        if len(r) > 2 and r.std() > 0:
            sd = float(r.std())
            volatility = sd * np.sqrt(TRADING_DAYS) * 100
            sharpe = float((r - rf_daily).mean() / sd * np.sqrt(TRADING_DAYS))
        else:
            volatility, sharpe = np.nan, np.nan

        sma = float(s.rolling(window=min(window, len(s)), min_periods=1).mean().iloc[-1])
        sma_diff = ((price - sma) / sma * 100) if sma else np.nan

        beta, excess = np.nan, np.nan
        if col == BENCHMARK_TICKER:
            beta, excess = 1.0, 0.0
        else:
            if not bench_rets.empty and len(r) > 2:
                joint = pd.concat([r.rename("a"), bench_rets.rename("b")], axis=1).dropna()
                if len(joint) >= 30:
                    var_b = float(joint["b"].var())
                    if var_b > 1e-12:
                        beta = float(joint["a"].cov(joint["b"]) / var_b)
            if not bench_px.empty:
                common = s.index.intersection(bench_px.index)
                if len(common) >= 2:
                    a0, a1 = float(s.loc[common[0]]), float(s.loc[common[-1]])
                    b0, b1 = float(bench_px.loc[common[0]]), float(bench_px.loc[common[-1]])
                    if a0 > 0 and b0 > 0:
                        excess = ((a1 / a0) - (b1 / b0)) * 100

        metrics[col] = {
            "Current Price": price,
            "Total Return (%)": total_return,
            "Excess vs Bench (pp)": excess,
            "Volatility (Ann %)": volatility,
            "Sharpe Ratio": sharpe,
            "Max Drawdown (%)": _max_drawdown(s),
            "Beta": beta,
            f"Diff from SMA{window} (%)": sma_diff,
            "P/E Ratio": (
                fundamental_df.loc[col, "P/E Ratio"]
                if (not fundamental_df.empty and col in fundamental_df.index
                    and "P/E Ratio" in fundamental_df.columns)
                else np.nan
            ),
            "Coverage (%)": min(100.0, len(s) / session_count * 100),
        }

    if not metrics:
        return pd.DataFrame()
    return pd.DataFrame.from_dict(metrics, orient="index").round(2)


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------
def _winsorized_normalize(series: pd.Series, quantile: Optional[float] = None) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(0.5, index=series.index)
    s = s.fillna(s.median())
    if len(s) >= 8:
        q = quantile if quantile is not None else WINSOR_QUANTILE
        if q is None:
            q = 0.10 if len(s) < 20 else 0.05
        if q <= 0:
            return (s - s.min()) / (s.max() - s.min()) if float(s.max() - s.min()) > 1e-9 \
                else pd.Series(0.5, index=series.index)
        s = s.clip(float(s.quantile(q)), float(s.quantile(1 - q)))
    rng = float(s.max() - s.min())
    if rng < 1e-9:
        return pd.Series(0.5, index=series.index)
    return (s - s.min()) / rng


def _cap_weights(weights: pd.Series, cap: float) -> pd.Series:
    w = weights.astype(float).copy()
    if cap <= 0 or cap >= 1 or w.sum() <= 0:
        return w
    held = int((w > 0).sum())
    if held * cap <= 1.0 + 1e-9:
        return w
    for _ in range(100):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        room = ~over & (w > 0)
        if not room.any():
            break
        w[room] += excess * w[room] / float(w[room].sum())
    return w


def compute_allocation(metrics: pd.DataFrame,
                       max_weight: float = DEFAULT_MAX_WEIGHT / 100,
                       min_weight: float = MIN_WEIGHT,
                       min_coverage: float = MIN_COVERAGE,
                       zero_out: bool = False) -> pd.Series:
    """Score-weighted allocation.

    zero_out=False (default): every scored name gets a positive weight. The
      score still sets the magnitude — best names biggest — but the 2% floor
      is not applied, and names below the coverage threshold are kept at a
      small floor weight instead of being dropped. So the allocation chart
      covers the whole list.
    zero_out=True: original behaviour — sub-2% and low-coverage names go to 0,
      giving a tighter book of only the strongest names.
    """
    df = metrics.copy()
    df = df.drop(index=[BENCHMARK_TICKER], errors="ignore")
    if df.empty:
        return pd.Series(dtype=float)

    # Coverage: in zero_out mode we exclude low-coverage names; otherwise we
    # keep them but remember which they are so they only get a floor weight
    # (their Sharpe/momentum are measured over a shorter window, so we don't
    # want the score to hand them a big position on thin data).
    low_cov = pd.Series(False, index=df.index)
    if "Coverage (%)" in df.columns:
        cov = pd.to_numeric(df["Coverage (%)"], errors="coerce").fillna(0)
        if zero_out:
            df = df[cov >= min_coverage]
        else:
            low_cov = cov < min_coverage
    if df.empty:
        return pd.Series(dtype=float)

    pe = df["P/E Ratio"].where(df["P/E Ratio"] > 0)
    median_pe = pe.median()
    pe = pe.fillna(median_pe if not pd.isna(median_pe) else 20.0)
    log_pe = np.log(pe.clip(lower=0.5))

    # In no-zero mode a missing Sharpe/Vol shouldn't drop the name entirely —
    # give it the neutral middle of each component instead.
    if zero_out:
        df = df.dropna(subset=["Sharpe Ratio", "Volatility (Ann %)"])
        if df.empty:
            return pd.Series(dtype=float)
    low_cov = low_cov.reindex(df.index).fillna(False)

    sma_cols = [c for c in df.columns if "Diff from SMA" in c]
    if not sma_cols:
        return pd.Series(dtype=float)
    momentum_col = sma_cols[0]

    if "Max Drawdown (%)" in df.columns:
        drawdown_pen = _winsorized_normalize((-df["Max Drawdown (%)"]).reindex(df.index))
    else:
        drawdown_pen = pd.Series(0.5, index=df.index)

    score = (
        0.40 * _winsorized_normalize(df["Sharpe Ratio"])
        + 0.30 * _winsorized_normalize(df[momentum_col])
        - 0.12 * _winsorized_normalize(df["Volatility (Ann %)"])
        - 0.08 * drawdown_pen
        - 0.10 * _winsorized_normalize(log_pe.reindex(df.index))
    )

    if zero_out:
        weight = score.clip(lower=0)
        if float(weight.sum()) < 1e-9:
            weight = pd.Series(1.0, index=df.index)
        weight = weight / weight.sum()
        below_min = weight < min_weight
        if below_min.any() and not below_min.all():
            weight[below_min] = 0.0
            weight = weight / weight.sum()
    else:
        # Everyone in. Shift the score so the lowest scorer still gets a
        # positive share, rather than clipping it to zero. A small epsilon
        # floor keeps even the weakest name visible on the chart.
        s = score.astype(float)
        lo = float(s.min())
        span = float(s.max() - lo)
        if span < 1e-9:
            weight = pd.Series(1.0, index=df.index)
        else:
            # Map scores to [0.15, 1.0] so the best name is ~7x the worst,
            # not infinitely larger. Low-coverage names are pulled toward the
            # floor so thin data can't earn a big weight.
            weight = 0.15 + 0.85 * (s - lo) / span
            weight[low_cov] = weight[low_cov].clip(upper=0.30)
        weight = weight / weight.sum()

    weight = _cap_weights(weight, max_weight)
    weight = weight / weight.sum()
    return (weight * 100).round(2)
