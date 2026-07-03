"""Delisted-name price recovery for the spinoff survivorship-hole re-attack (free).

The primary spinoff pass (strategies/spinoff/backtest.py) only prices names that resolve via
SEC's current-issuer ticker map, so the ~2,485 delisted/acquired Form 10 registrants are
absent -- the loud survivorship hole that biases the YELLOW verdict UPWARD. This module
recovers daily histories for those delisted names from free sources, keyed by the HISTORICAL
symbol resolved in :func:`libs.data.academic.form10.recover_delisted_tickers`, into a SEPARATE
cache so the recovered losers never contaminate the pre-registered primary basket.

Sources, tried in order (STATUS.md "Planned Retests"):
  1. Yahoo v8 chart API -- often retains delisted history under the old symbol; carries an
     adjusted close (splits + dividends), so it is preferred.
  2. Stooq daily CSV -- fallback for names Yahoo dropped. Stooq gives an UNADJUSTED close only,
     so ``adjclose`` is set equal to ``close`` (a disclosed caveat: over a 24-month hold,
     dividend drift is small but splits are not adjusted; Yahoo is preferred wherever it has
     coverage). Recorded via the ``source`` return of :func:`load_ohlcv_recovered`.

The output DataFrame matches :func:`strategies.spinoff.backtest.load_ohlcv` exactly (columns
``[adjclose, close, volume]``, ``DatetimeIndex``, parquet) so the existing basket construction
consumes it unchanged. Unlike the primary loader, both fetches are wrapped in a global throttle
+ retry/backoff: at 2,485-name scale in the cloud matrix a naive loop trips the same shared-IP
429 wall the EDGAR spine was hardened against.

RESEARCH ONLY (ADR 0003): may import libs.data.academic; never imported by live/execution code.
"""
from __future__ import annotations

import io
import os
import random
import time
from pathlib import Path

import pandas as pd

try:  # repo layout
    from libs.data.academic.form4 import _Throttle
except ImportError:  # flat checkout (public cloud-build matrix repo)
    from form4 import _Throttle  # type: ignore

# Separate git-ignored cache so recovered delisted prices are never confused with the primary
# basket's live-ticker caches. Overridable for the cloud matrix (parity with FORM10_CACHE_DIR).
_CACHE = Path(
    os.environ.get(
        "SPINOFF_DELISTED_CACHE_DIR",
        Path(__file__).resolve().parents[2] / "data" / "prices" / "spinoff_delisted",
    )
)

# Price feeds are not the SEC, but a 2,485-name sweep from shared cloud IPs still gets
# rate-limited; space requests and back off. Gentler than the 0.11s SEC spacing.
_MIN_INTERVAL = float(os.environ.get("SPINOFF_PRICE_MIN_INTERVAL", "0.3"))
_RETRIES = 4
_BACKOFF_BASE = 1.0
_BACKOFF_CAP = 20.0
_BACKOFF_JITTER = 0.5

_THROTTLE = _Throttle(min_interval=_MIN_INTERVAL)

_YAHOO_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    "?period1=0&period2=9999999999&interval=1d"
)
_STOOQ_URL = "https://stooq.com/q/d/l/?s={ticker}.us&i=d"


def _safe(ticker: str) -> str:
    return ticker.replace("/", "-").replace(".", "-")


def _fetch(url: str, throttle: _Throttle = _THROTTLE):
    """GET with throttle + retry/backoff. Returns the ``requests.Response`` or ``None``."""
    import requests

    for attempt in range(_RETRIES):
        throttle.wait()
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        except Exception:
            time.sleep(0.5 * (attempt + 1) + random.uniform(0, _BACKOFF_JITTER))
            continue
        if r.status_code == 200:
            return r
        if r.status_code == 429 or r.status_code >= 500:
            delay = min(_BACKOFF_CAP, _BACKOFF_BASE * (2 ** attempt)) + random.uniform(0, _BACKOFF_JITTER)
            time.sleep(delay)
            continue
        return None  # 404 etc.: nothing to retry
    return None


def _from_yahoo(ticker: str) -> pd.DataFrame | None:
    """Daily ``[adjclose, close, volume]`` from Yahoo v8, or ``None`` when Yahoo has no data."""
    r = _fetch(_YAHOO_URL.format(ticker=ticker))
    if r is None:
        return None
    try:
        res = r.json()["chart"]["result"][0]
        idx = pd.to_datetime(res["timestamp"], unit="s").normalize()
        quote = res["indicators"]["quote"][0]
        adj = res["indicators"]["adjclose"][0]["adjclose"]
        df = pd.DataFrame(
            {"adjclose": adj, "close": quote["close"], "volume": quote["volume"]}, index=idx
        )
    except Exception:
        return None
    df = df.apply(pd.to_numeric, errors="coerce").astype("float64").dropna(how="all")
    return df if not df.empty else None


def _from_stooq(ticker: str) -> pd.DataFrame | None:
    """Daily close/volume from Stooq CSV (UNADJUSTED: ``adjclose`` := ``close``), or ``None``."""
    r = _fetch(_STOOQ_URL.format(ticker=ticker.lower()))
    if r is None:
        return None
    try:
        raw = pd.read_csv(io.StringIO(r.text))
    except Exception:
        return None
    if raw.empty or "Date" not in raw.columns or "Close" not in raw.columns:
        return None  # Stooq returns "No data"/an HTML error page for unknown symbols
    idx = pd.to_datetime(raw["Date"], errors="coerce").dt.normalize()
    df = pd.DataFrame(
        {
            "adjclose": pd.to_numeric(raw["Close"], errors="coerce"),
            "close": pd.to_numeric(raw["Close"], errors="coerce"),
            "volume": pd.to_numeric(raw.get("Volume"), errors="coerce"),
        },
        index=idx,
    )
    df = df[df.index.notna()].astype("float64").dropna(how="all")
    return df if not df.empty else None


def load_ohlcv_recovered(
    ticker: str, refresh: bool = False, sources: tuple[str, ...] = ("yahoo", "stooq")
) -> pd.DataFrame | None:
    """Recovered daily OHLCV for one delisted spinoff symbol, cached; ``None`` if unrecoverable.

    Same shape as :func:`strategies.spinoff.backtest.load_ohlcv` (columns
    ``[adjclose, close, volume]``, ``DatetimeIndex``) so the basket consumes it unchanged.
    Tries ``sources`` in order (Yahoo, then Stooq) and caches the first hit under
    ``data/prices/spinoff_delisted/{TICKER}.parquet``. A miss caches an empty frame so the CIK
    is not refetched on the next matrix shard; the ``source`` is stamped in ``df.attrs``.
    """
    if not ticker:
        return None
    cache = _CACHE / f"{_safe(ticker)}.parquet"
    if cache.exists() and not refresh:
        df = pd.read_parquet(cache)
        return df if not df.empty else None

    df, source = None, "none"
    if "yahoo" in sources:
        df = _from_yahoo(ticker)
        if df is not None:
            source = "yahoo"
    if df is None and "stooq" in sources:
        df = _from_stooq(ticker)
        if df is not None:
            source = "stooq"

    out = df if df is not None else pd.DataFrame(columns=["adjclose", "close", "volume"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    if out.empty:
        return None
    out.attrs["source"] = source
    return out
