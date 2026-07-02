"""SEC EDGAR N-PORT (open-end fund holdings + flows) parser for the cloud-build matrix.

Public N-PORT (form type ``NPORT-P``) only exists from ~2019Q4 onward, and only the THIRD
month of each fund fiscal quarter is disseminated (the first two months are the non-public
``NPORT-NP``). So this pipeline yields QUARTERLY fund holdings, net assets, and directly
reported monthly flows (``sales`` / ``redemption`` / ``reinvestment``) from late 2019 to
present. Every ``NPORT-P`` in the quarterly index is fetched; funds with no long
common-equity holding or below the net-asset floor are dropped at parse time to bound the
cache. Point-in-time: every row carries the index ``filing_date``, never the quarter-end.

Reuses the hardened EDGAR fetch spine from :mod:`form4` (session, throttle, retry/backoff,
quarter enumeration) and the XML helpers from :mod:`edgar` (namespace strip, flat-tag text,
CUSIP normalization), so fair-access behavior and parsing match the other pipelines.
"""

from __future__ import annotations

import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

# Reuse the form4 spine (SEC-fair-access session, thread-safe throttle, retry/backoff GET,
# quarter enumerator) and the edgar XML helpers verbatim. The flat-import fallback lets this
# module run unchanged in the public cloud-build matrix repo, where form4.py / edgar.py sit
# alongside nport.py rather than under libs.data.academic.
try:  # repo layout
    from libs.data.academic.form4 import (  # noqa: F401
        _BACKOFF_CAP,
        _Throttle,
        _get,
        _quarters,
        _session,
    )
    from libs.data.academic.edgar import _norm_cusip, _strip_ns, _t
except ImportError:  # flat checkout (public cloud-build matrix repo)
    from form4 import (  # type: ignore  # noqa: F401
        _BACKOFF_CAP,
        _Throttle,
        _get,
        _quarters,
        _session,
    )
    from edgar import _norm_cusip, _strip_ns, _t  # type: ignore

log = logging.getLogger("nport")

# git-ignored parquet cache (AGENTS.md: cache pulled data, never commit it). Overridable via
# NPORT_CACHE_DIR so the same module works in the repo layout and in a flat checkout.
_CACHE = Path(
    os.environ.get(
        "NPORT_CACHE_DIR",
        Path(__file__).resolve().parents[3] / "data" / "edgar" / "nport",
    )
)
_IDX_CACHE = _CACHE / "_index"

_WORKERS = int(os.environ.get("NPORT_WORKERS", "10"))
_LOG_EVERY = int(os.environ.get("NPORT_LOG_EVERY", "200"))
# Refuse to cache a quarter that lost more than this fraction of its fetches (silent-corruption
# guard; a holed parquet that still exits 0 must not masquerade as a complete build).
_MAX_FAIL_FRAC = float(os.environ.get("NPORT_MAX_FAIL_FRAC", "0.05"))
# Net-asset floor (whole USD) applied at parse time to bound the cache to funds large enough
# to exert measurable pressure. The signal may raise this further; it never lowers it.
_MIN_NET_ASSETS = float(os.environ.get("NPORT_MIN_NET_ASSETS", "10000000"))  # $10M

# Headline run: original NPORT-P only. Amendments restate and complicate filing-date PIT logic.
INCLUDE_AMENDMENTS = os.environ.get("NPORT_AMENDMENTS", "0") not in ("0", "false", "False")
_NPORT_TYPES = {"NPORT-P", "NPORT-P/A"} if INCLUDE_AMENDMENTS else {"NPORT-P"}

# First public NPORT-P period is 2019-09-30 (filed Nov 2019); start the quarter before so an
# early filer is not missed. Everything before this simply has no public N-PORT holdings.
DEFAULT_START = "2019-07-01"

# Pull the primary N-PORT document out of the SGML full-submission text. It is wrapped as
# <XML><edgarSubmission ...>...</edgarSubmission></XML>; match the edgarSubmission block
# directly (tolerating an optional namespace prefix), same contract the 13F info table uses.
_SUBMISSION_RE = re.compile(
    r"<(?:\w+:)?edgarSubmission\b.*?</(?:\w+:)?edgarSubmission>", re.DOTALL
)
_PERIOD_RE = re.compile(r"CONFORMED PERIOD OF REPORT:\s*(\d{8})")

# Fund-level table: one row per (series, quarter). Flows/returns/net-assets describe the
# fiscal quarter ending at period_of_report; the holdings snapshot is as of that date.
FUND_COLUMNS = [
    "series_id",
    "reg_cik",
    "fund_name",
    "filing_date",
    "period_of_report",
    "net_assets",
    "redemption",     # quarterly sum of monthly redemptions (outflows), whole USD
    "sales",          # quarterly sum of monthly sales (creations / inflows), whole USD
    "reinvestment",   # quarterly sum of monthly dividend reinvestment, whole USD
    "ret_m1",         # class-averaged monthly total return, % (first month of quarter)
    "ret_m2",
    "ret_m3",
    "n_eq_holdings",
    "is_amendment",
]

# Holdings table: one row per (series, quarter, equity position). Long common-equity only.
HOLDING_COLUMNS = [
    "series_id",
    "filing_date",
    "period_of_report",
    "cusip",
    "name_of_issuer",
    "shares",
    "val_usd",
    "pct_val",
]


def _f(x) -> float | None:
    """Parse a possibly-messy numeric string to float; None on failure."""
    if x is None:
        return None
    try:
        return float(str(x).replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def _nport_index(session, throttle: _Throttle, year: int, qtr: int) -> pd.DataFrame:
    """NPORT-P rows from one quarterly EDGAR full-index ``form.idx`` (cached per quarter).

    Columns: ``cik`` (int, the registrant), ``name``, ``filing_date``, ``form_type``,
    ``path`` (the full-submission .txt URL path).
    """
    cache = _IDX_CACHE / f"{year}Q{qtr}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{qtr}/form.idx"
    log.info("%dQ%d: fetching quarterly index %s", year, qtr, url)
    txt = _get(session, throttle, url)
    if txt is None:
        raise RuntimeError(f"could not fetch form.idx for {year} QTR{qtr}")
    rows = []
    for line in txt.splitlines():
        # Fixed-ish layout: "<type> <company> <cik> <date> <path>". The NPORT-P type tokens
        # have no internal spaces, so split() keeps col 0 whole (same contract as 13F/Form 10).
        parts = line.split()
        if not parts or parts[0] not in _NPORT_TYPES:
            continue
        path = parts[-1]
        date = parts[-2]
        cik = parts[-3]
        if not (cik.isdigit() and re.match(r"\d{4}-\d{2}-\d{2}", date)):
            continue
        name = " ".join(parts[1:-3])
        rows.append((int(cik), name, date, parts[0], path))
    df = pd.DataFrame(rows, columns=["cik", "name", "filing_date", "form_type", "path"])
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    log.info("%dQ%d: index lists %d NPORT-P filings", year, qtr, len(df))
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def _parse_nport(txt: str) -> tuple[dict | None, list[dict]]:
    """Parse one NPORT-P full-submission .txt into (fund_dict, equity_holding_rows).

    ``fund_dict`` carries series id, name, period, net assets, the three monthly flows
    (summed to quarterly), and the class-averaged monthly total returns. Holdings are the
    long common-equity (``assetCat == 'EC'``, ``payoffProfile == 'Long'``, share units)
    positions with a usable CUSIP. Returns ``(None, [])`` when the doc is missing/unparseable
    or has no qualifying equity holding.
    """
    m = _SUBMISSION_RE.search(txt)
    if m is None:
        return None, []
    try:
        root = ET.fromstring(_strip_ns(m.group(0)))
    except ET.ParseError:
        return None, []

    gen = root.find(".//genInfo")
    fund = root.find(".//fundInfo")
    if gen is None or fund is None:
        return None, []

    net_assets = _f(_t(fund, "netAssets"))
    if net_assets is None or net_assets <= 0:
        return None, []

    # Equity holdings first: if a fund holds no long common equity, it is not in scope and we
    # skip the whole filing (keeps bond / money-market / derivative funds out of the cache).
    holdings: list[dict] = []
    for sec in root.iter("invstOrSec"):
        if _t(sec, "assetCat") != "EC":  # common equity only
            continue
        if (_t(sec, "payoffProfile") or "Long") != "Long":  # long positions only
            continue
        cusip = _norm_cusip(_t(sec, "cusip"))
        shares = _f(_t(sec, "balance"))
        val = _f(_t(sec, "valUSD"))
        if cusip is None or val is None:
            continue
        holdings.append(
            {
                "cusip": cusip,
                "name_of_issuer": _t(sec, "name"),
                "shares": shares,
                "val_usd": val,
                "pct_val": _f(_t(sec, "pctVal")),
            }
        )
    if not holdings:
        return None, []

    # Flows: sum the three public months of the fiscal quarter (whole USD).
    redemption = sales = reinvestment = 0.0
    for tag in ("mon1Flow", "mon2Flow", "mon3Flow"):
        el = fund.find(tag)
        if el is None:
            continue
        redemption += _f(el.get("redemption")) or 0.0
        sales += _f(el.get("sales")) or 0.0
        reinvestment += _f(el.get("reinvestment")) or 0.0

    # Monthly total returns: average across share classes (classes differ only by fee load).
    ret = {"ret_m1": [], "ret_m2": [], "ret_m3": []}
    for mr in fund.iter("monthlyTotReturn"):
        for i, key in enumerate(("ret_m1", "ret_m2", "ret_m3"), start=1):
            v = _f(mr.get(f"rtn{i}"))
            if v is not None:
                ret[key].append(v)
    ret_avg = {k: (sum(v) / len(v) if v else None) for k, v in ret.items()}

    fund_dict = {
        "series_id": _t(gen, "seriesId"),
        "reg_cik": _f(_t(gen, "regCik")),
        "fund_name": _t(gen, "seriesName") or _t(gen, "regName"),
        "net_assets": net_assets,
        "redemption": redemption,
        "sales": sales,
        "reinvestment": reinvestment,
        "n_eq_holdings": len(holdings),
        **ret_avg,
    }
    return fund_dict, holdings


def _fetch_filing(session, throttle: _Throttle, row) -> tuple[bool, dict | None, list[dict]]:
    """Fetch+parse one NPORT-P filing. Returns (fetched_ok, fund_dict|None, holding_rows)."""
    url = f"https://www.sec.gov/Archives/{row.path}"
    txt = _get(session, throttle, url)
    if txt is None:
        return False, None, []
    fund, holdings = _parse_nport(txt)
    if fund is None:
        return True, None, []  # fetched fine; just not an in-scope equity fund
    if fund["net_assets"] < _MIN_NET_ASSETS:
        return True, None, []  # below the parse-time net-asset floor
    pm = _PERIOD_RE.search(txt)
    period = pd.to_datetime(pm.group(1), format="%Y%m%d") if pm else pd.NaT
    fund["reg_cik"] = int(row.cik)
    fund["filing_date"] = row.filing_date
    fund["period_of_report"] = period
    fund["is_amendment"] = str(row.form_type).endswith("/A")
    # Fall back to the registrant CIK as the fund key when a series id is absent (rare).
    if not fund["series_id"]:
        fund["series_id"] = f"CIK{int(row.cik)}"
    for h in holdings:
        h["series_id"] = fund["series_id"]
        h["filing_date"] = row.filing_date
        h["period_of_report"] = period
    return True, fund, holdings


def _load_quarter(
    session,
    throttle: _Throttle,
    year: int,
    qtr: int,
    refresh: bool = False,
    workers: int = _WORKERS,
) -> pd.DataFrame:
    """Parse all in-scope NPORT-P filings in one quarter; cache holdings + fund tables.

    Two parquet products per quarter under ``data/edgar/nport/``: ``<Q>.parquet`` (equity
    holdings, long format) and ``<Q>_funds.parquet`` (fund-level flows/returns/net-assets).
    Returns the holdings frame. Fetches concurrently; the shared throttle keeps the aggregate
    request rate under SEC's fair-access ceiling. Refuses to cache a quarter that lost more
    than ``_MAX_FAIL_FRAC`` of its fetches.
    """
    cache = _CACHE / f"{year}Q{qtr}.parquet"
    fund_cache = _CACHE / f"{year}Q{qtr}_funds.parquet"
    if cache.exists() and fund_cache.exists() and not refresh:
        return pd.read_parquet(cache)

    idx = _nport_index(session, throttle, year, qtr)
    rows = list(idx.itertuples(index=False))
    log.info("%dQ%d: fetching %d NPORT-P filings with %d workers", year, qtr, len(rows), workers)

    hold_records: list[dict] = []
    fund_records: list[dict] = []
    n_fetch = n_fail = n_scope = 0
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for ok, fund, holds in ex.map(lambda r: _fetch_filing(session, throttle, r), rows):
            n_fetch += 1
            if not ok:
                n_fail += 1
                continue
            if fund is not None:
                n_scope += 1
                fund_records.append(fund)
                hold_records.extend(holds)
            if n_fetch % _LOG_EVERY == 0:
                elapsed = time.monotonic() - t0
                rate = n_fetch / elapsed if elapsed else 0.0
                remaining = (len(rows) - n_fetch) / rate if rate else 0.0
                log.info(
                    "%dQ%d: %d/%d filings (%.0f%%) | %.1f/s | %d equity funds | %d rows | "
                    "%d failed | ~%.0f min left",
                    year, qtr, n_fetch, len(rows), 100 * n_fetch / max(len(rows), 1),
                    rate, n_scope, len(hold_records), n_fail, remaining / 60,
                )

    funds = pd.DataFrame(fund_records, columns=FUND_COLUMNS)
    holds = pd.DataFrame(hold_records, columns=HOLDING_COLUMNS)
    log.info(
        "%dQ%d: DONE -> %d equity funds, %d holding rows from %d filings (%d failed) in %.0fs",
        year, qtr, len(funds), len(holds), n_fetch, n_fail, time.monotonic() - t0,
    )
    frac = n_fail / n_fetch if n_fetch else 0.0
    if frac > _MAX_FAIL_FRAC:
        raise RuntimeError(
            f"{year}Q{qtr}: {n_fail}/{n_fetch} fetches failed ({frac:.1%} > "
            f"{_MAX_FAIL_FRAC:.0%} cap); refusing to cache an incomplete quarter"
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    funds.to_parquet(fund_cache)
    holds.to_parquet(cache)
    return holds


def fetch_nport_filings(
    start: str = DEFAULT_START,
    end: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Long-only equity holdings from public NPORT-P, keyed on filing date (systematic).

    Enumerates the quarterly full-index across ``[start, end]``, parses each NPORT-P, and
    returns the concatenated holdings frame (:data:`HOLDING_COLUMNS`). Fund-level flows are
    cached alongside; use :func:`load_cached_funds` to read them. Cached per quarter under
    ``data/edgar/nport/`` so the (long) build resumes without refetching.
    """
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    start_ts = pd.Timestamp(start)
    session = _session()
    throttle = _Throttle()
    frames = []
    for year, qtr in _quarters(start_ts, end_ts):
        frames.append(_load_quarter(session, throttle, year, qtr, refresh))
    session.close()
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=HOLDING_COLUMNS)
    if not out.empty:
        out = out[(out["filing_date"] >= start_ts) & (out["filing_date"] <= end_ts)]
        out = out.sort_values("filing_date").reset_index(drop=True)
    return out


def _load_cached(kind: str, start: str, end: str | None, columns: list[str] | None) -> pd.DataFrame:
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    start_ts = pd.Timestamp(start)
    wanted = {f"{y}Q{q}" for (y, q) in _quarters(start_ts, end_ts)}
    suffix = "_funds.parquet" if kind == "funds" else ".parquet"
    files = sorted(
        f for f in _CACHE.glob(f"*Q*{suffix}")
        if f.stem.replace("_funds", "") in wanted and (kind == "funds") == f.stem.endswith("_funds")
    )
    if not files:
        raise FileNotFoundError(
            f"no cached NPORT-P {kind} quarters under {_CACHE}; run build_data.py (or the "
            "edgar-fetch matrix) first"
        )
    parts = [pd.read_parquet(f, columns=columns) for f in files]
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values("filing_date").reset_index(drop=True)


def load_cached_holdings(
    start: str = DEFAULT_START, end: str | None = None, columns: list[str] | None = None
) -> pd.DataFrame:
    """Read the per-quarter NPORT-P holdings cache (no network), optional column pruning."""
    return _load_cached("holdings", start, end, columns)


def load_cached_funds(
    start: str = DEFAULT_START, end: str | None = None, columns: list[str] | None = None
) -> pd.DataFrame:
    """Read the per-quarter NPORT-P fund-level (flows/returns/net-assets) cache (no network)."""
    return _load_cached("funds", start, end, columns)
