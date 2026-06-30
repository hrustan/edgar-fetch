"""SEC EDGAR 13F-HR ingestion + CUSIP->ticker resolution (free, public data).

Enumerates 13F-HR institutional-holdings filings from the EDGAR quarterly full-index,
fetches each full-submission ``.txt`` (which embeds the structured information-table XML),
and extracts the per-holding rows (issuer, CUSIP, value, shares). Output is one row per
holding, keyed on the FILING date (point-in-time).

Design notes:
  - Systematic universe: every 13F-HR filer is parsed (no CIK restriction at fetch time);
    any AUM floor is applied downstream, from each filing's own holdings. This is why the
    build is large and shardable across one quarter per runner.
  - XML era only: SEC mandated the structured information-table XML from mid-2013. Pre-2013
    filings are brittle free-text/HTML tables and are intentionally unsupported; the default
    start is 2013-07-01.
  - The 2023 value-scale break: SEC switched the ``<value>`` column from THOUSANDS of dollars
    to WHOLE dollars for filings on/after 2023-01-03 (with a sloppy transition). ``value_usd``
    is normalized to whole dollars per filing via a median-implied-price cross-check so AUM
    and weights are unit-consistent across the break.
  - Point-in-time: every row carries ``filing_date`` from the index, never
    ``period_of_report`` (the quarter-end the holdings describe).
  - Amendments (``13F-HR/A``) are excluded by default (set THIRTEENF_AMENDMENTS=1 to include).

Reuses the hardened EDGAR fetch spine from ``form4.py`` (session, global throttle,
retry/backoff, quarter enumeration) so SEC fair-access behavior is identical.

Respects SEC fair-access limits: descriptive User-Agent (set ``SEC_USER_AGENT``) and a
throttled request rate under 10/s.
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

# Reuse the form4 spine verbatim: the SEC-fair-access session, the thread-safe global
# request spacer, the retry/backoff GET, and the quarter enumerator are already hardened
# against shared-IP 429 bans. Only the index filter and the per-filing parse differ here.
# The flat-import fallback lets this module run unchanged in the public cloud-build matrix
# repo (where form4.py sits alongside edgar.py, not under libs.data.academic).
try:  # repo layout
    from libs.data.academic.form4 import (  # noqa: F401
        _BACKOFF_CAP,
        _Throttle,
        _get,
        _quarters,
        _session,
    )
except ImportError:  # flat checkout (public cloud-build matrix repo)
    from form4 import (  # type: ignore  # noqa: F401
        _BACKOFF_CAP,
        _Throttle,
        _get,
        _quarters,
        _session,
    )

log = logging.getLogger("edgar13f")

# git-ignored parquet cache (AGENTS.md: cache pulled data, never commit it). Overridable via
# THIRTEENF_CACHE_DIR so the same module works in the repo layout and in a flat checkout
# (e.g. a public cloud-build repo) without edits.
_CACHE = Path(
    os.environ.get(
        "THIRTEENF_CACHE_DIR",
        Path(__file__).resolve().parents[3] / "data" / "edgar" / "thirteen_f",
    )
)
_IDX_CACHE = _CACHE / "_index"

_WORKERS = int(os.environ.get("THIRTEENF_WORKERS", "10"))
_LOG_EVERY = int(os.environ.get("THIRTEENF_LOG_EVERY", "200"))
# Refuse to cache a quarter that lost more than this fraction of its fetches (silent-corruption
# guard; a holed parquet that still exits 0 must not masquerade as a complete build).
_MAX_FAIL_FRAC = float(os.environ.get("THIRTEENF_MAX_FAIL_FRAC", "0.05"))

# Headline run: original 13F-HR only. Amendments restate and complicate filing-date PIT logic.
INCLUDE_AMENDMENTS = os.environ.get("THIRTEENF_AMENDMENTS", "0") not in ("0", "false", "False")
_13F_TYPES = {"13F-HR", "13F-HR/A"} if INCLUDE_AMENDMENTS else {"13F-HR"}

DEFAULT_START = "2013-07-01"  # structured information-table XML era

# Pull the embedded information table out of the SGML full-submission text. Tolerates a
# namespace prefix (post-2013 info tables are namespaced, unlike Form 4's bare element).
_INFOTABLE_RE = re.compile(
    r"<(?:\w+:)?informationTable\b.*?</(?:\w+:)?informationTable>", re.DOTALL
)
# Period of report from the SGML header (always present, unambiguous YYYYMMDD).
_PERIOD_RE = re.compile(r"CONFORMED PERIOD OF REPORT:\s*(\d{8})")

COLUMNS = [
    "filer_cik",
    "filer_name",
    "filing_date",
    "period_of_report",
    "cusip",
    "name_of_issuer",
    "value_usd",
    "shares",
    "is_amendment",
]


def _strip_ns(xml: str) -> str:
    """Drop XML namespace declarations, prefixed attributes, and tag prefixes.

    Leaves ElementTree tags bare. The prefixed-attribute strip (e.g. ``xsi:schemaLocation``)
    is essential: once the matching ``xmlns:xsi`` declaration is removed, a surviving
    ``xsi:`` prefix on an attribute is an unbound prefix and ElementTree refuses to parse.
    """
    xml = re.sub(r'\sxmlns(:\w+)?="[^"]*"', "", xml)
    xml = re.sub(r'\s\w+:\w+="[^"]*"', "", xml)
    xml = re.sub(r"<(/?)\w+:", r"<\1", xml)
    return xml


def _t(el, tag: str) -> str | None:
    """Text of a flat ``<tag>...</tag>`` child (info-table fields are not nested)."""
    if el is None:
        return None
    node = el.find(tag)
    if node is None:
        return None
    return (node.text or "").strip() or None


def _norm_cusip(cusip: str | None) -> str | None:
    """Upper-case, strip, left-pad to 9 chars. EDGAR CUSIPs vary in width/case."""
    if not cusip:
        return None
    c = str(cusip).strip().upper()
    c = re.sub(r"\s+", "", c)
    if not c:
        return None
    return c.zfill(9) if len(c) < 9 else c


def _13f_index(session, throttle: _Throttle, year: int, qtr: int) -> pd.DataFrame:
    """13F-HR rows from one quarterly EDGAR full-index ``form.idx``.

    Columns: ``filer_cik`` (int, the institutional MANAGER cik), ``filer_name``,
    ``filing_date``, ``path`` (the full-submission .txt URL path). Cached raw per quarter.
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
        # Fixed-ish layout: "<type> <company> <cik> <date> <path>"; type is col 0 and the
        # 13F form type token has no internal spaces, so split() keeps it whole.
        parts = line.split()
        if not parts or parts[0] not in _13F_TYPES:
            continue
        path = parts[-1]
        date = parts[-2]
        cik = parts[-3]
        if not (cik.isdigit() and re.match(r"\d{4}-\d{2}-\d{2}", date)):
            continue
        name = " ".join(parts[1:-3])
        rows.append((int(cik), name, date, path, parts[0].endswith("/A")))
    df = pd.DataFrame(
        rows, columns=["filer_cik", "filer_name", "filing_date", "path", "is_amendment"]
    )
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    log.info("%dQ%d: index lists %d 13F-HR filings", year, qtr, len(df))
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def _parse_info_table(txt: str) -> list[dict]:
    """Extract long-only equity holding rows from one full-submission .txt.

    Keeps rows reported as shares (``sshPrnamtType == 'SH'``) that are not option lines
    (no ``putCall``). ``value`` is returned RAW (in filing units); the per-filing scale
    normalization to whole dollars happens in :func:`_normalize_value_scale`.
    """
    m = _INFOTABLE_RE.search(txt)
    if m is None:
        return []
    try:
        root = ET.fromstring(_strip_ns(m.group(0)))
    except ET.ParseError:
        return []

    out = []
    for it in root.iter("infoTable"):
        sh = it.find("shrsOrPrnAmt")
        sh_type = _t(sh, "sshPrnamtType")
        put_call = _t(it, "putCall")
        if sh_type != "SH" or put_call:  # long equity only; drop PRN debt and option lines
            continue
        cusip = _norm_cusip(_t(it, "cusip"))
        value = _t(it, "value")
        shares = _t(sh, "sshPrnamt")
        if cusip is None or value is None:
            continue
        try:
            value_f = float(value.replace(",", ""))
            shares_f = float(shares.replace(",", "")) if shares else None
        except ValueError:
            continue
        out.append(
            {
                "cusip": cusip,
                "name_of_issuer": _t(it, "nameOfIssuer"),
                "value": value_f,
                "shares": shares_f,
            }
        )
    return out


def _normalize_value_scale(rows: list[dict], filing_date) -> float:
    """Set ``value_usd`` (whole dollars) on each row; return the scale applied.

    Decides ONCE per filing (the unit is uniform within a filing). Primary signal is the
    median implied price ``value / shares`` over the filing's share rows: in the thousands
    era this lands near price/1000 (well under $1); in the whole-dollar era it is the real
    per-share price (>= $1 for any normal book). Falls back to the 2023-01-01 date cutoff
    when shares are unusable.
    """
    prices = [r["value"] / r["shares"] for r in rows if r.get("shares")]
    if prices:
        prices.sort()
        med = prices[len(prices) // 2]
        scale = 1.0 if med >= 1.0 else 1000.0
    else:
        scale = 1.0 if pd.Timestamp(filing_date) >= pd.Timestamp("2023-01-01") else 1000.0
    for r in rows:
        r["value_usd"] = r["value"] * scale
    return scale


def _fetch_filing(session, throttle: _Throttle, row) -> tuple[bool, list[dict]]:
    """Fetch+parse one 13F-HR filing. Returns (fetched_ok, holding_rows)."""
    url = f"https://www.sec.gov/Archives/{row.path}"
    txt = _get(session, throttle, url)
    if txt is None:
        return False, []
    holdings = _parse_info_table(txt)
    if not holdings:
        return True, []  # fetched fine; just no qualifying equity rows
    _normalize_value_scale(holdings, row.filing_date)
    pm = _PERIOD_RE.search(txt)
    period = pd.to_datetime(pm.group(1), format="%Y%m%d") if pm else pd.NaT
    for h in holdings:
        h["filer_cik"] = int(row.filer_cik)
        h["filer_name"] = row.filer_name
        h["filing_date"] = row.filing_date
        h["period_of_report"] = period
        h["is_amendment"] = bool(row.is_amendment)
        h.pop("value", None)
    return True, holdings


def _load_quarter(
    session,
    throttle: _Throttle,
    year: int,
    qtr: int,
    refresh: bool = False,
    workers: int = _WORKERS,
) -> pd.DataFrame:
    """Parse all 13F-HR holding rows filed in one quarter (cached to parquet).

    No CIK restriction: every 13F-HR in the index is fetched (the systematic universe; the
    AUM floor is applied later in the signal). Filings are fetched concurrently; the shared
    throttle keeps the aggregate request rate under SEC's fair-access ceiling.
    """
    cache = _CACHE / f"{year}Q{qtr}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    idx = _13f_index(session, throttle, year, qtr)
    rows = list(idx.itertuples(index=False))
    log.info("%dQ%d: fetching %d 13F-HR filings with %d workers", year, qtr, len(rows), workers)

    records: list[dict] = []
    n_fetch = n_fail = 0
    n_thousands = n_dollars = 0
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for ok, recs in ex.map(lambda r: _fetch_filing(session, throttle, r), rows):
            n_fetch += 1
            if not ok:
                n_fail += 1
            else:
                if recs:
                    # value_usd / shares >= 1 only when whole-dollar scale was applied.
                    sample = next((r for r in recs if r.get("shares")), None)
                    if sample is not None:
                        if sample["value_usd"] / sample["shares"] >= 1.0:
                            n_dollars += 1
                        else:
                            n_thousands += 1
                records.extend(recs)
            if n_fetch % _LOG_EVERY == 0:
                elapsed = time.monotonic() - t0
                rate = n_fetch / elapsed if elapsed else 0.0
                remaining = (len(rows) - n_fetch) / rate if rate else 0.0
                log.info(
                    "%dQ%d: %d/%d filings (%.0f%%) | %.1f/s | %d rows | %d failed | ~%.0f min left",
                    year, qtr, n_fetch, len(rows), 100 * n_fetch / max(len(rows), 1),
                    rate, len(records), n_fail, remaining / 60,
                )

    df = pd.DataFrame(records, columns=COLUMNS)
    if not df.empty:
        df = df.dropna(subset=["cusip", "filer_cik", "filing_date"])
    df.attrs["n_filings_fetched"] = n_fetch
    df.attrs["n_fetch_failed"] = n_fail
    log.info(
        "%dQ%d: DONE -> %d holding rows from %d filings (%d failed) in %.0fs; "
        "value-scale: %d filings thousands, %d whole-dollars",
        year, qtr, len(df), n_fetch, n_fail, time.monotonic() - t0, n_thousands, n_dollars,
    )
    frac = n_fail / n_fetch if n_fetch else 0.0
    if frac > _MAX_FAIL_FRAC:
        raise RuntimeError(
            f"{year}Q{qtr}: {n_fail}/{n_fetch} fetches failed ({frac:.1%} > "
            f"{_MAX_FAIL_FRAC:.0%} cap); refusing to cache an incomplete quarter"
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def fetch_13f_filings(
    cik_list=None,
    start: str = DEFAULT_START,
    end: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Long-only 13F-HR holdings from EDGAR, keyed on filing date (systematic universe).

    ``cik_list`` is accepted for signature compatibility but ignored by default: the
    universe is systematic (all filers; the AUM floor is applied in the signal). Enumerates
    the quarterly full-index across ``[start, end]``, parses each 13F-HR information table,
    and returns a frame with :data:`COLUMNS`. ``value_usd`` is normalized to whole dollars.
    Results are cached per quarter under ``data/edgar/thirteen_f/`` so the (long) build
    resumes without refetching.
    """
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    start_ts = pd.Timestamp(start)
    session = _session()
    throttle = _Throttle()
    frames = []
    for year, qtr in _quarters(start_ts, end_ts):
        frames.append(_load_quarter(session, throttle, year, qtr, refresh))
    session.close()

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    if cik_list is not None and not out.empty:
        out = out[out["filer_cik"].isin(set(cik_list))]
    if not out.empty:
        out = out[(out["filing_date"] >= start_ts) & (out["filing_date"] <= end_ts)]
        out = out.sort_values("filing_date").reset_index(drop=True)
    return out


# --------------------------------------------------------------------------------------
# CUSIP -> ticker resolution (OpenFIGI free mapping API), applied only to survivor CUSIPs
# --------------------------------------------------------------------------------------
_OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"


def _openfigi_map(cusips: list[str]) -> pd.DataFrame:
    """Map CUSIPs to primary US equity tickers via OpenFIGI. Unresolved -> ticker NaN.

    Returns ``[cusip, ticker]`` with one row per input CUSIP (so unresolved CUSIPs are
    cached too and not re-queried). Honors an optional ``OPENFIGI_API_KEY`` (env) for a
    higher rate limit / larger batches; without a key it stays under the ~25 req/min free
    ceiling.
    """
    import requests

    api_key = os.environ.get("OPENFIGI_API_KEY", "").strip()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-OPENFIGI-APIKEY"] = api_key
    batch = 100 if api_key else 10
    min_interval = 0.3 if api_key else 2.5  # spacing to respect the per-minute ceiling

    session = requests.Session()
    session.headers.update(headers)
    results: dict[str, str | None] = {}
    last = 0.0
    for i in range(0, len(cusips), batch):
        chunk = cusips[i : i + batch]
        jobs = [{"idType": "ID_CUSIP", "idValue": c, "exchCode": "US"} for c in chunk]
        for attempt in range(6):
            dt = time.monotonic() - last
            if dt < min_interval:
                time.sleep(min_interval - dt)
            last = time.monotonic()
            try:
                r = session.post(_OPENFIGI_URL, json=jobs, timeout=60)
            except Exception as e:
                log.warning("OpenFIGI network error (attempt %d): %s", attempt + 1, e)
                time.sleep(min(_BACKOFF_CAP, 2 ** attempt))
                continue
            if r.status_code == 429:
                delay = min(_BACKOFF_CAP, 5.0 * (attempt + 1))
                log.warning("OpenFIGI 429, backing off %.0fs", delay)
                time.sleep(delay)
                continue
            if r.status_code != 200:
                log.warning("OpenFIGI HTTP %d on batch %d", r.status_code, i // batch)
                break
            for c, item in zip(chunk, r.json()):
                ticker = None
                for d in item.get("data", []) or []:
                    if d.get("marketSector") == "Equity" and d.get("ticker"):
                        ticker = str(d["ticker"]).upper()
                        break
                results[c] = ticker
            break
    session.close()
    for c in cusips:  # ensure every queried cusip is recorded (even on a broken batch)
        results.setdefault(c, None)
    return pd.DataFrame({"cusip": list(results), "ticker": list(results.values())})


def resolve_cusips(cusips, refresh: bool = False) -> pd.DataFrame:
    """Resolve CUSIPs to US equity tickers, cached to parquet. Returns ``[cusip, ticker]``.

    Only the (small) set of CUSIPs that survive the conviction+consensus screen is ever
    passed here, so total OpenFIGI volume is tiny and fully cacheable across the backtest.
    Unresolved CUSIPs (delisted / acquired / not common equity) come back with ``ticker``
    NaN and are part of the flagged survivorship hole.
    """
    want = sorted({_norm_cusip(c) for c in cusips if _norm_cusip(c)})
    cache = _CACHE / "cusip_ticker.parquet"
    have = (
        pd.read_parquet(cache)
        if cache.exists() and not refresh
        else pd.DataFrame(columns=["cusip", "ticker"])
    )
    todo = [c for c in want if c not in set(have["cusip"])]
    if todo:
        log.info("resolving %d new CUSIPs via OpenFIGI", len(todo))
        new = _openfigi_map(todo)
        have = pd.concat([have, new], ignore_index=True).drop_duplicates("cusip")
        cache.parent.mkdir(parents=True, exist_ok=True)
        have.to_parquet(cache)
    return have[have["cusip"].isin(want)].reset_index(drop=True)
