"""SEC EDGAR Form 4 open-market-purchase ingestion (free, public data).

Enumerates Form 4 filings from the EDGAR quarterly full-index, fetches each full-submission
``.txt`` (which embeds the structured ``ownershipDocument`` XML), and extracts open-market
purchases (Transaction Code "P"). Output is one row per purchase, keyed on the filing date.

Design notes:
  - There is no official structured bulk Form 4 dataset at SEC; the transaction code lives
    only inside each filing's XML, so a full parse is per-filing. The fetch is bounded to
    issuers in SEC's ``company_tickers.json`` (CIK->ticker map for currently listed names),
    which both resolves issuer->ticker and restricts to names with a tradeable symbol.
  - Point-in-time: every row carries the Form 4 ``filing_date`` (from the index), never the
    transaction date.
  - Excludes pure 10% holders who are not also an officer or director.

Respects SEC fair-access limits: descriptive User-Agent (set ``SEC_USER_AGENT``) and a
throttled request rate under 10/s.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

# Module logger. The library never configures handlers (so unit tests stay quiet); the
# entrypoint (build_data.py) calls logging.basicConfig to stream INFO to stdout, which is
# what shows up live in a GitHub Actions runner log.
log = logging.getLogger("form4")

# git-ignored parquet cache (AGENTS.md: cache pulled data, never commit it). The location
# is overridable via FORM4_CACHE_DIR so the same module works in the repo layout and in a
# flat checkout (e.g. the public cloud-build repo) without edits.
_CACHE = Path(
    os.environ.get(
        "FORM4_CACHE_DIR", Path(__file__).resolve().parents[3] / "data" / "edgar" / "form4"
    )
)
_IDX_CACHE = _CACHE / "_index"

# SEC fair-access: descriptive User-Agent with contact, throttled to <=10 req/s.
_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT", "chakravarti-research harrisonrustan@gmail.com"
)
_MIN_INTERVAL = 0.11  # seconds between requests globally (~9/s, under SEC's 10/s ceiling)
_RETRIES = 4
_WORKERS = 10  # concurrent fetchers; the shared throttle still caps the aggregate rate
_LOG_EVERY = int(os.environ.get("FORM4_LOG_EVERY", "200"))  # heartbeat cadence (filings)
_LOG_PURCHASES = os.environ.get("FORM4_LOG_PURCHASES", "1") not in ("0", "false", "False")

_FORM4_TYPES = {"4"}  # original Form 4 only; "4/A" amendments restate and are excluded

# Pull the embedded ownership XML out of the SGML full-submission text.
_OWNERSHIP_RE = re.compile(r"<ownershipDocument>.*?</ownershipDocument>", re.DOTALL)

COLUMNS = [
    "issuer_cik",
    "issuer_ticker",
    "issuer_name",
    "owner_cik",
    "owner_name",
    "txn_date",
    "filing_date",
    "shares",
    "price",
    "is_officer",
    "is_director",
    "is_ten_pct",
]


class _Throttle:
    """Thread-safe global request spacer so concurrent fetchers stay under SEC's limit.

    The small spacing sleep is held under a lock, so requests are gated to one every
    ``min_interval`` seconds in aggregate across all worker threads; each request's network
    latency then overlaps off-lock, which is where concurrency buys throughput.
    """

    def __init__(self, min_interval: float = _MIN_INTERVAL) -> None:
        self.min_interval = min_interval
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            dt = time.monotonic() - self._last
            if dt < self.min_interval:
                time.sleep(self.min_interval - dt)
            self._last = time.monotonic()


def _session(pool: int = _WORKERS + 2):
    import requests
    from requests.adapters import HTTPAdapter

    s = requests.Session()
    s.headers.update({"User-Agent": _USER_AGENT, "Accept-Encoding": "gzip, deflate"})
    adapter = HTTPAdapter(pool_connections=pool, pool_maxsize=pool)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def _get(session, throttle: _Throttle, url: str) -> str | None:
    """GET text with throttling and retry/backoff. Returns None on persistent failure."""
    for attempt in range(_RETRIES):
        throttle.wait()
        try:
            r = session.get(url, timeout=60)
        except Exception as e:
            log.debug("network error (attempt %d) on %s: %s", attempt + 1, url, e)
            time.sleep(0.5 * (attempt + 1))
            continue
        if r.status_code == 200:
            return r.text
        if r.status_code in (403, 429) or r.status_code >= 500:
            # Rate-limited / server error: back off and retry. Surfaced so the runner log
            # shows when SEC is throttling us.
            log.warning("HTTP %d (attempt %d/%d), backing off: %s",
                        r.status_code, attempt + 1, _RETRIES, url)
            time.sleep(1.0 * (attempt + 1))
            continue
        log.debug("HTTP %d (no retry): %s", r.status_code, url)
        return None  # 404 etc. -> nothing to retry
    log.warning("giving up after %d attempts: %s", _RETRIES, url)
    return None


def load_ticker_map(session=None, throttle=None, refresh: bool = False) -> pd.DataFrame:
    """CIK -> ticker/title map from SEC ``company_tickers.json`` (listed issuers).

    Returns a frame indexed by integer ``cik`` with columns ``ticker`` and ``title``.
    Cached to parquet. This map is the bounded, survivorship-accepted universe.
    """
    cache = _CACHE / "company_tickers.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    own = session is None
    session = session or _session()
    throttle = throttle or _Throttle()
    log.info("fetching CIK->ticker map: https://www.sec.gov/files/company_tickers.json")
    txt = _get(session, throttle, "https://www.sec.gov/files/company_tickers.json")
    if txt is None:
        raise RuntimeError("could not fetch company_tickers.json from SEC")
    import json

    rows = [
        {"cik": int(v["cik_str"]), "ticker": str(v["ticker"]).upper(), "title": v["title"]}
        for v in json.loads(txt).values()
    ]
    df = pd.DataFrame(rows).drop_duplicates("cik").set_index("cik").sort_index()
    log.info("ticker map: %d listed issuers", len(df))
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    if own:
        session.close()
    return df


def _quarters(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[int, int]]:
    qs = []
    p = pd.Period(start, freq="Q")
    last = pd.Period(end, freq="Q")
    while p <= last:
        qs.append((p.year, p.quarter))
        p += 1
    return qs


def _form4_index(session, throttle: _Throttle, year: int, qtr: int) -> pd.DataFrame:
    """Form 4 rows from one quarterly EDGAR full-index ``form.idx``.

    Columns: ``issuer_cik`` (int), ``issuer_name``, ``filing_date``, ``path`` (the
    full-submission .txt URL path). Cached raw to parquet per quarter.
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
        # Fixed-ish layout: "<type> <company> <cik> <date> <path>"; type is col 0.
        if not line[:2].strip() == "4":
            continue
        parts = line.split()
        if not parts or parts[0] not in _FORM4_TYPES:
            continue
        # Path is the last token; date is the token before it; cik before that.
        path = parts[-1]
        date = parts[-2]
        cik = parts[-3]
        if not (cik.isdigit() and re.match(r"\d{4}-\d{2}-\d{2}", date)):
            continue
        name = " ".join(parts[1:-3])
        rows.append((int(cik), name, date, path))
    df = pd.DataFrame(rows, columns=["issuer_cik", "issuer_name", "filing_date", "path"])
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    log.info("%dQ%d: index lists %d Form 4 filings", year, qtr, len(df))
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def _text(el, tag: str) -> str | None:
    """Value of ``<tag><value>...</value></tag>`` or ``<tag>...</tag>``."""
    node = el.find(tag)
    if node is None:
        return None
    v = node.find("value")
    node = v if v is not None else node
    return (node.text or "").strip() or None


def _parse_form4(xml: str) -> list[dict]:
    """Extract open-market purchase (Code "P") rows from one ownership XML document."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []

    issuer = root.find("issuer")
    issuer_cik = _text(issuer, "issuerCik") if issuer is not None else None
    issuer_ticker = _text(issuer, "issuerTradingSymbol") if issuer is not None else None
    issuer_name = _text(issuer, "issuerName") if issuer is not None else None

    # Reporting owner relationship flags (officer / director / 10% holder).
    is_officer = is_director = is_ten_pct = False
    owner_cik = owner_name = None
    owner = root.find("reportingOwner")
    if owner is not None:
        oid = owner.find("reportingOwnerId")
        if oid is not None:
            owner_cik = _text(oid, "rptOwnerCik")
            owner_name = _text(oid, "rptOwnerName")
        rel = owner.find("reportingOwnerRelationship")
        if rel is not None:
            is_officer = (_text(rel, "isOfficer") or "0") in ("1", "true", "True")
            is_director = (_text(rel, "isDirector") or "0") in ("1", "true", "True")
            is_ten_pct = (_text(rel, "isTenPercentOwner") or "0") in ("1", "true", "True")

    # Exclude pure 10% holders who are not operating management (officer/director).
    if is_ten_pct and not (is_officer or is_director):
        return []

    out = []
    for txn in root.iter("nonDerivativeTransaction"):
        coding = txn.find("transactionCoding")
        code = _text(coding, "transactionCode") if coding is not None else None
        if code != "P":  # open-market purchase only; drops A (grant), M (exercise), etc.
            continue
        amounts = txn.find("transactionAmounts")
        shares = _text(amounts, "transactionShares") if amounts is not None else None
        price = _text(amounts, "transactionPricePerShare") if amounts is not None else None
        txn_date = _text(txn, "transactionDate")
        out.append(
            {
                "issuer_cik": int(issuer_cik) if issuer_cik else None,
                "issuer_ticker": (issuer_ticker or "").upper() or None,
                "issuer_name": issuer_name,
                "owner_cik": int(owner_cik) if owner_cik else None,
                "owner_name": owner_name,
                "txn_date": pd.to_datetime(txn_date) if txn_date else None,
                "shares": float(shares) if shares else None,
                "price": float(price) if price else None,
                "is_officer": is_officer,
                "is_director": is_director,
                "is_ten_pct": is_ten_pct,
            }
        )
    return out


def _fetch_filing(session, throttle: _Throttle, row) -> tuple[bool, list[dict]]:
    """Fetch+parse one Form 4 filing. Returns (fetched_ok, purchase_rows)."""
    url = f"https://www.sec.gov/Archives/{row.path}"
    txt = _get(session, throttle, url)
    if txt is None:
        return False, []
    m = _OWNERSHIP_RE.search(txt)
    if m is None:
        return True, []
    recs = _parse_form4(m.group(0))
    for rec in recs:
        rec["filing_date"] = row.filing_date
        if _LOG_PURCHASES:
            # Log every open-market purchase found, with the source filing URL.
            log.info(
                "  BUY %-6s %s  %s sh @ $%s  txn=%s filed=%s  %s",
                rec.get("issuer_ticker") or "?",
                (rec.get("owner_name") or "?")[:28],
                _fmt(rec.get("shares")),
                _fmt(rec.get("price")),
                _date(rec.get("txn_date")),
                _date(row.filing_date),
                url,
            )
    return True, recs


def _fmt(x) -> str:
    return f"{x:,.0f}" if isinstance(x, (int, float)) and x == x else "?"


def _date(x) -> str:
    try:
        return pd.Timestamp(x).date().isoformat()
    except Exception:
        return "?"


def _load_quarter(
    session,
    throttle: _Throttle,
    year: int,
    qtr: int,
    ciks: set[int] | None,
    refresh: bool,
    workers: int = _WORKERS,
) -> pd.DataFrame:
    """Parse all Code-"P" purchases filed in one quarter (cached to parquet).

    Filings are fetched concurrently across ``workers`` threads; the shared ``throttle``
    keeps the aggregate request rate under SEC's fair-access ceiling.
    """
    cache = _CACHE / f"{year}Q{qtr}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    idx_all = _form4_index(session, throttle, year, qtr)
    idx = idx_all[idx_all["issuer_cik"].isin(ciks)] if ciks is not None else idx_all
    rows = list(idx.itertuples(index=False))
    log.info(
        "%dQ%d: fetching %d filings (%d after CIK filter of %d listed) with %d workers",
        year, qtr, len(rows), len(idx), len(ciks) if ciks is not None else 0, len(idx_all),
    )

    records: list[dict] = []
    n_fetch = n_fail = 0
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for ok, recs in ex.map(lambda r: _fetch_filing(session, throttle, r), rows):
            n_fetch += 1
            if not ok:
                n_fail += 1
            else:
                records.extend(recs)
            if n_fetch % _LOG_EVERY == 0:
                elapsed = time.monotonic() - t0
                rate = n_fetch / elapsed if elapsed else 0.0
                remaining = (len(rows) - n_fetch) / rate if rate else 0.0
                log.info(
                    "%dQ%d: %d/%d filings (%.0f%%) | %.1f/s | %d buys | %d failed | "
                    "~%.0f min left",
                    year, qtr, n_fetch, len(rows), 100 * n_fetch / max(len(rows), 1),
                    rate, len(records), n_fail, remaining / 60,
                )

    df = pd.DataFrame(records, columns=COLUMNS)
    if not df.empty:
        df = df.dropna(subset=["issuer_ticker", "owner_cik", "filing_date"])
    df.attrs["n_filings_fetched"] = n_fetch
    df.attrs["n_fetch_failed"] = n_fail
    log.info(
        "%dQ%d: DONE -> %d purchase rows from %d filings (%d failed) in %.0fs",
        year, qtr, len(df), n_fetch, n_fail, time.monotonic() - t0,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def load_form4_purchases(
    start: str = "2003-01-01",
    end: str | None = None,
    refresh: bool = False,
    restrict_to_listed: bool = True,
) -> pd.DataFrame:
    """Open-market insider purchases (Form 4 Code "P") from EDGAR, keyed on filing date.

    Enumerates the quarterly full-index across ``[start, end]``, fetches each Form 4
    full-submission, and parses Code-"P" non-derivative purchases. When
    ``restrict_to_listed`` (default), the issuer universe is bounded by SEC's
    ``company_tickers.json`` (listed names with a resolvable ticker) -- the accepted,
    survivorship-biased free universe.

    Returns a frame with :data:`COLUMNS`; ``filing_date`` is the point-in-time stamp.
    Results are cached per quarter under ``data/edgar/form4/`` so the (long) build resumes
    without refetching. Set ``refresh=True`` to rebuild a quarter from the network.
    """
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    start_ts = pd.Timestamp(start)
    session = _session()
    throttle = _Throttle()
    ciks: set[int] | None = None
    if restrict_to_listed:
        ciks = set(load_ticker_map(session, throttle, refresh=refresh).index.tolist())

    frames = []
    for year, qtr in _quarters(start_ts, end_ts):
        frames.append(_load_quarter(session, throttle, year, qtr, ciks, refresh))
    session.close()

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    if not out.empty:
        out = out[(out["filing_date"] >= start_ts) & (out["filing_date"] <= end_ts)]
        out = out.sort_values("filing_date").reset_index(drop=True)
    return out
