"""SEC EDGAR Form 10 (spinoff registration) ingestion (free; research pre-validation).

Free and usable now. A company being spun off registers its shares under the Exchange Act
by filing a **Form 10 registration statement** -- ``10-12B`` (Section 12(b), an exchange
listing) or ``10-12G`` (Section 12(g)). Enumerating these from the EDGAR quarterly
full-index gives a systematic, point-in-time list of prospective spinoffs: each row is the
registrant CIK of the NEW, independent entity and the date it registered.

Design notes (see strategies/spinoff/README.md and the plan):
  - INDEX ONLY: unlike the 13F / Form 4 pipelines, we do NOT parse each full-submission
    ``.txt``. The registrant CIK + filing date from the quarterly ``form.idx`` is all the
    signal needs; the distribution anchor comes from the first free-price trade date
    downstream (strategies/spinoff/backtest.py). This makes the pull tiny (dozens of Form
    10s/year) and fast -- a minutes-long sequential fetch, no cloud matrix required.
  - POINT-IN-TIME: every row carries the ``filing_date`` from the index. A registrant is
    deduped to its EARLIEST Form 10 filing (an entity re-files ``10-12B/A`` amendments as
    the S-1-style information statement evolves), so the registration date is the first
    public point at which the spinoff entered the universe.
  - Some Form 10 registrants are not spinoffs (direct listings, post-bankruptcy emergences).
    :func:`classify_spinoffs` is an OPTIONAL filing-text screen (fetches each registrant's
    full submission and keyword-classifies spinoff vs not) that purges that noise; it is a
    disclosed hygiene sensitivity, off by default so the pre-registered universe is unchanged.
  - SURVIVORSHIP: CIK->ticker resolution uses SEC ``company_tickers.json`` (currently listed
    issuers only, via the Form 4 spine's :func:`load_ticker_map`). Spun-off names since
    acquired/delisted do not resolve and are dropped UPWARD -- a loudly-flagged limitation
    of the free pass, deferred to Compustat/CRSP after 2026-08-25 (ADR 0003 / STATUS.md).

Reuses the hardened EDGAR fetch spine from :mod:`libs.data.academic.form4` (session, global
throttle, retry/backoff, quarter enumeration, ticker map) so SEC fair-access behavior is
identical. The flat-import fallback lets this module run unchanged in a flat checkout.

RESEARCH ONLY (ADR 0003): never import from a live/execution path.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import pandas as pd

try:  # repo layout
    from libs.data.academic.form4 import (  # noqa: F401
        _Throttle,
        _get,
        _quarters,
        _session,
        load_ticker_map,
    )
except ImportError:  # flat checkout (public cloud-build matrix repo)
    from form4 import (  # type: ignore  # noqa: F401
        _Throttle,
        _get,
        _quarters,
        _session,
        load_ticker_map,
    )

log = logging.getLogger("form10")

# git-ignored parquet cache (AGENTS.md: cache pulled data, never commit it). Overridable via
# FORM10_CACHE_DIR so the same module works in the repo layout and in a flat checkout.
_CACHE = Path(
    os.environ.get(
        "FORM10_CACHE_DIR",
        Path(__file__).resolve().parents[3] / "data" / "edgar" / "form10",
    )
)
_IDX_CACHE = _CACHE / "_index"

# Form 10 registration statements used by spinoffs to register their shares. Both the
# original and the amended variants are enumerated; dedupe-to-earliest keeps the first date.
_FORM10_TYPES = {"10-12B", "10-12G", "10-12B/A", "10-12G/A"}

DEFAULT_START = "2000-01-01"  # README sample window (modern spinoff regime)

COLUMNS = ["cik", "name", "filing_date", "form_type", "path"]


def _form10_index(session, throttle: _Throttle, year: int, qtr: int) -> pd.DataFrame:
    """Form 10 rows from one quarterly EDGAR full-index ``form.idx`` (cached per quarter).

    Columns: ``cik`` (int, the registrant -- the new spun-off entity), ``name``,
    ``filing_date``, ``form_type``, ``path`` (the full-submission .txt URL path).
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
        # Fixed-ish layout: "<type> <company> <cik> <date> <path>". The Form 10 type tokens
        # ("10-12B", "10-12G", "10-12B/A", "10-12G/A") have no internal spaces, so split()
        # keeps col 0 whole -- same parsing contract the 13F index relies on.
        parts = line.split()
        if not parts or parts[0] not in _FORM10_TYPES:
            continue
        path = parts[-1]
        date = parts[-2]
        cik = parts[-3]
        if not (cik.isdigit() and re.match(r"\d{4}-\d{2}-\d{2}", date)):
            continue
        name = " ".join(parts[1:-3])
        rows.append((int(cik), name, date, parts[0], path))
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    log.info("%dQ%d: index lists %d Form 10 filings", year, qtr, len(df))
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def load_registrants(
    start: str = DEFAULT_START,
    end: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Point-in-time Form 10 registrants across ``[start, end]``, one row per CIK.

    Enumerates the quarterly full-index (lazily fetching + caching each quarter's index),
    concatenates, restricts to the date window, and dedupes each CIK to its EARLIEST Form 10
    filing (the registration date). Returns :data:`COLUMNS` sorted by ``filing_date``.
    """
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    start_ts = pd.Timestamp(start)
    if refresh:  # drop the per-quarter index cache so it is refetched
        for f in _IDX_CACHE.glob("*Q*.parquet"):
            f.unlink()
    session = _session()
    throttle = _Throttle()
    frames = []
    for year, qtr in _quarters(start_ts, end_ts):
        frames.append(_form10_index(session, throttle, year, qtr))
    session.close()

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    if out.empty:
        return out
    out = out[(out["filing_date"] >= start_ts) & (out["filing_date"] <= end_ts)]
    # Dedupe to the earliest registration per entity (amendments re-file under the same CIK).
    out = (
        out.sort_values("filing_date")
        .drop_duplicates(subset="cik", keep="first")
        .reset_index(drop=True)
    )
    return out


# ------------------------------------------------------------------ filing-text screen
# Keyword screen to separate genuine spinoffs from the other entities that file a Form 10
# (direct listings, post-bankruptcy emergences, uplisting shells). Applied to the FULL
# submission text of each registrant's earliest Form 10. Terms are fixed a priori from the
# structure of a spinoff information statement (do not tune to the backtest); the screen is a
# disclosed hygiene sensitivity, not a change to the pre-registered gate. Matching is
# case-insensitive on lowercased text.
#
# High-precision spinoff language: the Separation and Distribution Agreement and the pro-rata
# distribution to the PARENT's shareholders are near-unique to spinoffs.
_SPINOFF_TERMS = (
    "spin-off",
    "spinoff",
    "spin off",
    "separation and distribution agreement",
    "pro rata distribution",
    "distribution of our",
    "distribution of the",
    "record date for the distribution",
)
# Post-bankruptcy emergence: shares registered under a plan of reorganization, not a spinoff.
_BANKRUPTCY_TERMS = (
    "plan of reorganization",
    "chapter 11",
    "emergence from bankruptcy",
    "fresh start",
    "fresh-start",
    "bankruptcy court",
)
# Direct listing / small-issue registrations that are not spinoffs.
_DIRECTLIST_TERMS = ("direct listing", "regulation a", "form 1-a")

_CLASSIFY_CACHE = _CACHE / "_classify"

# ------------------------------------------------------------------ delisted recovery
# The survivorship-hole re-attack (STATUS.md "Planned Retests"): CIKs that do NOT resolve
# via the current-issuer company_tickers.json map (delisted / acquired) are recovered here to
# their HISTORICAL trading symbol from two free layers, tried in order:
#   A. SEC submissions API (data.sec.gov/submissions/CIK##########.json) -- its ``tickers``
#      field is populated for many (not all) once-listed names. Clean, one request per CIK.
#   B. filing-text parse of the registrant's earliest Form 10 cover page ("Trading Symbol(s)"
#      / under the symbol "XYZ") for the CIKs Layer A misses. Higher coverage, more fragile.
# Recovered symbols feed a delisted-price recovery loader (strategies/spinoff/recover.py) and
# then the losers-IN basket. Committed BEFORE results (PREREGISTRATION_recovery.md); every cut
# reported. RESEARCH ONLY (ADR 0003).
_RECOVERY_CACHE = _CACHE / "_recovery"

# SEC submissions API: CIK is zero-padded to 10 digits in the URL, un-padded in the payload.
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

# Cover-page trading-symbol patterns, tried in priority order. Kept conservative so we don't
# scrape a random capitalized word (the failure mode is grabbing the next table-header token
# like "Name" after "Trading Symbol(s)"):
#   1. a QUOTED uppercase token within a short window after an explicit symbol anchor (the
#      common `Trading Symbol(s) ... "TICK"` cover-page layout, and `under the symbol "TICK"`);
#   2. an UNQUOTED all-caps token immediately after "Trading Symbol(s)" (the plain-table layout)
#      -- case-sensitive [A-Z] here, so mixed-case header words ("Name", "Title") never match.
_TICKER_RE = [
    re.compile(
        r'(?:trading symbol\(?s?\)?|ticker symbol|under the (?:ticker )?symbol|\bsymbol)'
        r'[^"“\n]{0,40}["“]([A-Za-z]{1,5})["”]',
        re.IGNORECASE,
    ),
    re.compile(r'trading symbol\(?s?\)?[:\s|]+([A-Z]{1,5})\b'),
]
# Tokens that match the shape but are never a spun-off entity's ticker (legal/boilerplate and
# cover-page header words).
_TICKER_STOP = {
    "SEC", "LLC", "INC", "THE", "AND", "FOR", "NYSE", "OTC", "USD", "CIK", "LP", "CO",
    "CORP", "LTD", "PLC", "CLASS", "COMMON", "STOCK", "SHARE", "SHARES", "NA", "N", "A",
    "B", "C", "I", "TBD", "NONE", "NAME", "TITLE", "EACH", "OF", "ON", "EXCH",
}


def _parse_ticker_from_filing(text: str) -> str | None:
    """Best-effort historical trading symbol from a Form 10 cover page, or ``None``.

    Conservative on purpose: only returns a symbol adjacent to explicit "trading symbol" /
    "under the symbol" language, rejects legal-boilerplate tokens, and prefers the earliest
    match (the cover page). A miss returns ``None`` so the caller records the cut.
    """
    if not text:
        return None
    head = text[:400_000]  # cover page + item 1 register early; bound the scan
    for rx in _TICKER_RE:
        for m in rx.finditer(head):
            tick = m.group(1).upper()
            if tick and tick not in _TICKER_STOP:
                return tick
    return None


def _submission_url(path: str) -> str:
    """Full-submission .txt URL from the ``path`` column of the quarterly index."""
    return f"https://www.sec.gov/Archives/{path.lstrip('/')}"


def _classify_text(text: str) -> tuple[bool, str]:
    """Classify one Form 10 submission as spinoff / not, with a reason label.

    Returns ``(is_spinoff, label)``. A filing is a spinoff when it carries clear spinoff
    language and that language is not dominated by post-bankruptcy-emergence language (a few
    reorganization plans mention a "spin-off" of a unit while the registrant itself is the
    emerging debtor). Errs toward KEEPING names: any strong spinoff term with no dominant
    bankruptcy signal counts as a spinoff, so a real spinoff with terse language is not
    dropped.
    """
    t = text.lower()
    spin = sum(term in t for term in _SPINOFF_TERMS)
    bank = sum(term in t for term in _BANKRUPTCY_TERMS)
    direct = sum(term in t for term in _DIRECTLIST_TERMS)
    if spin and spin >= bank:
        return True, "spinoff"
    if bank > spin:
        return False, "non_spinoff_bankruptcy"
    if direct and not spin:
        return False, "non_spinoff_direct_listing"
    return False, "non_spinoff_other"


def classify_spinoffs(registrants: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """Filing-text spinoff/not classification for each registrant, cached per run.

    Fetches the earliest Form 10 full submission for each row of ``registrants`` (which must
    carry ``cik`` and ``path`` from :func:`load_registrants`), classifies it with
    :func:`_classify_text`, and returns ``[cik, is_spinoff, label]``. Results are cached to a
    single parquet keyed by ``cik`` and only missing CIKs are fetched, so re-runs are cheap and
    resumable. A submission that cannot be fetched is labelled ``fetch_failed`` and kept
    (``is_spinoff=True``) so a transient SEC error never silently drops a name.
    """
    cols = ["cik", "is_spinoff", "label"]
    if registrants.empty:
        return pd.DataFrame(columns=cols)
    _CLASSIFY_CACHE.mkdir(parents=True, exist_ok=True)
    cache = _CLASSIFY_CACHE / "labels.parquet"
    cached = pd.read_parquet(cache) if (cache.exists() and not refresh) else pd.DataFrame(columns=cols)
    have = set(cached["cik"].astype(int)) if not cached.empty else set()

    todo = registrants[~registrants["cik"].astype(int).isin(have)]
    if not todo.empty:
        session = _session()
        throttle = _Throttle()
        rows = []
        for r in todo.itertuples(index=False):
            txt = _get(session, throttle, _submission_url(r.path))
            if txt is None:
                rows.append((int(r.cik), True, "fetch_failed"))
                continue
            is_spin, label = _classify_text(txt)
            rows.append((int(r.cik), bool(is_spin), label))
        session.close()
        cached = pd.concat([cached, pd.DataFrame(rows, columns=cols)], ignore_index=True)
        cached.to_parquet(cache)

    want = set(registrants["cik"].astype(int))
    out = cached[cached["cik"].astype(int).isin(want)].reset_index(drop=True)
    return out


def resolve_tickers(ciks, refresh: bool = False) -> pd.DataFrame:
    """Resolve registrant CIKs to current tickers via SEC ``company_tickers.json``.

    Thin wrapper over the Form 4 spine's :func:`load_ticker_map` (CIK-keyed, so this is a
    direct index lookup -- no CUSIP/OpenFIGI round-trip needed). Returns ``[cik, ticker,
    name]`` for the CIKs that resolve; unresolved CIKs (delisted / acquired / never listed)
    are simply absent and are the flagged survivorship hole.
    """
    want = {int(c) for c in ciks}
    tmap = load_ticker_map(refresh=refresh)  # indexed by int cik, cols [ticker, title]
    hit = tmap[tmap.index.isin(want)]
    return pd.DataFrame(
        {"cik": hit.index.astype(int), "ticker": hit["ticker"].values, "name": hit["title"].values}
    ).reset_index(drop=True)


def _ticker_from_submissions(txt: str) -> tuple[str, str]:
    """First non-empty ``(ticker, exchange)`` from a data.sec.gov submissions JSON payload.

    Returns ``("", "")`` when the payload has no ticker (the common case for long-delisted
    names) or cannot be parsed -- the caller then falls through to the filing-text layer.
    """
    import json

    try:
        d = json.loads(txt)
    except Exception:
        return "", ""
    tickers = d.get("tickers") or []
    exchanges = d.get("exchanges") or []
    for i, t in enumerate(tickers):
        t = str(t).strip().upper()
        if t:
            exch = str(exchanges[i]).strip() if i < len(exchanges) and exchanges[i] else ""
            return t, exch
    return "", ""


def recover_delisted_tickers(registrants: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """Recover HISTORICAL trading symbols for delisted/acquired Form 10 registrants (free).

    The survivorship-hole re-attack. ``registrants`` must carry ``cik`` and (for the
    filing-text fallback) ``path`` from :func:`load_registrants`; pass ONLY the CIKs that do
    not resolve via :func:`resolve_tickers` (the delisted/acquired hole). For each CIK, tries
    two free layers in order: (A) the SEC submissions API ``tickers`` field; (B) a cover-page
    trading-symbol parse of the earliest Form 10. Returns ``[cik, ticker, source, exchange]``
    where ``source`` is ``submissions`` / ``filing_text`` / ``unresolved`` and ``ticker`` is
    ``""`` when neither layer recovers a symbol (a reported cut).

    Cached to a single parquet keyed by ``cik`` (only missing CIKs are fetched, so re-runs and
    the per-quarter cloud-matrix shards are cheap and resumable), mirroring
    :func:`classify_spinoffs`. RESEARCH ONLY (ADR 0003).
    """
    cols = ["cik", "ticker", "source", "exchange"]
    if registrants is None or registrants.empty:
        return pd.DataFrame(columns=cols)
    reg = registrants.drop_duplicates(subset="cik", keep="first")
    _RECOVERY_CACHE.mkdir(parents=True, exist_ok=True)
    cache = _RECOVERY_CACHE / "labels.parquet"
    cached = pd.read_parquet(cache) if (cache.exists() and not refresh) else pd.DataFrame(columns=cols)
    have = set(cached["cik"].astype(int)) if not cached.empty else set()

    todo = reg[~reg["cik"].astype(int).isin(have)]
    if not todo.empty:
        session = _session()
        throttle = _Throttle()
        has_path = "path" in todo.columns
        rows = []
        for r in todo.itertuples(index=False):
            cik = int(r.cik)
            # Layer A: SEC submissions API (clean, one request).
            txt = _get(session, throttle, _SUBMISSIONS_URL.format(cik=cik))
            tick, exch = _ticker_from_submissions(txt) if txt else ("", "")
            source = "submissions" if tick else ""
            # Layer B: cover-page trading-symbol parse for the Layer-A misses.
            path = getattr(r, "path", None) if has_path else None
            if not tick and path:
                ftxt = _get(session, throttle, _submission_url(path))
                tick = _parse_ticker_from_filing(ftxt) if ftxt else None
                if tick:
                    source, exch = "filing_text", ""
            rows.append((cik, tick or "", source or "unresolved", exch or ""))
        session.close()
        cached = pd.concat([cached, pd.DataFrame(rows, columns=cols)], ignore_index=True)
        cached.to_parquet(cache)

    want = set(reg["cik"].astype(int))
    out = cached[cached["cik"].astype(int).isin(want)].reset_index(drop=True)
    return out


def load_recovered_tickers() -> pd.DataFrame:
    """Read the built delisted-recovery map ``[cik, ticker, source, exchange]`` (no fetch).

    Prefers the per-quarter recovery indexes written by ``build_recovery.py`` (the cloud-matrix
    artifacts, ``_recovery/*Q*.parquet``); falls back to the resolver's ``labels.parquet`` for a
    purely local run. Returns an empty frame when nothing has been recovered yet, so the primary
    pass degrades gracefully. Only rows with a non-empty ``ticker`` are usable downstream.
    """
    cols = ["cik", "ticker", "source", "exchange"]
    qfiles = sorted(_RECOVERY_CACHE.glob("*Q*.parquet"))
    if qfiles:
        df = pd.concat([pd.read_parquet(f) for f in qfiles], ignore_index=True)
        df = df[[c for c in cols if c in df.columns]]
    else:
        cache = _RECOVERY_CACHE / "labels.parquet"
        df = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=cols)
    if df.empty:
        return pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols].drop_duplicates("cik", keep="first").reset_index(drop=True)
