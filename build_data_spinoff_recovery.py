"""Build (recover) delisted spinoff price histories for the survivorship-hole re-attack.

The scale step of STATUS.md "Planned Retests": for the ~2,485 Form 10 registrants that do NOT
resolve to a current ticker (the delisted/acquired losers absent from the free primary pass),
recover their HISTORICAL trading symbol and daily price history from free sources, so the
losers-IN basket (strategies/spinoff/validate_recovery.py) can be re-run. This is the piece
built in the public ``hrustan/edgar-fetch`` GitHub Actions matrix -- one runner per
registration-date quarter, resumable and 429-ban-safe (the EDGAR spine's throttle/backoff).

Per quarter shard, each runner:
  1. loads the full point-in-time registrant list (index-only, cached, cheap) and takes the
     cohort whose earliest Form 10 registration falls in this quarter;
  2. drops the CIKs that already resolve via the current-issuer map (handled by the primary);
  3. recovers the delisted CIKs' historical tickers (submissions API -> filing-text parse,
     :func:`libs.data.academic.form10.recover_delisted_tickers`);
  4. recovers each symbol's daily prices (Yahoo -> Stooq,
     :func:`strategies.spinoff.recover.load_ohlcv_recovered`);
  5. writes a per-quarter recovery index parquet plus the per-ticker price parquets (the
     collect job merges the indexes and bundles data/prices/spinoff_delisted/).

Three ways to run (parity with build_data.py; local sequential is the default):

    python build_recovery.py                    # recover every quarter (sequential)
    python build_recovery.py --list-quarters    # print JSON ["2000Q1", ...] for the CI matrix
    python build_recovery.py --quarter 2015Q2   # recover exactly one quarter, then exit

RESEARCH ONLY (ADR 0003): uses the academic data layer; never imported by live code.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd

try:  # repo layout
    from libs.data.academic import form10
    from strategies.spinoff import recover
except ImportError:  # flat checkout (public cloud-build repo)
    import form10  # type: ignore
    import recover  # type: ignore

log = logging.getLogger("spinoff_recovery")

DEFAULT_START = form10.DEFAULT_START  # "2000-01-01"

_RECOVERY_CACHE = Path(
    os.environ.get("FORM10_CACHE_DIR", form10._CACHE)
) / "_recovery"

_INDEX_COLUMNS = [
    "cik", "name", "registration_date", "ticker", "source", "exchange",
    "is_delisted", "price_recovered", "price_source", "n_obs", "first_date", "last_date",
]


def _label(year: int, qtr: int) -> str:
    return f"{year}Q{qtr}"


def _parse_label(label: str) -> tuple[int, int]:
    year, qtr = label.upper().split("Q")
    return int(year), int(qtr)


def list_quarters(start: str = DEFAULT_START, end: str | None = None) -> list[str]:
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    qs = form10._quarters(pd.Timestamp(start), end_ts)
    return [_label(y, q) for (y, q) in qs]


def _unresolved_cohort(year: int, qtr: int, start: str, end: str | None) -> pd.DataFrame:
    """Registrants whose earliest Form 10 registration is in ``year Q qtr`` and that do NOT
    resolve to a current ticker (the delisted/acquired losers to recover)."""
    reg = form10.load_registrants(start=start, end=end)
    if reg.empty:
        return reg
    period = pd.PeriodIndex(reg["filing_date"], freq="Q")
    cohort = reg[(period.year == year) & (period.quarter == qtr)]
    if cohort.empty:
        return cohort
    resolved = set(form10.resolve_tickers(cohort["cik"])["cik"].astype(int))
    return cohort[~cohort["cik"].astype(int).isin(resolved)].reset_index(drop=True)


def _recover_quarter(year: int, qtr: int, start: str, end: str | None) -> pd.DataFrame:
    """Resolve + price-recover one quarter's delisted cohort; return the recovery index frame."""
    cohort = _unresolved_cohort(year, qtr, start, end)
    if cohort.empty:
        return pd.DataFrame(columns=_INDEX_COLUMNS)

    tickers = form10.recover_delisted_tickers(cohort[["cik", "path"]])
    df = cohort.merge(tickers, on="cik", how="left")
    df = df.rename(columns={"filing_date": "registration_date"})

    rows = []
    for r in df.itertuples(index=False):
        ticker = (getattr(r, "ticker", "") or "").strip()
        price_recovered, price_source, n_obs, first_date, last_date = False, "", 0, None, None
        if ticker:
            px = recover.load_ohlcv_recovered(ticker)
            if px is not None and not px["adjclose"].dropna().empty:
                price_recovered = True
                price_source = px.attrs.get("source", "")
                n_obs = int(len(px))
                first_date = px.index.min()
                last_date = px.index.max()
        rows.append((
            int(r.cik), r.name, pd.Timestamp(r.registration_date), ticker,
            getattr(r, "source", "unresolved") or "unresolved", getattr(r, "exchange", "") or "",
            True, price_recovered, price_source, n_obs, first_date, last_date,
        ))
    out = pd.DataFrame(rows, columns=_INDEX_COLUMNS)
    # Normalize datetime resolution to ns (avoids the datetime[us] cloud-parquet merge gotcha).
    for col in ("registration_date", "first_date", "last_date"):
        out[col] = pd.to_datetime(out[col]).astype("datetime64[ns]")
    return out


def build_one(label: str, start: str = DEFAULT_START, end: str | None = None) -> None:
    """Recover exactly one quarter's delisted cohort (for a CI matrix); cached + idempotent."""
    year, qtr = _parse_label(label)
    cache = _RECOVERY_CACHE / f"{label}.parquet"
    t0 = time.time()
    out = _recover_quarter(year, qtr, start, end)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    n_tick = int((out["ticker"] != "").sum()) if not out.empty else 0
    n_px = int(out["price_recovered"].sum()) if not out.empty else 0
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {label}: {len(out):,} delisted, "
        f"{n_tick:,} ticker-resolved, {n_px:,} price-recovered in {time.time() - t0:.0f}s",
        flush=True,
    )


def build_all(start: str = DEFAULT_START, end: str | None = None) -> None:
    quarters = list_quarters(start, end)
    done_marker = _RECOVERY_CACHE / "RECOVERY_COMPLETE"
    if done_marker.exists():
        done_marker.unlink()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] recovering {len(quarters)} quarters", flush=True)
    tot_d = tot_t = tot_p = 0
    for label in quarters:
        year, qtr = _parse_label(label)
        cache = _RECOVERY_CACHE / f"{label}.parquet"
        if cache.exists():  # resumable: skip finished quarters
            out = pd.read_parquet(cache)
        else:
            t0 = time.time()
            out = _recover_quarter(year, qtr, start, end)
            cache.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(cache)
        n_tick = int((out["ticker"] != "").sum()) if not out.empty else 0
        n_px = int(out["price_recovered"].sum()) if not out.empty else 0
        tot_d += len(out); tot_t += n_tick; tot_p += n_px
        print(f"[{time.strftime('%H:%M:%S')}] {label}: {len(out):,} delisted / {n_tick:,} "
              f"resolved / {n_px:,} priced (cum {tot_d:,}/{tot_t:,}/{tot_p:,})", flush=True)
    done_marker.parent.mkdir(parents=True, exist_ok=True)
    done_marker.write_text(
        f"completed {time.strftime('%Y-%m-%d %H:%M:%S')}; "
        f"{tot_d} delisted, {tot_t} resolved, {tot_p} priced\n"
    )
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] DONE -> {done_marker}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Recover delisted spinoff price histories (free).")
    ap.add_argument("--start", default=DEFAULT_START, help="sample start (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="sample end (YYYY-MM-DD); default today")
    ap.add_argument("--list-quarters", action="store_true",
                    help="print JSON array of quarter labels and exit (for a CI matrix)")
    ap.add_argument("--quarter", help="recover exactly one quarter (e.g. 2015Q2) and exit")
    args = ap.parse_args()

    if not args.list_quarters:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    if args.list_quarters:
        print(json.dumps(list_quarters(args.start, args.end)))
    elif args.quarter:
        build_one(args.quarter, args.start, args.end)
    else:
        build_all(args.start, args.end)


if __name__ == "__main__":
    main()
