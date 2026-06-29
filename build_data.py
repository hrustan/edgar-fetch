"""Build the EDGAR Form 4 open-market-purchase cache.

Long-running by design: a full 2003-present parse is ~2-3M filings at SEC's ~10 req/s
fair-access ceiling. It is RESUMABLE -- each quarter is cached to parquet under the cache
dir (``FORM4_CACHE_DIR`` or ``data/edgar/form4/``) and re-running skips finished quarters.

This driver runs three ways, so it works both locally and as a GitHub Actions matrix where
each runner (a distinct IP) builds one quarter under SEC's per-IP rate limit:

    python build_data.py                  # build every quarter (sequential)
    python build_data.py --list-quarters  # print JSON ["2003Q1", ...] for a CI matrix
    python build_data.py --quarter 2015Q3 # build exactly one quarter, then exit

The full sequential build writes a ``BUILD_COMPLETE`` marker when done.
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import pandas as pd

try:  # repo layout
    from libs.data.academic import form4
except ImportError:  # flat checkout (public cloud-build repo)
    import form4  # type: ignore

DEFAULT_START = "2003-01-01"


def _label(year: int, qtr: int) -> str:
    return f"{year}Q{qtr}"


def _parse_label(label: str) -> tuple[int, int]:
    year, qtr = label.upper().split("Q")
    return int(year), int(qtr)


def list_quarters(start: str = DEFAULT_START, end: str | None = None) -> list[str]:
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    qs = form4._quarters(pd.Timestamp(start), end_ts)
    return [_label(y, q) for (y, q) in qs]


def build_one(label: str) -> None:
    """Build exactly one quarter (for the CI matrix); cached + idempotent."""
    year, qtr = _parse_label(label)
    session = form4._session()
    throttle = form4._Throttle()
    ciks = set(form4.load_ticker_map(session, throttle).index.tolist())
    t0 = time.time()
    df = form4._load_quarter(session, throttle, year, qtr, ciks, refresh=False)
    session.close()
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {label}: {len(df):,} P-buy rows from "
        f"{df.attrs.get('n_filings_fetched', '?'):,} filings in {time.time() - t0:.0f}s",
        flush=True,
    )


def build_all(start: str = DEFAULT_START) -> None:
    session = form4._session()
    throttle = form4._Throttle()
    ciks = set(form4.load_ticker_map(session, throttle).index.tolist())
    quarters = form4._quarters(pd.Timestamp(start), pd.Timestamp.today().normalize())
    done_marker = form4._CACHE / "BUILD_COMPLETE"
    if done_marker.exists():
        done_marker.unlink()

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {len(quarters)} quarters, "
          f"universe={len(ciks)} listed CIKs", flush=True)
    total_rows = 0
    for year, qtr in quarters:
        cache = form4._CACHE / f"{_label(year, qtr)}.parquet"
        if cache.exists():
            total_rows += len(pd.read_parquet(cache))
            print(f"[{time.strftime('%H:%M:%S')}] {_label(year, qtr)} cached, skip", flush=True)
            continue
        t0 = time.time()
        df = form4._load_quarter(session, throttle, year, qtr, ciks, refresh=False)
        total_rows += len(df)
        print(
            f"[{time.strftime('%H:%M:%S')}] {_label(year, qtr)}: {len(df):,} P-buy rows from "
            f"{df.attrs.get('n_filings_fetched', '?'):,} filings in {time.time() - t0:.0f}s "
            f"(cumulative {total_rows:,} rows)",
            flush=True,
        )
    session.close()
    done_marker.write_text(
        f"completed {time.strftime('%Y-%m-%d %H:%M:%S')}; {total_rows} P-buy rows\n"
    )
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] DONE -> {done_marker}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the EDGAR Form 4 purchase cache.")
    ap.add_argument("--start", default=DEFAULT_START, help="sample start (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="sample end (YYYY-MM-DD); default today")
    ap.add_argument("--list-quarters", action="store_true",
                    help="print JSON array of quarter labels and exit (for a CI matrix)")
    ap.add_argument("--quarter", help="build exactly one quarter (e.g. 2015Q3) and exit")
    args = ap.parse_args()

    # Stream the form4 logger's INFO records to stdout (live in a CI runner log). Skipped
    # for --list-quarters so its sole stdout line stays clean JSON for the matrix.
    if not args.list_quarters:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )

    if args.list_quarters:
        print(json.dumps(list_quarters(args.start, args.end)))
    elif args.quarter:
        build_one(args.quarter)
    else:
        build_all(args.start)


if __name__ == "__main__":
    main()
