"""Build (parse) the EDGAR NPORT-P holdings + fund-flow cache (cloud-build matrix driver).

Heavy all-filer parse: every NPORT-P in the quarterly index is fetched and parsed, keeping
the long common-equity holdings and the fund-level flows/returns/net-assets of funds above
the net-asset floor. Each quarter writes two parquet products under the cache dir
(``NPORT_CACHE_DIR`` or ``data/edgar/nport/``): ``<Q>.parquet`` (holdings) and
``<Q>_funds.parquet`` (fund flows); re-running skips finished quarters. Public NPORT-P starts
~2019Q4, so the quarter list is short but each quarter is a large fetch, which is why this
shards for the cloud matrix (one runner per quarter, distinct IPs under SEC fair access).

Three ways to run (parity with the Form 4 / 13F drivers; local sequential default):

    python build_data_nport.py                   # build every quarter (sequential)
    python build_data_nport.py --list-quarters    # print JSON ["2019Q3", ...] for a CI matrix
    python build_data_nport.py --quarter 2020Q1   # build exactly one quarter, then exit
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import pandas as pd

try:  # repo layout
    from libs.data.academic import nport
except ImportError:  # flat checkout (public cloud-build repo)
    import nport  # type: ignore

DEFAULT_START = nport.DEFAULT_START  # "2019-07-01"


def _label(year: int, qtr: int) -> str:
    return f"{year}Q{qtr}"


def _parse_label(label: str) -> tuple[int, int]:
    year, qtr = label.upper().split("Q")
    return int(year), int(qtr)


def list_quarters(start: str = DEFAULT_START, end: str | None = None) -> list[str]:
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    qs = nport._quarters(pd.Timestamp(start), end_ts)
    return [_label(y, q) for (y, q) in qs]


def build_one(label: str) -> None:
    """Build exactly one quarter's NPORT-P cache (for a CI matrix); cached + idempotent."""
    year, qtr = _parse_label(label)
    session = nport._session()
    throttle = nport._Throttle()
    t0 = time.time()
    holds = nport._load_quarter(session, throttle, year, qtr)
    session.close()
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {label}: {len(holds):,} equity holding rows "
        f"in {time.time() - t0:.0f}s",
        flush=True,
    )


def build_all(start: str = DEFAULT_START, end: str | None = None) -> None:
    session = nport._session()
    throttle = nport._Throttle()
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    quarters = nport._quarters(pd.Timestamp(start), end_ts)
    done_marker = nport._CACHE / "BUILD_COMPLETE"
    if done_marker.exists():
        done_marker.unlink()

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {len(quarters)} quarters, NPORT-P parse",
          flush=True)
    total = 0
    for year, qtr in quarters:
        t0 = time.time()
        holds = nport._load_quarter(session, throttle, year, qtr)
        total += len(holds)
        print(
            f"[{time.strftime('%H:%M:%S')}] {_label(year, qtr)}: {len(holds):,} holding rows "
            f"in {time.time() - t0:.0f}s (cumulative {total:,})",
            flush=True,
        )
    session.close()
    done_marker.parent.mkdir(parents=True, exist_ok=True)
    done_marker.write_text(
        f"completed {time.strftime('%Y-%m-%d %H:%M:%S')}; {total} holding rows\n"
    )
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] DONE -> {done_marker}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the EDGAR NPORT-P fund-holdings/flow cache.")
    ap.add_argument("--start", default=DEFAULT_START, help="sample start (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="sample end (YYYY-MM-DD); default today")
    ap.add_argument("--list-quarters", action="store_true",
                    help="print JSON array of quarter labels and exit (for a CI matrix)")
    ap.add_argument("--quarter", help="build exactly one quarter (e.g. 2020Q1) and exit")
    args = ap.parse_args()

    if not args.list_quarters:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
        )

    if args.list_quarters:
        print(json.dumps(list_quarters(args.start, args.end)))
    elif args.quarter:
        build_one(args.quarter)
    else:
        build_all(args.start, args.end)


if __name__ == "__main__":
    main()
