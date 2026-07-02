"""Reduce the raw NPORT-P cache to the compact fire-sale pressure panel (cloud reduce step).

The full per-quarter holdings cache is ~46M rows / 1.3GB. The fire-sale signal only ever uses
holdings of forced-seller funds (the bottom flow decile each quarter), so this writes the
pre-filtered, pre-joined slice (``pressure_panel.parquet``, ~25MB) plus the small fund-flow
table (``pressure_funds.parquet``, ~12MB) and nothing else, so the raw cache never has to be
downloaded. The reduction is EXACT: the panel is a superset of every run's forced-seller
holdings and the exact bottom-decile cutoff is recomputed downstream on the full fund table.

    python reduce_data_nport.py               # reduce the whole cached window
    python reduce_data_nport.py --out DIR      # write the panel to DIR instead of the cache dir
"""

from __future__ import annotations

import argparse
import logging

try:  # repo layout
    from libs.data.academic import nport
except ImportError:  # flat checkout (public cloud-build repo)
    import nport  # type: ignore


def main() -> None:
    ap = argparse.ArgumentParser(description="Reduce the NPORT-P cache to the pressure panel.")
    ap.add_argument("--start", default=nport.DEFAULT_START, help="sample start (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="sample end (YYYY-MM-DD); default today")
    ap.add_argument("--out", default=None, help="output dir (default the NPORT cache dir)")
    ap.add_argument("--flow-pct", type=float, default=nport.PANEL_FLOW_PCT,
                    help="forced-candidate flow decile to retain (superset of any run cutoff)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
    )
    hpath, fpath = nport.write_pressure_panel(
        out_dir=args.out, flow_pct=args.flow_pct, start=args.start, end=args.end
    )
    print(f"wrote {hpath}\nwrote {fpath}", flush=True)


if __name__ == "__main__":
    main()
