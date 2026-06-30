# edgar-form4-fetch

Cloud **matrix** for parsing SEC EDGAR filings into per-quarter parquet caches.

SEC's fair-access limit is ~10 requests/sec **per IP**, so a full multi-year parse takes
days on one machine. A GitHub Actions matrix runs one quarter per runner — each a distinct
IP under the limit — so a whole history builds in a few hours on free public-repo runners.
Each pipeline shares the same hardened fetch spine (`form4.py`: throttle, Retry-After +
exponential backoff, a >5%-fail-fraction guard that refuses to cache a holed quarter).

> Named for its first pipeline (Form 4); it now hosts multiple EDGAR pipelines and is the
> shared matrix repo going forward. New EDGAR pulls add a flat module + a `build_data_*.py`
> driver + a `build-*` workflow here.

## Pipelines

| Pipeline | Filings | Driver | Workflow | Cache dir | Artifact |
|---|---|---|---|---|---|
| **Form 4** | open-market insider purchases (Code "P") | `build_data.py` | `build-form4` | `data/edgar/form4/` | `form4_cache` |
| **13F-HR** | institutional holdings (best-ideas inputs) | `build_data_13f.py` | `build-13f` | `data/edgar/thirteen_f/` | `thirteenf_cache` |

Both are keyed on the **filing date** (point-in-time) and respect SEC fair access
(descriptive `User-Agent` + throttle under 10 req/s). Each driver runs three ways:

```bash
pip install -r requirements.txt
python build_data_13f.py --list-quarters     # JSON list of quarter labels
python build_data_13f.py --quarter 2018Q2    # build one quarter -> data/edgar/thirteen_f/2018Q2.parquet
python build_data_13f.py                      # build everything (sequential; slow)
```

(Swap `build_data.py` for the Form 4 pipeline; `data/edgar/form4/` cache dir.)

## Cloud run

1. Set a repo **variable** `SEC_USER_AGENT` to a descriptive string with a contact email
   (SEC requires it).
2. Run the workflow from the Actions tab, or:
   ```bash
   gh workflow run build-13f.yml                 # full 2013-07-present 13F-HR build
   gh workflow run build-13f.yml -f quarters='["2018Q2","2023Q1"]'   # surgical rebuild
   ```
3. When it finishes, download the `thirteenf_cache` artifact — a `thirteenf_cache.tar.gz`
   of all per-quarter parquet files. Unpack it into the consuming repo's cache dir.

Shared-IP note: GitHub runners share Azure egress IPs, so concurrent jobs can collide on
one IP and trip SEC's per-IP limit. `max-parallel` is kept low (12 for Form 4, 10 for the
heavier all-filer 13F parse) and the fetcher backs off; use the `quarters` input to rebuild
any quarter that came back holed.

## Output schemas

- **Form 4** — one row per Code-"P" purchase: `issuer_cik, issuer_ticker, issuer_name,
  owner_cik, owner_name, txn_date, filing_date, shares, price, is_officer, is_director,
  is_ten_pct`.
- **13F-HR** — one row per long equity holding: `filer_cik, filer_name, filing_date,
  period_of_report, cusip, name_of_issuer, value_usd, shares, is_amendment`. `value_usd` is
  normalized to whole dollars across the 2023 thousands-to-dollars `<value>` scale break.

Data is public SEC EDGAR; this tool only parses it.
