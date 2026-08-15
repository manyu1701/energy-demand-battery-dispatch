"""
Download real hourly grid data from public sources into data/raw/.

Every source here is public, documented and reusable with attribution — see
docs/DATA_SOURCES.md for licences, coverage and the trade-offs between them.
Two of the three need no account at all.

    python scripts/fetch_real_data.py opsd --zone DE
    python scripts/fetch_real_data.py energy-charts --zone DE-LU --start 2024-01-01 --end 2024-12-31
    python scripts/fetch_real_data.py open-meteo --lat 52.52 --lon 13.41 --start 2024-01-01 --end 2024-12-31
    python scripts/fetch_real_data.py eia --respondent CISO --start 2024-01-01 --end 2024-12-31

Each command writes a canonical CSV (timestamp, demand_mw, price_eur_mwh,
optionally temperature_c) that load_real_data() reads directly. `--merge`
joins a weather file onto an existing load/price file.

Nothing is committed: data/raw/ is gitignored, so downloads stay local and
this repository keeps shipping no third-party data.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import RAW_DIR, load_opsd_timeseries, normalize_hourly_frame

# The last OPSD time-series release. The platform stopped publishing new
# packages in 2020, so this URL is stable rather than stale-by-accident —
# see docs/DATA_SOURCES.md for what that means for recency.
OPSD_RELEASE = "2020-10-06"
OPSD_URL = (
    f"https://data.open-power-system-data.org/time_series/{OPSD_RELEASE}/"
    "time_series_60min_singleindex.csv"
)
ENERGY_CHARTS_BASE = "https://api.energy-charts.info"
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
EIA_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"

USER_AGENT = "energy-demand-battery-dispatch/0.1 (public data, educational use)"


def _get(url: str, params: dict | None = None, timeout: int = 120) -> bytes:
    """GET a URL, surfacing HTTP errors with the server's own explanation."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params, doseq=True)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", "replace")[:400]
        raise SystemExit(f"HTTP {err.code} from {url}\n{detail}") from err
    except urllib.error.URLError as err:
        raise SystemExit(
            f"Could not reach {url}: {err.reason}\n"
            "If you are behind a proxy or an egress allowlist, this host has to be "
            "permitted before the download can run."
        ) from err


def _get_json(url: str, params: dict | None = None) -> dict:
    return json.loads(_get(url, params))


def _write(df: pd.DataFrame, out: Path, source: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"\nWrote {len(df):,} hourly rows to {out}")
    print(f"  span    {df.index.min()} -> {df.index.max()} (UTC)")
    for column in df.columns:
        series = df[column]
        print(
            f"  {column:<16} min {series.min():>10,.1f}  mean {series.mean():>10,.1f}  "
            f"max {series.max():>10,.1f}  nulls {int(series.isna().sum())}"
        )
    print(f"\nSource: {source}")
    print("Attribution is required — see docs/DATA_SOURCES.md before publishing results.")


# ---------------------------------------------------------------- OPSD ----


def fetch_opsd(args: argparse.Namespace) -> None:
    """
    Open Power System Data — hourly load and day-ahead price, 30+ European
    zones, one wide CSV, CC-BY 4.0, no account.

    The file is large (a few hundred MB) because it holds every country; it is
    cached in data/raw/ and only the requested zone's columns are parsed.
    """
    cache = RAW_DIR / f"opsd_time_series_60min_{OPSD_RELEASE}.csv"

    if cache.exists() and not args.force:
        print(f"Using cached {cache} ({cache.stat().st_size / 1e6:.0f} MB)")
    else:
        print(f"Downloading {OPSD_URL}")
        print("  (a few hundred MB — every European zone is in one file)")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(_get(OPSD_URL, timeout=1800))
        print(f"  saved {cache.stat().st_size / 1e6:.0f} MB")

    df = load_opsd_timeseries(cache, zone=args.zone)
    if args.start:
        df = df.loc[df.index >= pd.Timestamp(args.start)]
    if args.end:
        df = df.loc[df.index <= pd.Timestamp(args.end)]

    _write(
        df,
        args.out or RAW_DIR / f"opsd_{args.zone.lower()}_hourly.csv",
        f"Open Power System Data, time_series {OPSD_RELEASE} (CC-BY 4.0), "
        "compiled from ENTSO-E Transparency Platform",
    )


# -------------------------------------------------------- energy-charts ----


def fetch_energy_charts(args: argparse.Namespace) -> None:
    """
    Fraunhofer ISE energy-charts — current load and day-ahead prices for
    European bidding zones, CC-BY 4.0, no account.

    Unlike OPSD this is maintained, so it covers recent years. Load and price
    come from separate endpoints and are joined on the hour here.
    """
    country = args.zone.split("-")[0].lower()

    print(f"Fetching load for country={country}...")
    load_payload = _get_json(
        f"{ENERGY_CHARTS_BASE}/total_power",
        {"country": country, "start": args.start, "end": args.end},
    )
    load = _extract_energy_charts_load(load_payload)

    print(f"Fetching day-ahead price for bzn={args.zone}...")
    price_payload = _get_json(
        f"{ENERGY_CHARTS_BASE}/price",
        {"bzn": args.zone, "start": args.start, "end": args.end},
    )
    price = _extract_energy_charts_price(price_payload)

    merged = pd.merge(load, price, on="timestamp", how="outer").sort_values("timestamp")
    df = normalize_hourly_frame(
        merged, "timestamp", {"demand_mw": "demand_mw", "price_eur_mwh": "price_eur_mwh"}
    )

    licence = price_payload.get("license_info") or load_payload.get("license_info") or "CC-BY 4.0"
    _write(
        df,
        args.out or RAW_DIR / f"energy_charts_{args.zone.lower().replace('-', '_')}_hourly.csv",
        f"Fraunhofer ISE energy-charts.info ({licence})",
    )


def _timestamps_from(payload: dict) -> pd.Series:
    if "unix_seconds" not in payload:
        raise SystemExit(
            f"Unexpected response shape — no `unix_seconds`. Keys: {list(payload)[:10]}"
        )
    return pd.to_datetime(pd.Series(payload["unix_seconds"]), unit="s", utc=True)


def _extract_energy_charts_load(payload: dict) -> pd.DataFrame:
    """
    Pull the load series out of a /total_power response.

    The response carries many named production types plus the load; the exact
    label has changed across API versions, so it is matched by name and the
    available options are printed on failure rather than guessed at.
    """
    timestamps = _timestamps_from(payload)
    series = payload.get("production_types") or []
    wanted = {"load", "last (netzlast)", "total load", "netzlast"}

    for entry in series:
        if str(entry.get("name", "")).strip().lower() in wanted:
            return pd.DataFrame({"timestamp": timestamps, "demand_mw": entry["data"]})

    names = [entry.get("name") for entry in series]
    raise SystemExit(
        "Could not find a load series in the /total_power response.\n"
        f"Available series: {names}\n"
        "Pick the right one and add its name to `wanted` in _extract_energy_charts_load()."
    )


def _extract_energy_charts_price(payload: dict) -> pd.DataFrame:
    if "price" not in payload:
        raise SystemExit(f"No `price` array in response. Keys: {list(payload)[:10]}")
    unit = str(payload.get("unit", "EUR/MWh"))
    if "MWh" not in unit:
        raise SystemExit(f"Unexpected price unit {unit!r}; the project assumes EUR/MWh.")
    return pd.DataFrame(
        {"timestamp": _timestamps_from(payload), "price_eur_mwh": payload["price"]}
    )


# ---------------------------------------------------------- open-meteo ----


def fetch_open_meteo(args: argparse.Namespace) -> None:
    """
    Open-Meteo ERA5 archive — hourly 2m temperature anywhere, CC-BY 4.0,
    no account. Used to add the weather features the model leans on.
    """
    print(f"Fetching ERA5 temperature for {args.lat}, {args.lon}...")
    payload = _get_json(
        OPEN_METEO_URL,
        {
            "latitude": args.lat,
            "longitude": args.lon,
            "start_date": args.start,
            "end_date": args.end,
            "hourly": "temperature_2m",
            "timezone": "UTC",
        },
    )
    hourly = payload.get("hourly")
    if not hourly or "temperature_2m" not in hourly:
        raise SystemExit(f"Unexpected response: {json.dumps(payload)[:400]}")

    raw = pd.DataFrame(
        {"timestamp": hourly["time"], "temperature_c": hourly["temperature_2m"]}
    )
    df = normalize_hourly_frame(
        raw, "timestamp", {"temperature_c": "temperature_c"}, treat_zero_as_missing=()
    )
    _write(
        df,
        args.out or RAW_DIR / f"open_meteo_{args.lat}_{args.lon}_hourly.csv",
        "Open-Meteo.com ERA5 reanalysis (CC-BY 4.0), derived from Copernicus Climate Data Store",
    )


# ----------------------------------------------------------------- EIA ----


def fetch_eia(args: argparse.Namespace) -> None:
    """
    EIA Hourly Electric Grid Monitor (Form EIA-930) — US balancing-authority
    demand. Public domain, but needs a free API key from
    https://www.eia.gov/opendata/ passed as --api-key or $EIA_API_KEY.

    Demand only: EIA-930 publishes no prices, so a US build needs a separate
    price feed (see docs/DATA_SOURCES.md).
    """
    if not args.api_key:
        raise SystemExit(
            "EIA needs a free API key. Register at https://www.eia.gov/opendata/ "
            "then pass --api-key KEY or set EIA_API_KEY."
        )

    rows: list[dict] = []
    offset, page_size = 0, 5000
    while True:
        payload = _get_json(
            EIA_URL,
            {
                "api_key": args.api_key,
                "frequency": "hourly",
                "data[0]": "value",
                "facets[respondent][]": args.respondent,
                "facets[type][]": "D",  # D = demand
                "start": f"{args.start}T00",
                "end": f"{args.end}T23",
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
                "offset": offset,
                "length": page_size,
            },
        )
        page = payload.get("response", {}).get("data", [])
        rows.extend(page)
        print(f"  fetched {len(rows):,} rows...")
        if len(page) < page_size:
            break
        offset += page_size

    if not rows:
        raise SystemExit(f"No data returned for respondent {args.respondent!r}.")

    raw = pd.DataFrame(rows)[["period", "value"]]
    df = normalize_hourly_frame(raw, "period", {"value": "demand_mw"})
    _write(
        df,
        args.out or RAW_DIR / f"eia_{args.respondent.lower()}_hourly.csv",
        f"U.S. EIA Hourly Electric Grid Monitor, Form EIA-930, respondent {args.respondent} "
        "(public domain)",
    )


# --------------------------------------------------------------- merge ----


def merge_files(args: argparse.Namespace) -> None:
    """Join a weather file onto a load/price file on the hourly UTC index."""
    left = pd.read_csv(args.merge, parse_dates=["timestamp"]).set_index("timestamp")
    right = pd.read_csv(args.With, parse_dates=["timestamp"]).set_index("timestamp")

    merged = left.join(right, how="inner", rsuffix="_weather")
    if merged.empty:
        raise SystemExit(
            "The two files do not overlap in time:\n"
            f"  {args.merge}: {left.index.min()} -> {left.index.max()}\n"
            f"  {args.With}: {right.index.min()} -> {right.index.max()}"
        )

    out = args.out or args.merge
    merged.to_csv(out)
    print(f"Merged {len(merged):,} overlapping hours -> {out}")
    print(f"  columns: {list(merged.columns)}")


# ---------------------------------------------------------------- main ----


def main() -> None:
    default_end = date.today() - timedelta(days=7)  # ERA5 lags ~5-7 days
    default_start = default_end.replace(year=default_end.year - 1)

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="source", required=True)

    def add_common(p, *, dates=True):
        p.add_argument("--out", type=Path, help="output CSV path (default: data/raw/<source>.csv)")
        if dates:
            p.add_argument("--start", default=str(default_start), help="YYYY-MM-DD")
            p.add_argument("--end", default=str(default_end), help="YYYY-MM-DD")

    opsd = sub.add_parser("opsd", help="Open Power System Data (EU load + price, no account)")
    opsd.add_argument("--zone", default="DE", help="bidding zone/country code, e.g. DE, FR, ES")
    opsd.add_argument("--force", action="store_true", help="re-download even if cached")
    opsd.add_argument("--start", help="YYYY-MM-DD (optional filter)")
    opsd.add_argument("--end", help="YYYY-MM-DD (optional filter)")
    opsd.add_argument("--out", type=Path)
    opsd.set_defaults(func=fetch_opsd)

    ec = sub.add_parser("energy-charts", help="Fraunhofer ISE (EU load + price, no account)")
    ec.add_argument("--zone", default="DE-LU", help="bidding zone, e.g. DE-LU, FR, NL")
    add_common(ec)
    ec.set_defaults(func=fetch_energy_charts)

    om = sub.add_parser("open-meteo", help="ERA5 hourly temperature (no account)")
    om.add_argument("--lat", type=float, required=True)
    om.add_argument("--lon", type=float, required=True)
    add_common(om)
    om.set_defaults(func=fetch_open_meteo)

    eia = sub.add_parser("eia", help="US balancing-authority demand (free API key)")
    eia.add_argument("--respondent", default="CISO", help="balancing authority, e.g. CISO, PJM, ERCO")
    eia.add_argument("--api-key", default=None, help="or set $EIA_API_KEY")
    add_common(eia)
    eia.set_defaults(func=fetch_eia)

    merge = sub.add_parser("merge", help="join a weather file onto a load/price file")
    merge.add_argument("--merge", type=Path, required=True, help="load/price CSV")
    merge.add_argument("--with", dest="With", type=Path, required=True, help="weather CSV")
    merge.add_argument("--out", type=Path)
    merge.set_defaults(func=merge_files)

    args = parser.parse_args()
    if args.source == "eia" and not args.api_key:
        import os

        args.api_key = os.environ.get("EIA_API_KEY")

    args.func(args)


if __name__ == "__main__":
    main()
