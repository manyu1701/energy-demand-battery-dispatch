# Real data sources

Vetted public sources for the two series this project needs — **hourly
electricity load** and **hourly day-ahead price** — plus temperature for the
weather features.

Selection criteria: a named institutional publisher, an explicit reuse
licence, hourly (or finer) resolution, and a documented, stable access path.
Scraped aggregators and unlicensed mirrors are excluded no matter how
convenient, because a forecast is only as defensible as its inputs.

## Recommendation

| Need | Use | Account |
|---|---|---|
| First real run, EU, load + price in one file | **Open Power System Data** | none |
| Recent years (2021→now), EU | **Fraunhofer ISE energy-charts** | none |
| Temperature for weather features | **Open-Meteo ERA5 archive** | none |
| Authoritative EU source of record | **ENTSO-E Transparency** | free registration |
| US demand | **EIA Form EIA-930** | free API key |

Start with OPSD: one download gives load *and* price on the same index for
30+ zones, which no other keyless source does. Move to energy-charts when you
need data past 2020.

```bash
python scripts/fetch_real_data.py opsd --zone DE
python scripts/fetch_real_data.py open-meteo --lat 52.52 --lon 13.41 \
    --start 2015-01-01 --end 2020-09-30
python scripts/fetch_real_data.py merge \
    --merge data/raw/opsd_de_hourly.csv \
    --with  data/raw/open_meteo_52.52_13.41_hourly.csv
```

---

## 1. Open Power System Data — *primary recommendation*

Hourly load, day-ahead prices, and wind/solar generation for ~37 European
countries, published as one wide CSV.

- **Data:** <https://data.open-power-system-data.org/time_series/>
- **Package:** `time_series`, final release `2020-10-06`
- **File:** `time_series_60min_singleindex.csv` (a few hundred MB — every
  country is in the one file)
- **Licence:** CC-BY 4.0. Attribution required; redistribution permitted.
- **Publisher:** a research consortium (Neon Neue Energieökonomik, TU Berlin,
  DIW and others), funded by the German Federal Ministry for Economic Affairs.
- **Upstream:** ENTSO-E Transparency Platform and ENTSO-E Power Statistics.

**Why first.** It is the only keyless source that hands you load and price
already aligned on a single UTC index. It is also *documented* — the
processing notebooks that built it are public on GitHub, so every column's
provenance is traceable, which matters more than freshness for a modelling
project.

**The catch: it stopped updating in 2020.** The platform is in maintenance
mode. For most zones you get roughly 2015 → September 2020 of paired
load+price, with some load series reaching back to 2005. That is five years
of real hourly data — plenty to train and backtest on, and it does not go
stale for that purpose. It is only a problem if you need *current* prices.

**Two traps the loader handles for you** (`load_opsd_timeseries`):

- Germany's price column is `DE_LU_price_day_ahead` only *after* the
  Germany/Austria bidding-zone split on 2018-10-01. Before that the same
  series is `DE_AT_LU_price_day_ahead`. Hardcoding the first silently
  discards most of the usable history.
- Load is published by two ENTSO-E systems with different coverage —
  `_entsoe_transparency` and `_entsoe_power_statistics`. Transparency is
  preferred, Power Statistics is the fallback.

## 2. Fraunhofer ISE — energy-charts

Current European load, generation by source, and day-ahead prices for 40+
countries and bidding zones.

- **API:** <https://api.energy-charts.info/> — no registration, no token
- **Endpoints:** `/total_power` (load and generation), `/price` (day-ahead by
  bidding zone `bzn`), plus cross-border flows and installed capacity
- **Licence:** CC-BY 4.0 for most series; each response carries its own
  `license_info` string, which the fetch script prints
- **Publisher:** Fraunhofer Institute for Solar Energy Systems ISE

**Why second.** This is the maintained answer to OPSD's staleness — same
underlying ENTSO-E data, kept current. The cost is that load and price come
from separate endpoints and must be joined, and the response schema has
shifted across API versions, so the adapter matches the load series *by name*
and fails loudly listing what it found rather than guessing.

Prices are returned at 15-minute resolution (European day-ahead markets moved
to a 15-minute market time unit). `normalize_hourly_frame()` averages the four
quarters into each hour, which is what an hourly day-ahead price means.

## 3. Open-Meteo — ERA5 archive (temperature)

- **API:** `https://archive-api.open-meteo.com/v1/archive` — no key
- **Parameters:** `latitude`, `longitude`, `start_date`, `end_date`,
  `hourly=temperature_2m`, `timezone=UTC`
- **Licence:** CC-BY 4.0, free for non-commercial use
- **Underlying data:** ECMWF ERA5 reanalysis via the Copernicus Climate Data
  Store — hourly, spatially complete, no missing values, 1940 → present at
  roughly 0.1–0.25° resolution, with a ~5 day lag

The ablation in the backtest panel shows weather is the *most* load-bearing
feature group for demand, so this is not optional if you want the real-data
model to behave like the synthetic one.

One caveat: a single lat/lon is a proxy for a whole control area. Real
utilities use population-weighted temperature across several cities. For a
single-zone model one big load centre (Berlin for DE, Paris for FR) is a
reasonable approximation — and an honest place to improve later.

## 4. ENTSO-E Transparency Platform — the source of record

- **Portal:** <https://transparency.entsoe.eu/>
- **Access:** free account, then request a RESTful API security token
- **Licence:** reuse permitted with attribution
- **Coverage:** all EU TSOs, 2015 → present. Actual total load [6.1.A/B] and
  day-ahead prices [12.1.D]

Both sources above are derived from this one. Go direct when you need current
authoritative data, a zone the aggregators skip, or series they do not
republish. No adapter ships here because it needs a credential — the token
flow is a few lines with the `entsoe-py` client once you have registered.

## 5. EIA Form EIA-930 (United States)

- **API:** `https://api.eia.gov/v2/electricity/rto/region-data/data/`
- **Access:** free API key from <https://www.eia.gov/opendata/>
- **Licence:** US Government work — public domain
- **Coverage:** hourly demand per balancing authority (CISO, PJM, ERCO, MISO…)
  from 2015, published ~1 hour after the hour

**Demand only — EIA-930 publishes no prices.** A US build needs a separate
price feed, and that is where it gets awkward: each ISO publishes LMPs on its
own portal in its own format (CAISO OASIS, ERCOT market reports, MISO data
exchange), with varying terms. This is why the European sources are the
recommended starting point despite the OPSD recency limit.

---

## What real data will change

Worth expecting before you run it, so the results are not a surprise:

- **Accuracy will drop.** The synthetic price process is a merit-order model
  designed to be learnable. Real prices carry fuel shocks, outages, and
  strategic bidding that no calendar feature predicts. Expect price WMAPE
  materially worse than the synthetic 13%; demand should hold up better,
  since real load genuinely is mostly calendar plus weather.
- **The dispatch gap will widen.** Lower forecast skill means the
  forecast-driven battery captures a smaller share of the perfect-foresight
  ceiling. That number dropping *is the finding*, not a regression.
- **The data will be dirty.** Missing hours, DST folds, duplicated
  timestamps, sub-hourly feeds, and outages reported as `0 MW` rather than
  null. `normalize_hourly_frame()` handles all five and refuses a series that
  is more than 5% missing after interpolation, rather than quietly modelling
  a gap-filled fiction. Run `data_quality_report()` on any fresh download
  before trusting a result computed from it.

## Attribution

Any published result must credit the source. Minimum text:

> Load and price data from Open Power System Data (CC-BY 4.0), compiled from
> the ENTSO-E Transparency Platform. Temperature from Open-Meteo.com
> (CC-BY 4.0), derived from ECMWF ERA5 via the Copernicus Climate Data Store.

Nothing third-party is committed to this repository. `data/raw/` is
gitignored, so downloads stay on your machine and the licences above stay
yours to comply with.

## Verification status

The adapters in `scripts/fetch_real_data.py` were written against each
provider's documented schema and are unit-tested against fixtures shaped like
that documented output (`tests/test_real_data_ingestion.py`, 19 tests).

**The live request paths have not been exercised.** The environment this was
built in blocks outbound HTTPS to every host above at the egress proxy, so no
download could be attempted. Expect to adjust field names on first run —
particularly the energy-charts load-series label, which is matched by name
and will print the available options if it misses.
