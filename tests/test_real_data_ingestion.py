"""
Tests for the real-data ingestion layer.

These use fixtures shaped like the real providers' documented output rather
than live downloads, so the suite stays offline and deterministic. They cover
the failure modes real grid data actually has and synthetic data never does:
DST folds, missing hours, duplicate timestamps, sub-hourly feeds, and outages
reported as zero.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.loader import (
    DataQualityError,
    data_quality_report,
    load_opsd_timeseries,
    normalize_hourly_frame,
)


def _hours(start, n, freq="h"):
    return pd.date_range(start, periods=n, freq=freq, tz="UTC")


def _frame(timestamps, **columns):
    return pd.DataFrame({"utc_timestamp": timestamps, **columns})


RENAME = {"load": "demand_mw", "price": "price_eur_mwh"}


def _clean_frame(n=240):
    timestamps = _hours("2024-01-01", n)
    return _frame(
        timestamps,
        load=np.linspace(1000, 1200, n),
        price=np.linspace(40, 90, n),
    )


def test_normalizes_a_clean_frame():
    out = normalize_hourly_frame(_clean_frame(), "utc_timestamp", RENAME)
    assert list(out.columns) == ["demand_mw", "price_eur_mwh"]
    assert len(out) == 240
    assert out.index.is_monotonic_increasing
    assert not out.index.has_duplicates
    assert out.index.tz is None  # naive UTC, matching the rest of the project


def test_missing_hours_are_restored():
    frame = _clean_frame()
    frame = frame.drop(index=[10, 11])  # two-hour outage
    out = normalize_hourly_frame(frame, "utc_timestamp", RENAME)

    assert len(out) == 240, "the gap should be reindexed back in, not skipped"
    assert out["demand_mw"].isna().sum() == 0, "a 2h gap is short enough to interpolate"


def test_long_gaps_are_left_as_null_not_invented():
    frame = _clean_frame(720)
    frame = frame.drop(index=range(20, 40))  # 20-hour outage
    out = normalize_hourly_frame(frame, "utc_timestamp", RENAME, max_gap_hours=3)

    # Interpolation reaches 3 hours in from each edge; the rest stays unknown.
    assert out["demand_mw"].isna().sum() == 20 - 2 * 3
    assert len(out) == 720


def test_refuses_a_series_that_is_mostly_missing():
    timestamps = _hours("2024-01-01", 240)
    frame = _frame(timestamps, load=np.linspace(1000, 1200, 240), price=40.0)
    frame.loc[frame.index[50:], "load"] = np.nan

    with pytest.raises(DataQualityError, match="still missing"):
        normalize_hourly_frame(frame, "utc_timestamp", RENAME)


def test_duplicate_timestamps_are_collapsed():
    """
    The autumn DST fold and re-published corrections both produce exact
    duplicate timestamps. A duplicated index silently corrupts every lag
    feature downstream.
    """
    frame = _clean_frame(48)
    duplicated = pd.concat([frame, frame.iloc[10:12]], ignore_index=True)

    out = normalize_hourly_frame(duplicated, "utc_timestamp", RENAME)
    assert not out.index.has_duplicates
    assert len(out) == 48


def test_dst_transition_does_not_duplicate_or_skip_an_hour():
    """
    Europe/Berlin repeats 02:00 on the October change and skips it in March.
    Read as UTC, 24 October hours stay 24 distinct hours.
    """
    autumn = _hours("2024-10-27", 48)  # EU clocks go back on this date
    frame = _frame(autumn, load=np.linspace(1000, 1100, 48), price=50.0)

    out = normalize_hourly_frame(frame, "utc_timestamp", RENAME)
    assert len(out) == 48
    assert not out.index.has_duplicates
    assert (out.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()


def test_unparseable_timestamps_are_dropped():
    frame = _clean_frame(24)
    # Providers ship timestamps as text; a corrupt row is a string, not a NaT.
    frame["utc_timestamp"] = frame["utc_timestamp"].astype(str)
    frame.loc[5, "utc_timestamp"] = "not-a-timestamp"

    out = normalize_hourly_frame(frame, "utc_timestamp", RENAME)
    # The bad row is dropped, then restored as a gap by the reindex.
    assert len(out) == 24
    assert out.index.is_monotonic_increasing


def test_sub_hourly_feeds_are_averaged_to_hourly():
    """energy-charts publishes 15-minute prices; the project models hours."""
    quarters = _hours("2024-01-01", 96, freq="15min")
    frame = _frame(quarters, load=1000.0, price=np.tile([10.0, 20.0, 30.0, 40.0], 24))

    out = normalize_hourly_frame(frame, "utc_timestamp", RENAME)
    assert len(out) == 24
    assert out["price_eur_mwh"].iloc[0] == pytest.approx(25.0)  # mean of the four quarters


def test_zero_demand_is_treated_as_missing():
    """ENTSO-E reports failed submissions as 0 MW, not null."""
    frame = _clean_frame(48)
    frame.loc[20, "load"] = 0.0

    out = normalize_hourly_frame(frame, "utc_timestamp", RENAME)
    assert out["demand_mw"].iloc[20] > 500, "a spurious zero should be repaired, not modelled"


def test_zero_prices_are_preserved():
    """Zero and negative prices are real market outcomes, not bad data."""
    frame = _clean_frame(48)
    frame.loc[20, "price"] = 0.0
    frame.loc[21, "price"] = -35.0

    out = normalize_hourly_frame(frame, "utc_timestamp", RENAME)
    assert out["price_eur_mwh"].iloc[20] == 0.0
    assert out["price_eur_mwh"].iloc[21] == -35.0


def test_missing_source_columns_are_reported_clearly():
    with pytest.raises(KeyError, match="not found"):
        normalize_hourly_frame(_clean_frame(), "utc_timestamp", {"nope": "demand_mw"})


def test_missing_timestamp_column_is_reported_clearly():
    with pytest.raises(KeyError, match="timestamp"):
        normalize_hourly_frame(_clean_frame(), "no_such_column", RENAME)


# ------------------------------------------------------------------ OPSD ---


def _write_opsd(path, columns: dict, n=72):
    """Write a miniature file shaped like OPSD's wide time_series export."""
    timestamps = _hours("2024-01-01", n)
    frame = pd.DataFrame(
        {
            "utc_timestamp": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cet_cest_timestamp": timestamps.tz_convert("Europe/Berlin").strftime(
                "%Y-%m-%dT%H:%M:%S%z"
            ),
            **{name: values for name, values in columns.items()},
        }
    )
    frame.to_csv(path, index=False)
    return path


def test_opsd_extracts_the_requested_zone(tmp_path):
    path = _write_opsd(
        tmp_path / "opsd.csv",
        {
            "DE_load_actual_entsoe_transparency": np.linspace(40000, 60000, 72),
            "DE_LU_price_day_ahead": np.linspace(20, 90, 72),
            "FR_load_actual_entsoe_transparency": np.linspace(50000, 70000, 72),
            "FR_price_day_ahead": np.linspace(25, 95, 72),
        },
    )

    german = load_opsd_timeseries(path, zone="DE")
    french = load_opsd_timeseries(path, zone="FR")

    assert list(german.columns) == ["demand_mw", "price_eur_mwh"]
    assert german["demand_mw"].iloc[0] == pytest.approx(40000)
    assert french["demand_mw"].iloc[0] == pytest.approx(50000)


def test_opsd_resolves_the_pre_split_german_price_column(tmp_path):
    """
    Before the bidding-zone split on 2018-10-01 the German price lives under
    DE_AT_LU_price_day_ahead. Hardcoding DE_LU_ silently loses the earlier
    years, which is most of the usable history.
    """
    path = _write_opsd(
        tmp_path / "opsd_old.csv",
        {
            "DE_load_actual_entsoe_transparency": np.linspace(40000, 60000, 72),
            "DE_AT_LU_price_day_ahead": np.linspace(20, 90, 72),
        },
    )

    out = load_opsd_timeseries(path, zone="DE")
    assert out["price_eur_mwh"].iloc[0] == pytest.approx(20)


def test_opsd_falls_back_to_power_statistics_load(tmp_path):
    path = _write_opsd(
        tmp_path / "opsd_ps.csv",
        {
            "DE_load_actual_entsoe_power_statistics": np.linspace(40000, 60000, 72),
            "DE_LU_price_day_ahead": np.linspace(20, 90, 72),
        },
    )
    assert load_opsd_timeseries(path, zone="DE")["demand_mw"].iloc[0] == pytest.approx(40000)


def test_opsd_lists_available_zones_when_one_is_missing(tmp_path):
    path = _write_opsd(
        tmp_path / "opsd.csv",
        {
            "DE_load_actual_entsoe_transparency": np.linspace(40000, 60000, 72),
            "DE_LU_price_day_ahead": np.linspace(20, 90, 72),
        },
    )

    with pytest.raises(KeyError) as excinfo:
        load_opsd_timeseries(path, zone="XX")
    assert "DE" in str(excinfo.value), "the error should name the zones that do exist"


def test_opsd_missing_file_names_the_fetch_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_real_data.py"):
        load_opsd_timeseries(tmp_path / "absent.csv")


# -------------------------------------------------------- quality report ---


def test_quality_report_counts_what_matters():
    frame = _clean_frame(48)
    out = normalize_hourly_frame(frame, "utc_timestamp", RENAME)
    out.iloc[5, out.columns.get_loc("demand_mw")] = np.nan

    report = data_quality_report(out)
    assert report["rows"] == 48
    assert report["expected_hours"] == 48
    assert report["duplicated_timestamps"] == 0
    assert report["columns"]["demand_mw"]["nulls"] == 1
    assert report["columns"]["price_eur_mwh"]["nulls"] == 0
    assert report["columns"]["demand_mw"]["mean"] > 0


def test_normalized_real_data_feeds_the_feature_builder():
    """
    The whole point of normalising: a real download has to drop straight into
    the existing pipeline with no special-casing.
    """
    from src.features.build_features import build_feature_matrix, feature_columns

    timestamps = _hours("2024-01-01", 24 * 30)
    rng = np.random.default_rng(0)
    frame = _frame(
        timestamps,
        load=1000 + 200 * np.sin(np.arange(720) * 2 * np.pi / 24) + rng.normal(0, 20, 720),
        price=50 + 20 * np.sin(np.arange(720) * 2 * np.pi / 24),
    )

    normalized = normalize_hourly_frame(frame, "utc_timestamp", RENAME)
    matrix = build_feature_matrix(normalized, "demand_mw")

    assert len(matrix) == 720 - 168
    assert not matrix.isna().any().any()
    assert "lag_24h" in feature_columns(matrix, "demand_mw")
