import numpy as np
import pytest

from src.backtesting.harness import (
    evaluate_all_regimes,
    feature_ablation_splits,
    random_split,
    recency_weighted_split,
    short_horizon_split,
    temporal_split,
)
from src.data.loader import generate_hourly_dataset
from src.features.build_features import build_feature_matrix


@pytest.fixture(scope="module")
def matrix():
    df = generate_hourly_dataset(n_days=45, seed=13)
    return build_feature_matrix(df, "demand_mw")


def test_temporal_split_is_chronological(matrix):
    train, test = temporal_split(matrix, test_size=0.25)
    assert train.index.max() < test.index.min()
    assert len(train) + len(test) == len(matrix)
    assert len(test) == pytest.approx(len(matrix) * 0.25, rel=0.02)


def test_temporal_split_rejects_a_bad_test_size(matrix):
    for bad in (0, 1, -0.1, 1.5):
        with pytest.raises(ValueError):
            temporal_split(matrix, test_size=bad)


def test_random_split_partitions_without_overlap(matrix):
    train, test = random_split(matrix, test_size=0.2, seed=7)
    assert len(train) + len(test) == len(matrix)
    assert not train.index.intersection(test.index).size


def test_random_split_is_reproducible(matrix):
    first, _ = random_split(matrix, seed=7)
    second, _ = random_split(matrix, seed=7)
    assert first.index.equals(second.index)


def test_recency_weights_favour_recent_rows(matrix):
    train, _, weights = recency_weighted_split(matrix, test_size=0.2, half_life_days=10)
    assert len(weights) == len(train)
    assert weights[-1] == pytest.approx(1.0)  # newest row is the reference
    assert weights[0] < weights[-1]
    assert (np.diff(weights) >= -1e-9).all()  # monotonically increasing with recency


def test_recency_half_life_is_honoured(matrix):
    train, _, weights = recency_weighted_split(matrix, test_size=0.2, half_life_days=10)
    age_days = (train.index.max() - train.index).total_seconds() / 86_400
    ten_days_back = int(np.argmin(np.abs(age_days - 10)))
    assert weights[ten_days_back] == pytest.approx(0.5, rel=0.02)


def test_short_horizon_split_keeps_only_the_final_window(matrix):
    train, test = short_horizon_split(matrix, horizon_hours=24)
    assert len(test) == 24
    assert len(train) == len(matrix) - 24
    assert train.index.max() < test.index.min()


def test_short_horizon_split_validates_the_horizon(matrix):
    with pytest.raises(ValueError):
        short_horizon_split(matrix, horizon_hours=0)
    with pytest.raises(ValueError):
        short_horizon_split(matrix, horizon_hours=len(matrix) + 1)


def test_ablation_drops_exactly_one_group_at_a_time(matrix):
    splits = feature_ablation_splits(matrix, "demand_mw", test_size=0.2)
    assert "all_features" in splits

    reference = splits["all_features"][2]
    for label, (_, _, cols) in splits.items():
        if label == "all_features":
            continue
        assert set(cols) < set(reference), f"{label} did not drop any feature"

    assert not any(c.startswith("lag_") for c in splits["without_lag"][2])
    assert not any(c.startswith("roll_") for c in splits["without_rolling"][2])


def test_ablation_splits_share_one_chronological_split(matrix):
    splits = feature_ablation_splits(matrix, "demand_mw", test_size=0.2)
    train_indexes = [train.index for train, _, _ in splits.values()]
    assert all(index.equals(train_indexes[0]) for index in train_indexes)


def test_evaluate_all_regimes_reports_every_regime(matrix):
    results = evaluate_all_regimes(matrix, "demand_mw", test_size=0.2, horizon_hours=24)
    regimes = set(results["regime"])

    assert {"temporal", "random", "recency_weighted", "short_horizon_24h"} <= regimes
    assert any(name.startswith("ablation_") for name in regimes)
    assert (results["wmape"] > 0).all()
    assert (results["n_test"] > 0).all()
    assert set(results.columns) >= {"wmape", "bias", "mae", "rmse", "baseline_wmape", "skill_vs_naive"}


def test_random_split_looks_easier_than_temporal(matrix):
    """
    Random holdout is optimistic by construction — it lets the model
    interpolate between hours it has already seen. If this ever inverts, the
    temporal split is leaking future information.
    """
    results = evaluate_all_regimes(matrix, "demand_mw", test_size=0.2).set_index("regime")
    assert results.loc["random", "wmape"] < results.loc["temporal", "wmape"]


def test_evaluate_all_regimes_rejects_a_bad_index(matrix):
    with pytest.raises(TypeError):
        evaluate_all_regimes(matrix.reset_index(), "demand_mw")
