"""
Multi-regime backtesting harness — Phase 1 (not yet built).

Planned split regimes, mirroring standard practice for evaluating demand
forecasting models robustly rather than on a single lucky train/test split:

  - temporal:          chronological train/test split
  - random:            random holdout across the full history
  - recency_weighted:  evaluation weights recent periods more heavily
  - short_horizon:     evaluate only on the next N hours/days ahead
  - feature_ablation:  compare performance with/without specific feature groups

Each split function will return train/test indices (or DataFrames), and a
shared evaluate_all_regimes() will report WMAPE + bias per regime so the
model's robustness — not just its best-case accuracy — is visible.
"""

# TODO: temporal_split(df, test_size) -> (train, test)
# TODO: random_split(df, test_size, seed) -> (train, test)
# TODO: recency_weighted_split(df, ...) -> (train, test, sample_weight)
# TODO: short_horizon_split(df, horizon) -> (train, test)
# TODO: feature_ablation_split(df, feature_groups) -> {group_name: (train, test)}
# TODO: evaluate_all_regimes(model_fn, df, target_col) -> pd.DataFrame of results
