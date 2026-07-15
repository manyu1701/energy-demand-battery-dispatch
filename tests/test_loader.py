from src.data.loader import generate_synthetic_sample


def test_shape_and_columns():
    df = generate_synthetic_sample(n_days=7)
    assert len(df) == 7 * 24
    assert {"demand_mw", "price_eur_mwh", "temperature_c", "is_weekend"}.issubset(df.columns)


def test_demand_positive():
    df = generate_synthetic_sample(n_days=10)
    assert (df["demand_mw"] > 0).all()


def test_weekday_weekend_pattern():
    df = generate_synthetic_sample(n_days=14, seed=1)
    weekday_mean = df.loc[~df["is_weekend"], "demand_mw"].mean()
    weekend_mean = df.loc[df["is_weekend"], "demand_mw"].mean()
    assert weekday_mean > weekend_mean


def test_reproducible_with_seed():
    df1 = generate_synthetic_sample(n_days=5, seed=99)
    df2 = generate_synthetic_sample(n_days=5, seed=99)
    assert df1.equals(df2)
