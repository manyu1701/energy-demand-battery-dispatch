"""
The nightly job: forecast tomorrow, then plan where the lifts should wait.

This is what a building would actually run — once a day, after midnight, on
the call log accumulated so far:

    history -> features -> demand forecast for tomorrow -> parking plan

`plan_day()` returns a schedule that names, for each hour, the floor each lift
should idle on. That is the artefact you hand to the lift AMC engineer, who can
enter it as the group controller's parking/homing configuration.

Keeping the forecast and the optimization behind one function matters for
honesty as much as convenience: the plan must be built from a forecast that
could only have used data available the night before, never from the demand it
is later scored against.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.lifts.building import BuildingConfig
from src.lifts.features import build_features, build_future_frame
from src.lifts.model import CallDemandForecaster, temporal_split
from src.lifts.placement import (
    PlacementConfig,
    PlacementResult,
    evaluate_placement,
    optimize_day_placement,
)


@dataclass
class DayPlan:
    date: pd.Timestamp
    result: PlacementResult
    forecast_demand: np.ndarray
    index: pd.DatetimeIndex

    @property
    def park_floors(self) -> np.ndarray:
        """(n_slots, n_lifts) array of parking floors, for scoring and plotting."""
        return self.result.schedule.filter(like="_floor").to_numpy()


def train_forecaster(
    history: pd.DataFrame,
    building: BuildingConfig,
    test_days: int = 14,
    slot_minutes: int = 60,
    **model_kwargs,
) -> tuple[CallDemandForecaster, dict]:
    """
    Fit the demand model and report its accuracy on a chronological hold-out.

    Returns the model refit on the full history — the hold-out exists to size
    the error you should expect, not to be permanently withheld from a model
    that will run in production tomorrow.
    """
    featured = build_features(history, building, slot_minutes)
    train, test = temporal_split(featured, test_days=test_days)
    metrics = CallDemandForecaster(**model_kwargs).fit(train).evaluate(test)
    return CallDemandForecaster(**model_kwargs).fit(featured), metrics


def plan_day(
    forecaster: CallDemandForecaster,
    history: pd.DataFrame,
    date: str | pd.Timestamp,
    building: BuildingConfig,
    config: Optional[PlacementConfig] = None,
    holidays: Iterable[str] = (),
) -> DayPlan:
    """
    Forecast one day's per-floor call demand and solve that day's parking plan.

    `history` must end before `date` — everything the forecast leans on (the
    lag and rolling features) is drawn from it.
    """
    config = config or PlacementConfig()
    date = pd.Timestamp(date).normalize()
    if history["slot_start"].max() >= date:
        raise ValueError(
            f"history runs to {history['slot_start'].max()}, which is not before {date}: "
            "planning a day the model has already seen would flatter the results"
        )

    frame = build_future_frame(
        history, date, building, slot_minutes=config.slot_minutes, holidays=tuple(holidays)
    )
    index, forecast = forecaster.predict_demand_matrix(frame, building)
    result = optimize_day_placement(forecast, building, config, index=index)
    return DayPlan(date=date, result=result, forecast_demand=forecast, index=index)


def score_day_plan(
    plan: DayPlan,
    actual_demand: np.ndarray,
    building: BuildingConfig,
    config: Optional[PlacementConfig] = None,
) -> dict:
    """Score a plan built on the forecast against the day's actual calls."""
    return evaluate_placement(actual_demand, plan.park_floors, building, config)
