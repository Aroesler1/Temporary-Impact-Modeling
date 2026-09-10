"""Convert an identified return response into a price-level response.

If r[t] = sum_l b[l] f(v[t-l]), one isolated unit of transformed flow moves
the log price by sum_{j=0}^l b[j] through lag l. Nothing here identifies the
response beyond the fitted lags or the causal effect of an additional trade.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable


@dataclass(frozen=True)
class LevelResponse:
    """Typed price-level response with an explicit beyond-fit tail policy."""

    values: tuple[float, ...]
    tail_policy: str
    provenance: str

    def for_horizon(self, horizon: int) -> list[float]:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if horizon <= len(self.values):
            return list(self.values[:horizon])
        if self.tail_policy == "constant":
            tail = self.values[-1]
        elif self.tail_policy == "zero":
            tail = 0.0
        elif self.tail_policy == "refuse":
            raise ValueError(
                "level response does not identify the requested scheduling horizon"
            )
        else:
            raise ValueError(f"unknown tail policy: {self.tail_policy}")
        return [*self.values, *([tail] * (horizon - len(self.values)))]


def return_to_level_response(coefficients: Iterable[float]) -> list[float]:
    """Accumulate finite coefficients over their observed support only.

    A zero return coefficient means no further price change, not zero price
    displacement. No zero padding or extrapolation is applied to either object.
    """
    values = [float(value) for value in coefficients]
    if not values or not all(isfinite(value) for value in values):
        raise ValueError("return coefficients must be nonempty and finite")
    level, result = 0.0, []
    for value in values:
        level += value
        if not isfinite(level):
            raise ValueError("cumulative response is not finite")
        result.append(level)
    return result


def level_response_from_returns(coefficients: Iterable[float]) -> LevelResponse:
    """Convert fitted return coefficients to a typed level response.

    A finite distributed-lag return model sets omitted later return responses
    to zero. Its implied price displacement therefore stays constant after the
    fitted support. This is a model implication, not evidence that the fitted
    response is causal or suitable for execution.
    """
    values = return_to_level_response(coefficients)
    return LevelResponse(tuple(values), "constant", "fitted_return_coefficients")


def hypothetical_level_response(
    values: Iterable[float],
    *,
    tail_policy: str,
) -> LevelResponse:
    """Construct an explicit theoretical level response for algebra checks."""
    checked = [float(value) for value in values]
    if not checked or not all(isfinite(value) for value in checked):
        raise ValueError("level response must be nonempty and finite")
    if tail_policy not in {"constant", "zero", "refuse"}:
        raise ValueError("tail_policy must be constant, zero, or refuse")
    return LevelResponse(tuple(checked), tail_policy, "explicit_hypothetical")
