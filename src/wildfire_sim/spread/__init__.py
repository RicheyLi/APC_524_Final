"""Spread rules: swappable models that turn grid + weather into a rate-of-spread field.

Rules register themselves with :func:`register` and are looked up by name from
``ScenarioConfig.spread.rule``.

Available rules (implemented in later work packages):

* ``constant`` - fixed rate in every direction; Drossel-Schwabl-style baseline (WP 1.1)
* ``alexandridis`` - vegetation/density/wind/slope factors, Alexandridis et al. 2008 (WP 1.2)
* ``rothermel`` - Rothermel surface fire via ``pyrothermel`` + elliptical spread (WP 2.1)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from wildfire_sim.spread.base import (
    SpreadOutput,
    SpreadRule,
    ignition_probability,
    max_stable_dt,
    neighbour_ignition_probability,
)

SpreadRuleFactory = Callable[..., SpreadRule]
"""Callable taking keyword parameters (from config) and returning a rule instance."""

_REGISTRY: dict[str, SpreadRuleFactory] = {}


def register(name: str) -> Callable[[SpreadRuleFactory], SpreadRuleFactory]:
    """Class decorator that makes a rule available as ``name`` to :func:`make_spread_rule`."""

    def deco(factory: SpreadRuleFactory) -> SpreadRuleFactory:
        if name in _REGISTRY:
            raise ValueError(f"spread rule {name!r} already registered")
        _REGISTRY[name] = factory
        return factory

    return deco


def available_rules() -> list[str]:
    return sorted(_REGISTRY)


def make_spread_rule(name: str, **params: Any) -> SpreadRule:
    """Instantiate the rule registered as ``name`` with ``params`` as keyword arguments."""
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown spread rule {name!r}; available: {available_rules()}") from None
    return factory(**params)


__all__ = [
    "SpreadOutput",
    "SpreadRule",
    "available_rules",
    "ignition_probability",
    "make_spread_rule",
    "max_stable_dt",
    "neighbour_ignition_probability",
    "register",
]
