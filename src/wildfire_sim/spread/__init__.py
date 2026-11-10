"""Spread rules: swappable models that turn grid + weather into a rate-of-spread field.

Rules register themselves with :func:`register` and are looked up by name from
``ScenarioConfig.spread.rule``. The rule named ``x`` lives in ``wildfire_sim/spread/x.py``;
:func:`make_spread_rule` imports that module on first use.

Available rules (implemented in later work packages):

* ``constant`` - fixed rate in every direction; Drossel-Schwabl-style baseline (WP 1.1)
* ``alexandridis`` - vegetation/density/wind/slope factors, Alexandridis et al. 2008 (WP 1.2)
* ``rothermel`` - Rothermel surface fire via ``pyrothermel`` + elliptical spread (WP 2.1)
"""

from __future__ import annotations

import importlib
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


def register[F: SpreadRuleFactory](name: str) -> Callable[[F], F]:
    """Class decorator that makes a rule available as ``name`` to :func:`make_spread_rule`.

    A rule registered as ``name`` must live in the module ``wildfire_sim.spread.<name>`` so
    that :func:`make_spread_rule` can find it without the caller importing it first. The
    decorated class is returned unchanged.
    """

    def deco(factory: F) -> F:
        if name in _REGISTRY:
            raise ValueError(f"spread rule {name!r} already registered")
        _REGISTRY[name] = factory
        return factory

    return deco


def available_rules() -> list[str]:
    """Names of the rules registered so far (rule modules register on import)."""
    return sorted(_REGISTRY)


def make_spread_rule(name: str, **params: Any) -> SpreadRule:
    """Instantiate the rule registered as ``name`` with ``params`` as keyword arguments.

    If ``name`` is not registered yet, the module ``wildfire_sim.spread.<name>`` is imported
    first, which lets a scenario file select a rule by name alone.

    Raises:
        KeyError: No rule called ``name`` is registered and no such module exists.
    """
    if name not in _REGISTRY and name.isidentifier():
        module = f"{__name__}.{name}"
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as exc:
            # Only swallow "the rule module itself is missing"; a missing dependency of an
            # existing rule (for example pyrothermel) must surface to the user.
            if exc.name != module:
                raise
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
