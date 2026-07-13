"""Registry for scenario-selected objective factories.

Scenarios bind to an objective through ``scenario["cost"]``:

    {"name": "default_lyapunov", "params": {"omega_s": 1.0}}

The solver still owns final cost assembly: it receives the selected objective
and wraps it with Constran when constraints are passed to ``build_solver``.
"""

from __future__ import annotations

from Cartest.planning.costs.default_lyapunov import make_cost as make_default_lyapunov
from Cartest.planning.costs.lane_borrow_overtake import make_cost as make_lane_borrow_overtake


COST_FACTORIES = {}


def register_cost(name=None, factory=None, *, replace=False):
    """Register a cost factory with signature ``factory(gen, **params)``.

    Can be used directly:

        register_cost("my_cost", make_my_cost)

    or as a decorator:

        @register_cost("my_cost")
        def make_my_cost(gen, **params): ...
    """
    if callable(name) and factory is None:
        factory = name
        name = factory.__name__

    if factory is None:
        def decorator(fn):
            return register_cost(name, fn, replace=replace)
        return decorator

    if not name:
        raise ValueError("cost name must be a non-empty string")
    if not callable(factory):
        raise TypeError(f"cost factory for {name!r} must be callable")
    if not replace and name in COST_FACTORIES:
        raise ValueError(f"cost {name!r} is already registered")

    COST_FACTORIES[name] = factory
    return factory


def get_cost_factory(name):
    """Return a registered cost factory by name."""
    try:
        return COST_FACTORIES[name]
    except KeyError as exc:
        available = ", ".join(sorted(COST_FACTORIES))
        raise ValueError(f"Unknown cost_name={name!r}. Available costs: {available}") from exc


def available_costs():
    """Return registered cost names."""
    return tuple(sorted(COST_FACTORIES))


def get_cost_spec(scenario):
    """Return ``(name, params, factory)`` from a scenario."""
    cost = scenario.get("cost")
    if callable(cost):
        return getattr(cost, "__name__", "<callable>"), {}, cost
    if isinstance(cost, str):
        return cost, {}, None
    if cost is not None:
        if not isinstance(cost, dict):
            raise TypeError(f"scenario cost must be str, dict, or callable, got {type(cost)!r}")
        name = cost.get("name", scenario.get("cost_name", "default_lyapunov"))
        params = dict(cost.get("params", scenario.get("cost_params", {})))
        factory = cost.get("factory")
        if factory is not None and not callable(factory):
            raise TypeError(f"scenario cost factory for {name!r} must be callable")
        return name, params, factory

    return (
        scenario.get("cost_name", "default_lyapunov"),
        dict(scenario.get("cost_params", {})),
        None,
    )


def make_objective_from_scenario(gen, scenario):
    """Build the objective selected by a scenario dict."""
    cost_name, cost_params, factory = get_cost_spec(scenario)
    factory = factory or get_cost_factory(cost_name)
    return factory(gen, **cost_params)


def make_constraint_config_from_scenario(scenario):
    """Build the constraint/Constran config paired with a scenario's cost."""
    cost_name, cost_params, factory = get_cost_spec(scenario)
    factory = factory or get_cost_factory(cost_name)
    config_factory = getattr(factory, "constraint_config_factory", None)
    if config_factory is None:
        raise ValueError(
            f"Cost {cost_name!r} does not provide constraint_config_factory. "
            "Attach one to the registered cost factory."
        )
    return config_factory(**cost_params)


register_cost("default_lyapunov", make_default_lyapunov)
register_cost("lane_borrow_overtake", make_lane_borrow_overtake)
