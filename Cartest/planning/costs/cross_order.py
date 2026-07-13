"""Cross-order coupling Lyapunov costs from cost_transform templates.

Each template (conservative/standard/active/aggressive/emergency) is registered
as a separate cost factory.  The template's ACC_MAX/JERK_MAX are passed as
``safety_overrides`` in the constraint config so that ``make_constraints`` can
merge them into the scenario's safety dict.
"""

from __future__ import annotations

from copy import deepcopy

from Cartest.planning.cost_transform import (
    make_objective_cross_order,
    template_coupling,
)
from Cartest.planning.costs.default_lyapunov import DEFAULT_CONSTRAINTS
from Cartest.planning.costs.registry import register_cost


_BASE_SPECS = deepcopy(DEFAULT_CONSTRAINTS["specs"])


def _make_cross_order_factory(template_name: str):
    """Build a cost factory for a named cross-order coupling template."""
    C_ba, C_ab, omega_z, omega_w, acc_max, jerk_max = template_coupling(template_name)

    def make_cost(gen, **_params):
        return make_objective_cross_order(
            gen, omega_z=omega_z, omega_w=omega_w, C_ba=C_ba, C_ab=C_ab)

    def make_constraints_config(**_params):
        cfg = deepcopy(DEFAULT_CONSTRAINTS)
        cfg["safety_overrides"] = {"acc_max": acc_max, "jerk_max": jerk_max}
        return cfg

    make_cost.constraint_config_factory = make_constraints_config
    make_cost.__name__ = f"make_cross_order_{template_name}"
    return make_cost


for _name in ("conservative", "standard", "active", "aggressive", "emergency"):
    register_cost(f"cross_order_{_name}", _make_cross_order_factory(_name))
