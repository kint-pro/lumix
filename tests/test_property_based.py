"""Property-based and metamorphic tests for CPSAT features.

Catches bugs that example-based tests miss by testing INVARIANTS over
random inputs rather than specific cases.

Categories:
- Property: invariants that must hold for ALL inputs satisfying preconditions.
- Metamorphic: relations between outputs of related inputs (e.g. scaling).
- Differential: same problem in two formulations agrees.
"""

import pytest
from hypothesis import given, settings, strategies as st

from lumix import LXConstraint, LXLinearExpression, LXModel, LXOptimizer, LXVariable


def _binary_assignment(costs):
    """Reusable mini-MIP: binary assignment of workers to tasks with cost minimization."""
    model = LXModel("prop_test")
    assign = (
        LXVariable[tuple, int]("assign")
        .binary()
        .indexed_by(lambda x: x)
        .from_data(list(costs.keys()))
    )
    model.add_variable(assign)
    tasks = sorted({k[1] for k in costs})
    for t in tasks:
        model.add_constraint(
            LXConstraint(f"cover_{t}")
            .expression(
                LXLinearExpression().add_term(
                    assign, coeff=lambda wt, _t=t: 1.0 if wt[1] == _t else 0.0
                )
            )
            .ge()
            .rhs(1)
        )
    model.minimize(
        LXLinearExpression().add_term(assign, coeff=lambda wt: float(costs[wt]))
    )
    return model


# ============================== INVARIANT (Property) ==============================

@given(
    cost_a=st.integers(min_value=1, max_value=100),
    cost_b=st.integers(min_value=1, max_value=100),
    cost_c=st.integers(min_value=1, max_value=100),
    cost_d=st.integers(min_value=1, max_value=100),
)
@settings(max_examples=20, deadline=None)
def test_dual_bound_le_primal_for_minimization(cost_a, cost_b, cost_c, cost_d):
    """For a min problem solved to optimality, best_objective_bound ≤ objective."""
    model = _binary_assignment(
        {("W1", "T1"): cost_a, ("W1", "T2"): cost_b,
         ("W2", "T1"): cost_c, ("W2", "T2"): cost_d}
    )
    sol = LXOptimizer().use_solver("cpsat").solve(model)
    assert sol.status == "optimal"
    assert sol.best_objective_bound is not None
    # For optimal solutions, the dual bound must equal the primal up to numerical
    # tolerance — and it must NEVER exceed it (would be unsound).
    assert sol.best_objective_bound <= sol.objective_value + 1e-6


# ============================== METAMORPHIC ==============================

@given(
    base_costs=st.fixed_dictionaries({
        ("W1", "T1"): st.integers(min_value=1, max_value=100),
        ("W1", "T2"): st.integers(min_value=1, max_value=100),
        ("W2", "T1"): st.integers(min_value=1, max_value=100),
        ("W2", "T2"): st.integers(min_value=1, max_value=100),
    }),
    scale=st.integers(min_value=2, max_value=10),
)
@settings(max_examples=10, deadline=None)
def test_scaling_costs_scales_objective(base_costs, scale):
    """Metamorphic: multiplying all costs by k must multiply optimum by k."""
    sol_base = LXOptimizer().use_solver("cpsat").solve(_binary_assignment(base_costs))
    scaled_costs = {k: v * scale for k, v in base_costs.items()}
    sol_scaled = LXOptimizer().use_solver("cpsat").solve(_binary_assignment(scaled_costs))
    assert sol_base.status == "optimal"
    assert sol_scaled.status == "optimal"
    assert sol_scaled.objective_value == pytest.approx(
        sol_base.objective_value * scale, abs=1e-6
    ), f"scale={scale}, base obj={sol_base.objective_value}, scaled obj={sol_scaled.objective_value}"


# ============================== DIFFERENTIAL ==============================

@given(
    cost_a=st.integers(min_value=1, max_value=50),
    cost_b=st.integers(min_value=1, max_value=50),
    cost_c=st.integers(min_value=1, max_value=50),
    cost_d=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=15, deadline=None)
def test_cpsat_and_ortools_agree_on_optimum(cost_a, cost_b, cost_c, cost_d):
    """Two solvers on the same model must produce the same optimum."""
    costs = {("W1", "T1"): cost_a, ("W1", "T2"): cost_b,
             ("W2", "T1"): cost_c, ("W2", "T2"): cost_d}
    sol_cp = LXOptimizer().use_solver("cpsat").solve(_binary_assignment(costs))
    sol_or = LXOptimizer().use_solver("ortools").solve(_binary_assignment(costs))
    assert sol_cp.status == "optimal"
    assert sol_or.status == "optimal"
    assert sol_cp.objective_value == pytest.approx(sol_or.objective_value, abs=1e-6)


# ============================== ROUND-TRIP ==============================

@given(
    cost_a=st.integers(min_value=1, max_value=50),
    cost_b=st.integers(min_value=1, max_value=50),
    cost_c=st.integers(min_value=1, max_value=50),
    cost_d=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=10, deadline=None)
def test_warmstart_optimum_stable(cost_a, cost_b, cost_c, cost_d):
    """Solve cold → set sol as hint → re-solve must reach same objective."""
    costs = {("W1", "T1"): cost_a, ("W1", "T2"): cost_b,
             ("W2", "T1"): cost_c, ("W2", "T2"): cost_d}
    model = _binary_assignment(costs)
    sol1 = LXOptimizer().use_solver("cpsat").solve(model)
    model.set_solution_hint(sol1)
    sol2 = LXOptimizer().use_solver("cpsat").solve(model)
    assert sol2.objective_value == pytest.approx(sol1.objective_value, abs=1e-6)


# ============================== PROPERTY: hint robustness ==============================

@given(
    extra_garbage=st.dictionaries(
        keys=st.text(min_size=1, max_size=8),
        values=st.integers(min_value=-1000, max_value=1000),
        min_size=0,
        max_size=5,
    ),
)
@settings(max_examples=10, deadline=None)
def test_hint_with_unknown_keys_never_crashes(extra_garbage):
    """Hint with unknown variable names must be silently ignored, not crash."""
    costs = {("W1", "T1"): 18, ("W1", "T2"): 25, ("W2", "T1"): 30, ("W2", "T2"): 13}
    model = _binary_assignment(costs)
    model.set_solution_hint(extra_garbage)
    sol = LXOptimizer().use_solver("cpsat").solve(model)
    assert sol.status == "optimal"
    assert sol.objective_value == pytest.approx(31.0, abs=1.0)
