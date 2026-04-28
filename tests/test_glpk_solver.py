"""
Tests for GLPK solver dual-bound exposure.

Regression: GLPK previously left `LXSolution.best_objective_bound = None`
even when `glp_get_obj_val` held the LP-relaxation root value (e.g., MIP
solver hit time limit before finding any incumbent).
"""

import pytest

from lumix import LXConstraint, LXLinearExpression, LXModel, LXOptimizer, LXVariable


def _build_assignment_model():
    """Tiny binary assignment: 2 workers x 2 tasks, minimize cost."""
    model = LXModel("glpk_assignment")
    costs = {("W1", "T1"): 18, ("W1", "T2"): 25, ("W2", "T1"): 30, ("W2", "T2"): 13}

    assign = (
        LXVariable("assign")
        .binary()
        .indexed_by(lambda x: x)
        .from_data(list(costs.keys()))
    )
    model.add_variable(assign)

    for t in ["T1", "T2"]:
        model.add_constraint(
            LXConstraint(f"cover_{t}")
            .expression(
                LXLinearExpression().add_term(
                    assign,
                    coeff=lambda wt, _t=t: 1.0 if wt[1] == _t else 0.0,
                )
            )
            .ge()
            .rhs(1)
        )

    model.minimize(
        LXLinearExpression().add_term(assign, coeff=lambda wt: float(costs[wt]))
    )
    return model


def _build_hard_knapsack(n=30):
    """N binary vars, equality `sum = n/2 + 0.5` -> LP feasible, MIP infeasible.

    LP relaxation root is exactly n/2 + 0.5 (each x_i set to 0.5 + 0.5/n
    or similar). The fractional RHS forces zero integer feasible region,
    so under any nontrivial time budget GLPK returns GLP_UNDEF.
    """
    model = LXModel("glpk_hard")
    items = list(range(n))
    target = n / 2 + 0.5

    x = (
        LXVariable("x")
        .binary()
        .indexed_by(lambda i: i)
        .from_data(items)
    )
    model.add_variable(x)

    model.add_constraint(
        LXConstraint("budget")
        .expression(LXLinearExpression().add_term(x, coeff=lambda _i: 1.0))
        .eq()
        .rhs(target)
    )

    model.minimize(LXLinearExpression().add_term(x, coeff=lambda _i: 1.0))
    return model, target


def test_glpk_optimal_populates_bound():
    """GLPK MIP solved to proven optimum: bound equals objective."""
    model = _build_assignment_model()
    sol = LXOptimizer().use_solver("glpk").solve(model)
    assert sol.status == "optimal"
    assert sol.objective_value == pytest.approx(31.0, abs=1.0)
    assert sol.best_objective_bound is not None
    assert sol.best_objective_bound == pytest.approx(sol.objective_value, abs=1e-6)


def test_glpk_undefined_exposes_lp_root():
    """When MIP times out / is undefined but LP relaxation solved, bound is the LP root."""
    model, target = _build_hard_knapsack(n=30)
    sol = LXOptimizer().use_solver("glpk").solve(model, time_limit=0.001)
    assert sol.status in ("undefined", "infeasible")
    assert sol.best_objective_bound is not None
    assert sol.best_objective_bound == pytest.approx(target, abs=1e-6)


def test_glpk_infeasible_no_crash():
    """Trivially infeasible model still parses cleanly; no bound populated."""
    model = LXModel("glpk_infeas")
    items = [0]
    x = LXVariable("x").binary().indexed_by(lambda i: i).from_data(items)
    model.add_variable(x)
    model.add_constraint(
        LXConstraint("force_two")
        .expression(LXLinearExpression().add_term(x, coeff=lambda _i: 1.0))
        .ge()
        .rhs(2)
    )
    model.minimize(LXLinearExpression().add_term(x, coeff=lambda _i: 1.0))

    sol = LXOptimizer().use_solver("glpk").solve(model)
    assert sol.status in ("infeasible", "undefined")
    assert sol.best_objective_bound is None


def test_glpk_lp_optimal_populates_bound():
    """Pure LP solved to optimum: bound equals objective (consistency)."""
    model = LXModel("glpk_lp")
    items = [("a",), ("b",)]
    x = (
        LXVariable("x")
        .continuous()
        .bounds(lower=0.0, upper=1.0)
        .indexed_by(lambda t: t)
        .from_data(items)
    )
    model.add_variable(x)
    model.add_constraint(
        LXConstraint("sum_one")
        .expression(LXLinearExpression().add_term(x, coeff=lambda _t: 1.0))
        .ge()
        .rhs(1)
    )
    model.minimize(LXLinearExpression().add_term(x, coeff=lambda _t: 1.0))

    sol = LXOptimizer().use_solver("glpk").solve(model)
    assert sol.status == "optimal"
    assert sol.best_objective_bound == pytest.approx(sol.objective_value, abs=1e-6)
