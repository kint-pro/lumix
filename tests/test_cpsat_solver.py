"""
Tests for CP-SAT solver constraint naming fix.

Regression test for: ct.Proto().name corrupts CP-SAT protobuf,
causing 'Check failed: LoadConstraint' crash.
Fix: use ct.WithName() instead.
"""

import pytest

from lumix import LXConstraint, LXLinearExpression, LXModel, LXOptimizer, LXVariable


def _build_assignment_model():
    """Minimal binary assignment: 2 workers, 2 tasks, minimize cost."""
    model = LXModel("cpsat_test")
    costs = {("W1", "T1"): 18, ("W1", "T2"): 25, ("W2", "T1"): 30, ("W2", "T2"): 13}

    assign = (
        LXVariable[str, int]("assign")
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
        LXLinearExpression().add_term(
            assign, coeff=lambda wt: float(costs[wt])
        )
    )
    return model


def test_cpsat_solves_without_crash():
    """CP-SAT must return optimal solution without segfault."""
    model = _build_assignment_model()
    sol = LXOptimizer().use_solver("cpsat").solve(model)
    assert sol.status == "optimal"
    assert sol.objective_value == pytest.approx(31.0, abs=1.0)


def test_cpsat_matches_ortools():
    """CP-SAT and OR-Tools MIP must agree on objective."""
    model = _build_assignment_model()
    sol_lp = LXOptimizer().use_solver("ortools").solve(model)
    sol_cp = LXOptimizer().use_solver("cpsat").solve(model)
    assert sol_cp.objective_value == pytest.approx(sol_lp.objective_value, rel=0.01)


def test_cpsat_float_coefficients():
    """CP-SAT must handle float objective coefficients via rational conversion."""
    model = LXModel("float_test")
    costs = {("W1", "T1"): 18.11, ("W1", "T2"): 25.50, ("W2", "T1"): 30.00, ("W2", "T2"): 12.75}

    assign = (
        LXVariable[str, int]("assign")
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
        LXLinearExpression().add_term(assign, coeff=lambda wt: costs[wt])
    )

    sol = LXOptimizer().use_solver("cpsat").solve(model)
    assert sol.status == "optimal"
    assert sol.objective_value == pytest.approx(30.86, rel=0.01)
