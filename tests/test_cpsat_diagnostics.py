"""Tests for CPSAT diagnostics surfaced via LXSolution."""

import pytest

from lumix import LXConstraint, LXLinearExpression, LXModel, LXOptimizer, LXVariable
from lumix.solvers.capabilities import CPSAT_CAPABILITIES, LXSolverFeature


def _trivial_assignment_model():
    """4 binaries + cover constraint + objective. Tiny but exercises CPSAT."""
    model = LXModel("diag_test")
    costs = {("W1", "T1"): 18, ("W1", "T2"): 25, ("W2", "T1"): 30, ("W2", "T2"): 13}
    assign = (
        LXVariable[tuple, int]("assign")
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
                    assign, coeff=lambda wt, _t=t: 1.0 if wt[1] == _t else 0.0
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


def test_capability_flags_present():
    """CPSAT_CAPABILITIES declares scheduling primitives + solution hint."""
    feats = CPSAT_CAPABILITIES.features
    assert LXSolverFeature.INTERVAL_VARIABLES in feats
    assert LXSolverFeature.NO_OVERLAP in feats
    assert LXSolverFeature.SOLUTION_HINT in feats


def test_solution_has_dual_bound_after_optimal():
    sol = LXOptimizer().use_solver("cpsat").solve(_trivial_assignment_model())
    assert sol.status == "optimal"
    assert sol.best_objective_bound is not None
    # For an optimal MIP, dual bound equals primal objective up to tolerance.
    assert sol.best_objective_bound == pytest.approx(sol.objective_value, abs=1.0)


def test_solution_has_deterministic_time():
    """deterministic_time is a CPSAT-only field; must be populated and positive."""
    sol = LXOptimizer().use_solver("cpsat").solve(_trivial_assignment_model())
    assert sol.deterministic_time is not None
    assert sol.deterministic_time >= 0


def test_solution_has_conflicts_field():
    """conflicts may be 0 on a trivial problem but the field exists."""
    sol = LXOptimizer().use_solver("cpsat").solve(_trivial_assignment_model())
    assert sol.conflicts is not None
    assert sol.conflicts >= 0


def test_solution_intervals_field_default_empty():
    sol = LXOptimizer().use_solver("cpsat").solve(_trivial_assignment_model())
    assert isinstance(sol.intervals, dict)
    # No interval variables in this model → empty dict.
    assert sol.intervals == {}
