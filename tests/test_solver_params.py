"""Tests for LXOptimizer.use_solver(**kwargs) param storage and merging in solve()."""

from dataclasses import dataclass

import pytest

from lumix import (
    LXConstraint,
    LXLinearExpression,
    LXModel,
    LXOptimizer,
    LXVariable,
)


@dataclass
class _X:
    id: str


def _trivial_lp() -> LXModel:
    m = LXModel("trivial")
    x = (
        LXVariable[_X, float]("x")
        .continuous()
        .bounds(lower=0, upper=10)
        .indexed_by(lambda v: v.id)
        .from_data([_X(id="x")])
    )
    m.add_variable(x)
    m.minimize(LXLinearExpression().add_term(x, coeff=lambda v: 1.0))
    m.add_constraint(
        LXConstraint("lb")
        .expression(LXLinearExpression().add_term(x, coeff=lambda v: 1.0))
        .ge()
        .rhs(5.0)
    )
    return m


def test_solver_params_default_empty():
    opt = LXOptimizer()
    assert opt._solver_params == {}


def test_use_solver_stores_kwargs():
    opt = LXOptimizer().use_solver("ortools", time_limit=30.0)
    assert opt._solver_params == {"time_limit": 30.0}


def test_stored_params_applied_in_solve():
    sol = (
        LXOptimizer()
        .use_solver("ortools", time_limit=30.0)
        .solve(_trivial_lp())
    )
    assert sol.is_optimal()
    assert sol.objective_value == 5.0


def test_overlap_between_use_solver_and_solve_raises():
    opt = LXOptimizer().use_solver("ortools", time_limit=30.0)
    with pytest.raises(ValueError, match="set in both"):
        opt.solve(_trivial_lp(), time_limit=60.0)


def test_stored_and_call_site_params_merged(monkeypatch):
    captured: dict = {}

    from lumix.solvers import ortools_solver

    original_solve = ortools_solver.LXORToolsSolver.solve

    def spy_solve(self, model, **kwargs):
        captured.update(kwargs)
        return original_solve(self, model, **kwargs)

    monkeypatch.setattr(ortools_solver.LXORToolsSolver, "solve", spy_solve)

    LXOptimizer().use_solver("ortools", time_limit=30.0).solve(
        _trivial_lp(), gap_tolerance=0.01
    )

    assert captured["time_limit"] == 30.0
    assert captured["gap_tolerance"] == 0.01
