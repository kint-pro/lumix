"""Tests for LXModel.set_solution_hint() and CPSAT AddHint integration."""

import pytest

from lumix import LXConstraint, LXLinearExpression, LXModel, LXOptimizer, LXVariable


def _model():
    m = LXModel("hint_test")
    costs = {("W1", "T1"): 18, ("W1", "T2"): 25, ("W2", "T1"): 30, ("W2", "T2"): 13}
    assign = (
        LXVariable[tuple, int]("assign")
        .binary()
        .indexed_by(lambda x: x)
        .from_data(list(costs.keys()))
    )
    m.add_variable(assign)
    for t in ["T1", "T2"]:
        m.add_constraint(
            LXConstraint(f"cover_{t}")
            .expression(
                LXLinearExpression().add_term(
                    assign, coeff=lambda wt, _t=t: 1.0 if wt[1] == _t else 0.0
                )
            )
            .ge()
            .rhs(1)
        )
    m.minimize(
        LXLinearExpression().add_term(assign, coeff=lambda wt: float(costs[wt]))
    )
    return m


def test_default_no_hint():
    assert _model()._solution_hint is None


def test_set_hint_via_dict():
    m = _model().set_solution_hint({"assign": {("W1", "T1"): 1, ("W2", "T2"): 1}})
    assert m._solution_hint == {"assign": {("W1", "T1"): 1, ("W2", "T2"): 1}}


def test_set_hint_via_solution():
    m = _model()
    sol = LXOptimizer().use_solver("cpsat").solve(m)
    m.set_solution_hint(sol)
    assert m._solution_hint is not None
    assert "assign" in m._solution_hint


def test_set_hint_none_clears():
    m = _model().set_solution_hint({"assign": {("W1", "T1"): 1}})
    m.set_solution_hint(None)
    assert m._solution_hint is None


def test_set_hint_rejects_invalid_type():
    with pytest.raises(TypeError, match="must be LXSolution or dict"):
        _model().set_solution_hint("not a hint")


def test_warmstart_solve_succeeds():
    """Solving with a hint must not crash and must reach optimal."""
    m = _model()
    sol1 = LXOptimizer().use_solver("cpsat").solve(m)
    m.set_solution_hint(sol1)
    sol2 = LXOptimizer().use_solver("cpsat").solve(m)
    assert sol2.status == "optimal"
    assert sol2.objective_value == pytest.approx(sol1.objective_value, abs=1.0)


def test_warmstart_with_partial_hint_ignored_safely():
    """Hint missing variables or with unknown keys must not crash."""
    m = _model()
    m.set_solution_hint({"unknown_var": 1, "assign": {("W1", "T1"): 999}})
    sol = LXOptimizer().use_solver("cpsat").solve(m)
    assert sol.status == "optimal"


def test_warmstart_float_hint_truncates_safely():
    """Float hint values must be coerced to int without crashing.

    Regression for integer-truncation bug paths: 0.7 → int(0.7)=0 is the
    expected coercion, but the SOLVE must still terminate cleanly.
    """
    m = _model()
    m.set_solution_hint({"assign": {("W1", "T1"): 0.7, ("W2", "T2"): 0.3}})
    sol = LXOptimizer().use_solver("cpsat").solve(m)
    assert sol.status == "optimal"


def test_empty_hint_dict_no_op():
    """An empty hint dict must be a benign no-op."""
    m = _model().set_solution_hint({})
    sol = LXOptimizer().use_solver("cpsat").solve(m)
    assert sol.status == "optimal"


def test_hint_ignored_by_ortools_solver():
    """Non-CPSAT solver (ortools MILP) ignores the hint silently."""
    m = _model().set_solution_hint({"assign": {("W1", "T1"): 1, ("W2", "T2"): 1}})
    sol = LXOptimizer().use_solver("ortools").solve(m)
    assert sol.status == "optimal"
    assert sol.objective_value == pytest.approx(31.0, abs=1.0)


def test_warmstart_actually_applied_via_logger(caplog):
    """Confirm hint reaches the solver by checking the diagnostic log line.

    Without this, an AddHint() that is silently dropped would still let the
    happy-path solve test (test_warmstart_solve_succeeds) pass — the optimum
    is reachable cold-start. This catches a no-op hint mutation.
    """
    import logging
    m = _model()
    sol1 = LXOptimizer().use_solver("cpsat").solve(m)
    m.set_solution_hint(sol1)
    with caplog.at_level(logging.INFO, logger="lumix.optimizer"):
        LXOptimizer().use_solver("cpsat").solve(m)
    hint_logs = [r for r in caplog.records if "Solution hint:" in r.message]
    assert hint_logs, "expected a 'Solution hint: N applied' log line"
    # 4 binary assign vars in the model — sol1 has values for all of them.
    assert "4 applied" in hint_logs[-1].message


def test_diagnostics_on_infeasible_no_crash():
    """Infeasible model must populate solution fields safely (None or 0), not crash."""
    m = LXModel("infeasible")
    x = (
        LXVariable[tuple, int]("x")
        .binary()
        .indexed_by(lambda x: x)
        .from_data([("a",)])
    )
    m.add_variable(x)
    # Force infeasibility: x >= 1 AND x <= 0.
    m.add_constraint(
        LXConstraint("ge1")
        .expression(LXLinearExpression().add_term(x, coeff=lambda _: 1.0))
        .ge()
        .rhs(1)
    )
    m.add_constraint(
        LXConstraint("le0")
        .expression(LXLinearExpression().add_term(x, coeff=lambda _: 1.0))
        .le()
        .rhs(0)
    )
    m.minimize(LXLinearExpression().add_term(x, coeff=lambda _: 1.0))
    sol = LXOptimizer().use_solver("cpsat").solve(m)
    assert sol.status in {"infeasible", "unknown", "model_invalid"}
    # Must NOT crash on best_objective_bound/conflicts/deterministic_time access.
    assert sol.best_objective_bound is None or isinstance(sol.best_objective_bound, (int, float))
    assert sol.conflicts is None or isinstance(sol.conflicts, int)
    assert sol.deterministic_time is None or isinstance(sol.deterministic_time, float)
