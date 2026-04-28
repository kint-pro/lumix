"""Tests for LXIntervalVariable, LXNoOverlapConstraint, and CPSAT dispatch."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import pytest

from lumix import (
    LXConstraint,
    LXLinearExpression,
    LXModel,
    LXOptimizer,
    LXVariable,
)
from lumix.core.constraints import LXNoOverlapConstraint
from lumix.core.interval import LXIntervalVariable


@dataclass
class Op:
    id: str
    machine: str
    duration: int


# ==================== UNIT: builder API ====================


def test_interval_builder_returns_self():
    start = LXVariable[Op, int]("start").integer().bounds(0, 100)
    end = LXVariable[Op, int]("end").integer().bounds(0, 100)
    iv = LXIntervalVariable[Op]("op_iv")
    assert iv.start(start) is iv
    assert iv.end(end) is iv
    assert iv.duration_fixed(5) is iv
    assert iv.indexed_by(lambda o: o.id) is iv
    assert iv.from_data([Op("a", "m1", 5)]) is iv


def test_interval_duration_fixed_and_var_mutually_exclusive():
    start = LXVariable[Op, int]("start").integer().bounds(0, 100)
    end = LXVariable[Op, int]("end").integer().bounds(0, 100)
    dvar = LXVariable[Op, int]("dur").integer().bounds(1, 10)

    iv = LXIntervalVariable[Op]("op_iv").duration_fixed(5)
    with pytest.raises(ValueError):
        iv.duration_var(dvar)

    iv2 = LXIntervalVariable[Op]("op_iv2").duration_var(dvar)
    with pytest.raises(ValueError):
        iv2.duration_fixed(5)


def test_interval_fields_populated():
    start = LXVariable[Op, int]("start").integer().bounds(0, 100)
    end = LXVariable[Op, int]("end").integer().bounds(0, 100)
    ops = [Op("a", "m1", 5), Op("b", "m1", 3)]
    iv = (
        LXIntervalVariable[Op]("op_iv")
        .start(start)
        .end(end)
        .duration_fixed(5)
        .indexed_by(lambda o: o.id)
        .from_data(ops)
    )
    assert iv.name == "op_iv"
    assert iv.start_var is start
    assert iv.end_var is end
    assert iv.duration == 5
    assert iv.duration_var_ref is None
    assert iv.index_func is not None
    assert iv._data == ops


def test_interval_deepcopy_independent():
    start = LXVariable[Op, int]("start").integer().bounds(0, 100)
    end = LXVariable[Op, int]("end").integer().bounds(0, 100)
    ops = [Op("a", "m1", 5)]
    iv = (
        LXIntervalVariable[Op]("op_iv")
        .start(start)
        .end(end)
        .duration_fixed(5)
        .indexed_by(lambda o: o.id)
        .from_data(ops)
    )
    iv2 = deepcopy(iv)
    assert iv2 is not iv
    assert iv2.name == iv.name
    assert iv2.duration == 5
    assert iv2.start_var is not iv.start_var
    assert iv2._data is not iv._data
    assert iv2._data[0].id == "a"


# ==================== UNIT: LXNoOverlapConstraint ====================


def test_no_overlap_constructor_and_is_goal():
    iv = LXIntervalVariable[Op]("op_iv")
    cons = LXNoOverlapConstraint("no_overlap_m1", [iv])
    assert cons.name == "no_overlap_m1"
    assert cons.intervals == [iv]
    assert cons.is_goal() is False


def test_no_overlap_deepcopy():
    iv = LXIntervalVariable[Op]("op_iv")
    cons = LXNoOverlapConstraint("c", [iv])
    cons2 = deepcopy(cons)
    assert cons2 is not cons
    assert cons2.name == "c"
    assert len(cons2.intervals) == 1
    assert cons2.intervals[0] is not iv


# ==================== UNIT: model wiring ====================


def test_model_add_interval_variable_returns_self():
    model: LXModel = LXModel("m")
    iv = LXIntervalVariable("op_iv")
    assert model.add_interval_variable(iv) is model
    assert model.intervals == [iv]


def test_model_add_no_overlap_polymorphic_dispatch():
    model: LXModel = LXModel("m")
    iv = LXIntervalVariable("op_iv")
    cons = LXNoOverlapConstraint("c", [iv])
    model.add_constraint(cons)
    assert model.scheduling_constraints == [cons]
    assert model.constraints == []


def test_model_add_constraint_keeps_regular_path():
    model: LXModel = LXModel("m")
    x = LXVariable[Op, int]("x").integer().bounds(0, 10)
    regular = (
        LXConstraint("cap")
        .expression(LXLinearExpression().add_term(x, coeff=lambda _o: 1.0))
        .le()
        .rhs(5)
    )
    model.add_constraint(regular)
    assert model.constraints == [regular]
    assert model.scheduling_constraints == []


def test_model_deepcopy_includes_new_fields():
    model: LXModel = LXModel("m")
    iv = LXIntervalVariable("op_iv")
    cons = LXNoOverlapConstraint("c", [iv])
    model.add_interval_variable(iv).add_constraint(cons)
    copy = deepcopy(model)
    assert len(copy.intervals) == 1
    assert len(copy.scheduling_constraints) == 1
    assert copy.intervals[0] is not iv
    assert copy.scheduling_constraints[0] is not cons


# ==================== INTEGRATION: tiny JSP via interval+no-overlap ====================


def _build_jsp_interval(ops, machines, horizon: int) -> LXModel:
    """JSP using LXIntervalVariable + LXNoOverlapConstraint, minimize makespan."""
    model: LXModel = LXModel("jsp_interval")

    # Op-keyed duration dict for solver-side per-instance bound use.
    op_dur = {o.id: o.duration for o in ops}

    start = (
        LXVariable[Op, int]("start")
        .integer()
        .bounds(0, horizon)
        .indexed_by(lambda o: o.id)
        .from_data(ops)
    )
    end = (
        LXVariable[Op, int]("end")
        .integer()
        .bounds(0, horizon)
        .indexed_by(lambda o: o.id)
        .from_data(ops)
    )
    # Per-op duration variable: bounds locked equal via constraint (LXVariable bounds are uniform).
    duration = (
        LXVariable[Op, int]("duration")
        .integer()
        .bounds(0, max(op_dur.values()))
        .indexed_by(lambda o: o.id)
        .from_data(ops)
    )
    makespan = (
        LXVariable[object, int]("makespan").integer().bounds(0, horizon).from_data([None])
    )

    model.add_variables(start, end, duration, makespan)

    # Pin per-op duration via two inequalities: duration[o] == op.duration.
    for o in ops:
        model.add_constraint(
            LXConstraint(f"dur_eq_{o.id}")
            .expression(
                LXLinearExpression().add_term(
                    duration, coeff=lambda x, _id=o.id: 1.0 if x.id == _id else 0.0
                )
            )
            .eq()
            .rhs(o.duration)
        )

    # makespan >= end[o]
    for o in ops:
        model.add_constraint(
            LXConstraint(f"mk_{o.id}")
            .expression(
                LXLinearExpression()
                .add_term(makespan, coeff=lambda _x: 1.0)
                .add_term(end, coeff=lambda x, _id=o.id: -1.0 if x.id == _id else 0.0)
            )
            .ge()
            .rhs(0)
        )

    # Interval var: CP-SAT IntervalVar enforces start + size == end internally.
    iv = (
        LXIntervalVariable[Op]("op_iv")
        .start(start)
        .end(end)
        .duration_var(duration)
        .indexed_by(lambda o: o.id)
        .from_data(ops)
    )
    model.add_interval_variable(iv)

    # No-overlap per machine — for each machine, only ops on that machine.
    for m in machines:
        machine_ops = [o for o in ops if o.machine == m]
        if not machine_ops:
            continue
        iv_m = (
            LXIntervalVariable[Op](f"op_iv_{m}")
            .start(start)
            .end(end)
            .duration_var(duration)
            .indexed_by(lambda o: o.id)
            .from_data(machine_ops)
        )
        model.add_interval_variable(iv_m)
        model.add_constraint(LXNoOverlapConstraint(f"no_overlap_{m}", [iv_m]))

    model.minimize(LXLinearExpression().add_term(makespan, coeff=lambda _x: 1.0))
    return model


def _build_jsp_bigm(ops, horizon: int) -> LXModel:
    """Same JSP via manual disjunction Big-M."""
    model: LXModel = LXModel("jsp_bigm")
    start = (
        LXVariable[Op, int]("start")
        .integer()
        .bounds(0, horizon)
        .indexed_by(lambda o: o.id)
        .from_data(ops)
    )
    makespan = (
        LXVariable[object, int]("makespan").integer().bounds(0, horizon).from_data([None])
    )
    model.add_variables(start, makespan)

    # makespan >= start[o] + duration[o]
    for o in ops:
        model.add_constraint(
            LXConstraint(f"mk_{o.id}")
            .expression(
                LXLinearExpression()
                .add_term(makespan, coeff=lambda _x: 1.0)
                .add_term(start, coeff=lambda x, _id=o.id: -1.0 if x.id == _id else 0.0)
            )
            .ge()
            .rhs(o.duration)
        )

    # Pairwise disjunction on same machine: y[i,j] in {0,1}
    same_machine_pairs = []
    for i, oi in enumerate(ops):
        for oj in ops[i + 1 :]:
            if oi.machine == oj.machine:
                same_machine_pairs.append((oi, oj))

    if same_machine_pairs:
        @dataclass(frozen=True)
        class Pair:
            i_id: str
            j_id: str
            i_dur: int
            j_dur: int

        pair_data = [Pair(oi.id, oj.id, oi.duration, oj.duration) for oi, oj in same_machine_pairs]
        y = (
            LXVariable[Pair, int]("y")
            .binary()
            .indexed_by(lambda p: (p.i_id, p.j_id))
            .from_data(pair_data)
        )
        model.add_variable(y)

        for p in pair_data:
            i_id, j_id = p.i_id, p.j_id

            # start[i] + dur[i] <= start[j] + M*(1 - y) ->
            # start[i] - start[j] + M*y <= M - dur[i]
            model.add_constraint(
                LXConstraint(f"disj_a_{i_id}_{j_id}")
                .expression(
                    LXLinearExpression()
                    .add_term(start, coeff=lambda x, _id=i_id: 1.0 if x.id == _id else 0.0)
                    .add_term(start, coeff=lambda x, _id=j_id: -1.0 if x.id == _id else 0.0)
                    .add_term(y, coeff=lambda pp, _ii=i_id, _jj=j_id: float(horizon) if (pp.i_id, pp.j_id) == (_ii, _jj) else 0.0)
                )
                .le()
                .rhs(horizon - p.i_dur)
            )
            # start[j] + dur[j] <= start[i] + M*y ->
            # start[j] - start[i] - M*y <= -dur[j]
            model.add_constraint(
                LXConstraint(f"disj_b_{i_id}_{j_id}")
                .expression(
                    LXLinearExpression()
                    .add_term(start, coeff=lambda x, _id=j_id: 1.0 if x.id == _id else 0.0)
                    .add_term(start, coeff=lambda x, _id=i_id: -1.0 if x.id == _id else 0.0)
                    .add_term(y, coeff=lambda pp, _ii=i_id, _jj=j_id: -float(horizon) if (pp.i_id, pp.j_id) == (_ii, _jj) else 0.0)
                )
                .le()
                .rhs(-p.j_dur)
            )

    model.minimize(LXLinearExpression().add_term(makespan, coeff=lambda _x: 1.0))
    return model


def test_differential_2x2_jsp_interval_matches_bigm():
    """2 jobs x 2 machines: interval+no-overlap matches Big-M disjunction."""
    ops = [
        Op("J1_M1", "M1", 3),
        Op("J1_M2", "M2", 2),
        Op("J2_M1", "M1", 2),
        Op("J2_M2", "M2", 4),
    ]
    horizon = 20

    iv_model = _build_jsp_interval(ops, machines=["M1", "M2"], horizon=horizon)
    bigm_model = _build_jsp_bigm(ops, horizon=horizon)

    sol_iv = LXOptimizer().use_solver("cpsat").solve(iv_model)
    sol_bm = LXOptimizer().use_solver("cpsat").solve(bigm_model)

    assert sol_iv.status == "optimal"
    assert sol_bm.status == "optimal"
    assert sol_iv.objective_value == pytest.approx(sol_bm.objective_value, abs=1e-6)


def test_intervals_populated_in_solution():
    """Solution must expose intervals[name][key] = {start, end, duration}."""
    ops = [Op("o1", "M1", 3), Op("o2", "M1", 2)]
    horizon = 20
    model = _build_jsp_interval(ops, machines=["M1"], horizon=horizon)
    sol = LXOptimizer().use_solver("cpsat").solve(model)
    assert sol.status == "optimal"
    assert "op_iv" in sol.intervals
    iv_values = sol.intervals["op_iv"]
    assert set(iv_values.keys()) == {"o1", "o2"}
    for key in ("o1", "o2"):
        rec = iv_values[key]
        assert "start" in rec and "end" in rec and "duration" in rec
        assert rec["end"] - rec["start"] == rec["duration"]


def test_no_overlap_actually_enforced_in_schedule():
    """Mutation guard: confirm NoOverlap is REALLY applied to CP-SAT, not silently
    skipped.

    The differential test (interval vs Big-M) and the optimum-equals-sum test
    can both be passed by an implementation that secretly skips
    `_apply_no_overlap` — the optima would coincide for trivial reasons
    (no constraints → solver finds 0; or both broken in the same way).

    This test asserts the SCHEDULE in the solution has no time overlap on
    any machine — which can ONLY hold if NoOverlap was emitted and
    propagated. A no-op mutation of `_apply_no_overlap` would let intervals
    overlap and this test would fail.
    """
    ops = [Op("o1", "M1", 4), Op("o2", "M1", 3), Op("o3", "M1", 5)]
    horizon = 40
    model = _build_jsp_interval(ops, machines=["M1"], horizon=horizon)
    sol = LXOptimizer().use_solver("cpsat").solve(model)
    assert sol.status == "optimal"
    iv_values = sol.intervals["op_iv"]
    intervals = sorted(
        [(iv_values[op.id]["start"], iv_values[op.id]["end"]) for op in ops]
    )
    for i in range(len(intervals) - 1):
        # End of one must be ≤ start of next: no overlap on the shared machine.
        assert intervals[i][1] <= intervals[i + 1][0], (
            f"NoOverlap violated: {intervals[i]} overlaps {intervals[i+1]} "
            f"— _apply_no_overlap likely silently skipped"
        )


def test_5op_jsp_optimum_makespan():
    """5 ops on a single machine, sum of durations = optimum makespan."""
    ops = [
        Op("o1", "M1", 4),
        Op("o2", "M1", 3),
        Op("o3", "M1", 5),
        Op("o4", "M1", 2),
        Op("o5", "M1", 6),
    ]
    horizon = 50
    model = _build_jsp_interval(ops, machines=["M1"], horizon=horizon)
    sol = LXOptimizer().use_solver("cpsat").solve(model)
    assert sol.status == "optimal"
    assert sol.objective_value == pytest.approx(sum(o.duration for o in ops), abs=1e-6)


# ==================== Per-instance fixed durations ====================


def test_duration_per_instance_per_op_fixed_durations():
    """Per-instance fixed duration via lambda — each Op carries its own duration."""
    ops = [Op("o1", "M1", 4), Op("o2", "M1", 7), Op("o3", "M1", 2)]
    horizon = 30
    model = LXModel("per_instance_dur")
    starts = (
        LXVariable[Op, int]("start").integer().bounds(0, horizon)
        .indexed_by(lambda o: o.id).from_data(ops)
    )
    ends = (
        LXVariable[Op, int]("end").integer().bounds(0, horizon)
        .indexed_by(lambda o: o.id).from_data(ops)
    )
    model.add_variable(starts).add_variable(ends)
    interval = (
        LXIntervalVariable[Op]("op_iv")
        .start(starts).end(ends)
        .duration_per_instance(lambda o: o.duration)
        .indexed_by(lambda o: o.id).from_data(ops)
    )
    model.add_interval_variable(interval)
    model.add_constraint(LXNoOverlapConstraint("m1_no_ovr", [interval]))
    sol = LXOptimizer().use_solver("cpsat").solve(model)
    assert sol.status == "optimal"
    iv_values = sol.intervals["op_iv"]
    for o in ops:
        rec = iv_values[o.id]
        assert rec["duration"] == o.duration


def test_duration_modes_mutually_exclusive():
    """All three duration modes are mutually exclusive."""
    iv = LXIntervalVariable[Op]("test")
    iv.duration_fixed(5)
    with pytest.raises(ValueError, match="mutually exclusive"):
        iv.duration_per_instance(lambda o: o.duration)
    iv2 = LXIntervalVariable[Op]("test2")
    iv2.duration_per_instance(lambda o: o.duration)
    with pytest.raises(ValueError, match="mutually exclusive"):
        iv2.duration_fixed(5)


# ==================== Cross-solver: NotImplementedError ====================


@pytest.mark.parametrize("solver_name", ["ortools", "glpk"])
def test_non_cpsat_solvers_reject_no_overlap(solver_name):
    """Solvers that don't support scheduling primitives must raise, not silently ignore."""
    model: LXModel = LXModel("m")
    x = (
        LXVariable[Op, int]("x")
        .integer()
        .bounds(0, 10)
        .indexed_by(lambda o: o.id)
        .from_data([Op("a", "M", 1)])
    )
    model.add_variable(x)
    model.minimize(LXLinearExpression().add_term(x, coeff=lambda _o: 1.0))

    iv = LXIntervalVariable("iv")
    model.add_interval_variable(iv)
    model.add_constraint(LXNoOverlapConstraint("c", [iv]))

    with pytest.raises(NotImplementedError):
        LXOptimizer().use_solver(solver_name).solve(model)
