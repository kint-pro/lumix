"""Interval variable for scheduling problems (CP-SAT IntervalVar wrapper)."""

from __future__ import annotations

from typing import Any, Callable, Generic, List, Optional, TypeVar

from typing_extensions import Self

from .variables import LXVariable

TModel = TypeVar("TModel")
TIndex = TypeVar("TIndex")


class LXIntervalVariable(Generic[TModel]):
    """
    Interval variable family for scheduling.

    Composes three LXVariable instances: start, end, and either a fixed
    duration (int) or a duration variable. Used in conjunction with
    LXNoOverlapConstraint and (future) LXCumulativeConstraint to express
    scheduling primitives natively on solvers that support them (CP-SAT).

    Builder API mirrors LXVariable: chained `.start(...)`, `.end(...)`,
    `.duration_fixed(...)` xor `.duration_var(...)`, `.indexed_by(...)`,
    `.from_data(...)`, all returning Self.

    Example::

        interval = (
            LXIntervalVariable[Op]("op_iv")
            .start(start_var)
            .end(end_var)
            .duration_fixed(5)
            .indexed_by(lambda o: o.id)
            .from_data(ops)
        )
        model.add_interval_variable(interval)
        model.add_constraint(LXNoOverlapConstraint("m1_no_overlap", [interval]))
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.start_var: Optional[LXVariable] = None
        self.end_var: Optional[LXVariable] = None
        # Three mutually-exclusive duration modes:
        # - duration: same int for every instance ("fixed")
        # - duration_func: lambda(data) -> int for per-instance fixed duration
        # - duration_var_ref: LXVariable with per-instance integer/continuous duration
        self.duration: Optional[int] = None
        self.duration_func: Optional[Callable[[TModel], int]] = None
        self.duration_var_ref: Optional[LXVariable] = None
        self.index_func: Optional[Callable[[TModel], TIndex]] = None
        self._data: Optional[List[TModel]] = None

    def __deepcopy__(self, memo: dict) -> "LXIntervalVariable":
        from copy import deepcopy

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        result.name = self.name
        result.duration = self.duration
        result.start_var = deepcopy(self.start_var, memo) if self.start_var is not None else None
        result.end_var = deepcopy(self.end_var, memo) if self.end_var is not None else None
        result.duration_var_ref = (
            deepcopy(self.duration_var_ref, memo) if self.duration_var_ref is not None else None
        )

        # Lambdas with closures: detach safely the same way LXVariable does.
        from ..utils.copy_utils import copy_function_detaching_closure
        if self.index_func is not None:
            result.index_func = copy_function_detaching_closure(self.index_func, memo)
        else:
            result.index_func = None
        if self.duration_func is not None:
            result.duration_func = copy_function_detaching_closure(
                self.duration_func, memo
            )
        else:
            result.duration_func = None

        if self._data is not None:
            from ..utils.copy_utils import materialize_and_detach_list
            result._data = materialize_and_detach_list(self._data, memo)
        else:
            result._data = None

        return result

    def start(self, var: LXVariable) -> Self:
        """Bind start variable family. Returns self for chaining."""
        self.start_var = var
        return self

    def end(self, var: LXVariable) -> Self:
        """Bind end variable family. Returns self for chaining."""
        self.end_var = var
        return self

    def _check_duration_unset(self, kind_new: str) -> None:
        already = []
        if self.duration is not None:
            already.append("duration_fixed")
        if self.duration_func is not None:
            already.append("duration_per_instance")
        if self.duration_var_ref is not None:
            already.append("duration_var")
        if already:
            raise ValueError(
                f"Interval '{self.name}' already has {', '.join(already)}; "
                f"{kind_new} is mutually exclusive."
            )

    def duration_fixed(self, value: int) -> Self:
        """
        Set a fixed integer duration applied to every interval instance.

        Mutually exclusive with `duration_per_instance(...)` and `duration_var(...)`.
        """
        self._check_duration_unset("duration_fixed")
        self.duration = int(value)
        return self

    def duration_per_instance(self, func: Callable[[TModel], int]) -> Self:
        """
        Provide a lambda that returns a fixed integer duration per data instance.

        Use this when each instance has its own fixed duration (e.g., JSP ops
        with per-op processing time). The function is called once per
        ``from_data`` instance during solver build.

        Mutually exclusive with `duration_fixed(...)` and `duration_var(...)`.
        """
        self._check_duration_unset("duration_per_instance")
        self.duration_func = func
        return self

    def duration_var(self, var: LXVariable) -> Self:
        """
        Bind a duration variable family (per-instance variable duration).

        Mutually exclusive with `duration_fixed(...)` and `duration_per_instance(...)`.
        """
        self._check_duration_unset("duration_var")
        self.duration_var_ref = var
        return self

    def indexed_by(self, func: Callable[[TModel], TIndex]) -> Self:
        """Provide an index extractor lambda. Returns self for chaining."""
        self.index_func = func
        return self

    def from_data(self, data: List[TModel]) -> Self:
        """Bind concrete data instances. Returns self for chaining."""
        self._data = data
        return self

    def get_instances(self) -> List[TModel]:
        """Return data instances bound to this interval; empty list if none."""
        return list(self._data) if self._data is not None else []


__all__ = ["LXIntervalVariable"]
