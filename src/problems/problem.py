"""High-level symbolic problem API for the primal-dual solver.

The public modeling flow mirrors KKT-HardNet's problem-definition API::

    problem = Problem(name="example")
    p = problem.add_parameter(["p1", "p2"])
    y = problem.add_variable(["y1", "y2"])
    problem.objective = 0.5 * (y.y1**2 + y.y2**2)
    problem.constraints.add(
        y.y1 + y.y2 == p.p1,
        y.y1 >= 0.0,
    )
    solution = problem.optimize(parameters={"p1": 1.0, "p2": 0.0})

The generic numerical path delegates to the existing primal-dual kernels.
Distance-projection objectives additionally expose dense-batched, matrix-free
sparse, and shared-linear batch modes.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from ..opt.solvers.bounds import BoundIndices, BoundValues
from ..opt.solvers.iteration import FixedMuOptions, FixedMuResult, run_fixed_mu_iterations
from ..opt.solvers.line_search import FilterLineSearchOptions
from ..opt.solvers.residuals import NLPQuantities
from ..opt.solvers.state import PrimalDualState
from ..opt.sparse.batch_iteration import (
    BATCH_STATUS_NAMES,
    BatchedFixedMuResult,
    run_batched_fixed_mu_iterations,
)
from ..opt.sparse.projection_iteration import (
    run_shared_linear_projection_batch,
)
from ..opt.sparse.sparse_condensed import (
    prepare_batched_projection_condensed_sparse_solver,
)
from ..opt.sparse.conversion import bounds_from_numpy, indices_from_numpy
from ..opt.sparse.types import (
    NLPQuantities as JaxNLPQuantities,
    NewtonRegularization as JaxNewtonRegularization,
    PrimalDualState as JaxPrimalDualState,
)

# The NumPy reference solver is float64.  Enabling x64 here makes the symbolic
# derivative compiler agree with that representation on both CPU and CUDA JAX.
jax.config.update("jax_enable_x64", True)

ArrayLike = Any
ScalarFunction = Callable[[jnp.ndarray, jnp.ndarray], jax.Array]
VectorFunction = Callable[[jnp.ndarray, jnp.ndarray], jax.Array]


def _resolve_jax_device(device: Any | None):
    """Resolve CPU/GPU/CUDA aliases to one concrete local JAX device.

    Accepted strings include ``cpu``, ``gpu``, ``cuda``, ``gpu:0`` and
    ``cuda:1``.  Explicit placement is important for accelerator execution:
    all arrays supplied to one compiled solve are committed to the same device.
    """

    if device is None:
        return None
    if hasattr(device, "platform") and hasattr(device, "id"):
        return device

    requested_text = str(device).strip().lower()
    platform_text, separator, device_id_text = requested_text.partition(":")
    aliases = {"cuda": "gpu", "nvidia": "gpu"}
    requested_platform = aliases.get(platform_text, platform_text)
    if requested_platform not in {"cpu", "gpu", "tpu"}:
        raise ValueError(
            "Unknown JAX backend/device {!r}; expected cpu, gpu, cuda, "
            "gpu:<id>, or cuda:<id>.".format(device)
        )
    requested_id = None
    if separator:
        try:
            requested_id = int(device_id_text)
        except ValueError as error:
            raise ValueError(
                f"Device id in {device!r} must be an integer."
            ) from error

    try:
        candidates = list(jax.devices(requested_platform))
    except RuntimeError:
        candidates = []
    if requested_id is not None:
        candidates = [candidate for candidate in candidates if candidate.id == requested_id]
    if not candidates:
        visible = [
            (candidate.platform, candidate.device_kind, candidate.id)
            for candidate in jax.devices()
        ]
        extra = ""
        if requested_platform == "gpu":
            extra = (
                " JAX must be installed with NVIDIA GPU support. Native Windows "
                "CUDA execution is not supported by JAX; use Linux or WSL2."
            )
        raise ValueError(
            f"Requested JAX device {device!r} is unavailable; visible devices={visible}."
            + extra
        )
    return candidates[0]


def _transpose_gather_padding_fraction(
    equality_support: np.ndarray,
    equality_mask: np.ndarray,
    inequality_support: np.ndarray,
    inequality_mask: np.ndarray,
    n_variables: int,
) -> float:
    """Estimate padding in the column-gather transpose representation."""

    supports = []
    masks = []
    for support, mask in (
        (equality_support, equality_mask),
        (inequality_support, inequality_mask),
    ):
        support = np.asarray(support, dtype=np.int32)
        mask = np.asarray(mask, dtype=bool)
        if support.ndim == 2 and support.shape[0]:
            supports.append(support)
            masks.append(mask)
    if not supports or n_variables <= 0:
        return 0.0
    support = np.concatenate(supports, axis=0)
    mask = np.concatenate(masks, axis=0)
    degrees = np.bincount(support[mask], minlength=n_variables)
    width = int(np.max(degrees, initial=0))
    capacity = n_variables * width
    if capacity == 0:
        return 0.0
    return float((capacity - int(np.sum(mask))) / capacity)


def _transpose_bucket_padding_fraction(
    equality_support: np.ndarray,
    equality_mask: np.ndarray,
    inequality_support: np.ndarray,
    inequality_mask: np.ndarray,
    n_variables: int,
) -> float:
    """Estimate padding for power-of-two degree-bucketed transpose gathers."""

    supports = []
    masks = []
    for support, mask in (
        (equality_support, equality_mask),
        (inequality_support, inequality_mask),
    ):
        support = np.asarray(support, dtype=np.int32)
        mask = np.asarray(mask, dtype=bool)
        if support.ndim == 2 and support.shape[0]:
            supports.append(support)
            masks.append(mask)
    if not supports or n_variables <= 0:
        return 0.0
    support = np.concatenate(supports, axis=0)
    mask = np.concatenate(masks, axis=0)
    degrees = np.bincount(support[mask], minlength=n_variables)
    capacity = 0
    valid = 0
    for degree in degrees:
        degree = int(degree)
        if degree == 0:
            continue
        capacity += 1 << (degree - 1).bit_length()
        valid += degree
    if capacity == 0:
        return 0.0
    return float((capacity - valid) / capacity)


def _coerce_names(names: str | Iterable[str], *, allow_empty: bool = False) -> list[str]:
    if isinstance(names, str):
        result = [names]
    else:
        result = [str(name) for name in names]
    if not result and not allow_empty:
        raise ValueError("At least one name is required.")
    if any(not name for name in result):
        raise ValueError("Symbol names must be nonempty strings.")
    if len(set(result)) != len(result):
        raise ValueError(f"Symbol names must be unique; received {result}.")
    return result


@dataclass(frozen=True)
class _EvaluationContext:
    variables: jax.Array
    parameters: jax.Array
    variable_index: Mapping[str, int]
    parameter_index: Mapping[str, int]

    def value(self, kind: str, name: str):
        if kind == "variable":
            return self.variables[self.variable_index[name]]
        if kind == "parameter":
            return self.parameters[self.parameter_index[name]]
        raise KeyError(f"Unknown symbol kind {kind!r}.")


@dataclass(frozen=True)
class _AffineScalar:
    """Parameter-only scalar used by the affine expression decomposition."""

    function: Callable[[_EvaluationContext], Any]
    parameter_dependencies: frozenset[str] = frozenset()

    def evaluate(self, context: _EvaluationContext):
        return self.function(context)


def _constant_affine_scalar(value: float) -> _AffineScalar:
    scalar = float(value)
    return _AffineScalar(
        lambda context, scalar=scalar: jnp.asarray(
            scalar, dtype=context.variables.dtype
        )
    )


def _combine_affine_scalars(
    left: _AffineScalar,
    right: _AffineScalar,
    operator: Callable,
) -> _AffineScalar:
    return _AffineScalar(
        lambda context, left=left, right=right, operator=operator: operator(
            left.evaluate(context), right.evaluate(context)
        ),
        left.parameter_dependencies | right.parameter_dependencies,
    )


class Expression:
    """Scalar symbolic expression backed by a JAX-traceable callable.

    The dependency metadata is static and is used only to select and compile
    projection-specialized sparse kernels.  Numerical derivatives are still
    evaluated by JAX.
    """

    __array_priority__ = 1000

    def __init__(
        self,
        function: Callable[[_EvaluationContext], Any],
        text: str,
        *,
        variable_dependencies: Iterable[str] = (),
        parameter_dependencies: Iterable[str] = (),
        affine_in_variables: bool = True,
        symbol_kind: str | None = None,
        symbol_name: str | None = None,
        projection_target_names: tuple[str, ...] | None = None,
        affine_coefficients: Mapping[str, _AffineScalar] | None = None,
        affine_offset: _AffineScalar | None = None,
    ) -> None:
        self._function = function
        self.text = str(text)
        self.variable_dependencies = frozenset(variable_dependencies)
        self.parameter_dependencies = frozenset(parameter_dependencies)
        self.affine_in_variables = bool(affine_in_variables)
        self.symbol_kind = symbol_kind
        self.symbol_name = symbol_name
        self.projection_target_names = projection_target_names
        if affine_coefficients is not None:
            self.affine_coefficients = dict(affine_coefficients)
            self.affine_offset = affine_offset or _constant_affine_scalar(0.0)
        elif self.affine_in_variables and not self.variable_dependencies:
            self.affine_coefficients = {}
            self.affine_offset = _AffineScalar(
                function,
                frozenset(parameter_dependencies),
            )
        else:
            self.affine_coefficients = None
            self.affine_offset = None

    def evaluate(self, context: _EvaluationContext):
        return self._function(context)

    def _combine_metadata(self, right: "Expression", symbol: str) -> tuple[frozenset[str], frozenset[str], bool]:
        variable_dependencies = self.variable_dependencies | right.variable_dependencies
        parameter_dependencies = self.parameter_dependencies | right.parameter_dependencies
        if symbol in {"+", "-"}:
            affine = self.affine_in_variables and right.affine_in_variables
        elif symbol == "*":
            affine = (
                self.affine_in_variables
                and right.affine_in_variables
                and (not self.variable_dependencies or not right.variable_dependencies)
            )
        elif symbol == "/":
            affine = self.affine_in_variables and not right.variable_dependencies
        else:
            affine = not variable_dependencies
        return variable_dependencies, parameter_dependencies, affine

    def _binary(self, other: Any, operator: Callable, symbol: str) -> "Expression":
        right = _as_expression(other)
        variable_dependencies, parameter_dependencies, affine = self._combine_metadata(right, symbol)
        affine_coefficients = None
        affine_offset = None
        if affine and self.affine_coefficients is not None and right.affine_coefficients is not None:
            if symbol in {"+", "-"}:
                names = self.variable_dependencies | right.variable_dependencies
                affine_coefficients = {}
                zero = _constant_affine_scalar(0.0)
                for name in names:
                    affine_coefficients[name] = _combine_affine_scalars(
                        self.affine_coefficients.get(name, zero),
                        right.affine_coefficients.get(name, zero),
                        operator,
                    )
                affine_offset = _combine_affine_scalars(
                    self.affine_offset,
                    right.affine_offset,
                    operator,
                )
            elif symbol == "*":
                if not self.variable_dependencies:
                    scalar = self.affine_offset
                    variable_expression = right
                else:
                    scalar = right.affine_offset
                    variable_expression = self
                affine_coefficients = {
                    name: _combine_affine_scalars(coefficient, scalar, operator)
                    for name, coefficient in variable_expression.affine_coefficients.items()
                }
                affine_offset = _combine_affine_scalars(
                    variable_expression.affine_offset,
                    scalar,
                    operator,
                )
            elif symbol == "/":
                affine_coefficients = {
                    name: _combine_affine_scalars(coefficient, right.affine_offset, operator)
                    for name, coefficient in self.affine_coefficients.items()
                }
                affine_offset = _combine_affine_scalars(
                    self.affine_offset,
                    right.affine_offset,
                    operator,
                )
        return Expression(
            lambda context, left=self, right=right: operator(
                left.evaluate(context), right.evaluate(context)
            ),
            f"({self.text} {symbol} {right.text})",
            variable_dependencies=variable_dependencies,
            parameter_dependencies=parameter_dependencies,
            affine_in_variables=affine,
            affine_coefficients=affine_coefficients,
            affine_offset=affine_offset,
        )

    def _reverse_binary(self, other: Any, operator: Callable, symbol: str) -> "Expression":
        left = _as_expression(other)
        return left._binary(self, operator, symbol)

    def __add__(self, other: Any) -> "Expression":
        return self._binary(other, lambda left, right: left + right, "+")

    def __radd__(self, other: Any) -> "Expression":
        return self._reverse_binary(other, lambda left, right: left + right, "+")

    def __sub__(self, other: Any) -> "Expression":
        return self._binary(other, lambda left, right: left - right, "-")

    def __rsub__(self, other: Any) -> "Expression":
        return self._reverse_binary(other, lambda left, right: left - right, "-")

    def __mul__(self, other: Any) -> "Expression":
        return self._binary(other, lambda left, right: left * right, "*")

    def __rmul__(self, other: Any) -> "Expression":
        return self._reverse_binary(other, lambda left, right: left * right, "*")

    def __truediv__(self, other: Any) -> "Expression":
        return self._binary(other, lambda left, right: left / right, "/")

    def __rtruediv__(self, other: Any) -> "Expression":
        return self._reverse_binary(other, lambda left, right: left / right, "/")

    def __pow__(self, other: Any) -> "Expression":
        right = _as_expression(other)
        variable_dependencies = self.variable_dependencies | right.variable_dependencies
        parameter_dependencies = self.parameter_dependencies | right.parameter_dependencies
        return Expression(
            lambda context, left=self, right=right: left.evaluate(context) ** right.evaluate(context),
            f"({self.text} ** {right.text})",
            variable_dependencies=variable_dependencies,
            parameter_dependencies=parameter_dependencies,
            affine_in_variables=not variable_dependencies,
        )

    def __rpow__(self, other: Any) -> "Expression":
        return _as_expression(other).__pow__(self)

    def __neg__(self) -> "Expression":
        affine_coefficients = None
        affine_offset = None
        if self.affine_coefficients is not None:
            minus_one = _constant_affine_scalar(-1.0)
            affine_coefficients = {
                name: _combine_affine_scalars(
                    coefficient, minus_one, lambda left, right: left * right
                )
                for name, coefficient in self.affine_coefficients.items()
            }
            affine_offset = _combine_affine_scalars(
                self.affine_offset, minus_one, lambda left, right: left * right
            )
        return Expression(
            lambda context, expression=self: -expression.evaluate(context),
            f"(-{self.text})",
            variable_dependencies=self.variable_dependencies,
            parameter_dependencies=self.parameter_dependencies,
            affine_in_variables=self.affine_in_variables,
            affine_coefficients=affine_coefficients,
            affine_offset=affine_offset,
        )

    def __abs__(self) -> "Expression":
        return Expression(
            lambda context, expression=self: jnp.abs(expression.evaluate(context)),
            f"abs({self.text})",
            variable_dependencies=self.variable_dependencies,
            parameter_dependencies=self.parameter_dependencies,
            affine_in_variables=not self.variable_dependencies,
        )

    def __le__(self, other: Any) -> "Constraint":
        right = _as_expression(other)
        return Constraint(self - right, "ineq", f"{self.text} <= {right.text}")

    def __ge__(self, other: Any) -> "Constraint":
        right = _as_expression(other)
        # Normalize every inequality to residual <= 0.
        return Constraint(right - self, "ineq", f"{self.text} >= {right.text}")

    def __eq__(self, other: Any) -> "Constraint":  # type: ignore[override]
        right = _as_expression(other)
        return Constraint(self - right, "eq", f"{self.text} == {right.text}")

    def __bool__(self) -> bool:
        raise TypeError(
            "Symbolic expressions do not have a truth value. Use ==, <=, or >= "
            "inside problem.constraints.add(...)."
        )

    def __repr__(self) -> str:
        return f"Expression({self.text})"


class VectorExpression:
    """One-dimensional symbolic expression with elementwise comparisons."""

    __array_priority__ = 1000

    def __init__(self, components: Sequence[Expression], text: str) -> None:
        self.components = tuple(components)
        self.text = str(text)

    @property
    def size(self) -> int:
        return len(self.components)

    def evaluate(self, context: _EvaluationContext):
        if not self.components:
            return jnp.empty((0,), dtype=context.variables.dtype)
        return jnp.stack([component.evaluate(context) for component in self.components])

    def __getitem__(self, index: int) -> Expression:
        return self.components[index]

    def __iter__(self) -> Iterator[Expression]:
        return iter(self.components)

    def _coerce(self, other: Any) -> "VectorExpression":
        if isinstance(other, VectorExpression):
            if other.size != self.size:
                raise ValueError(
                    f"Vector sizes must match; received {self.size} and {other.size}."
                )
            return other
        if isinstance(other, _SymbolNamespace):
            return self._coerce(other.vector())
        if np.isscalar(other) or isinstance(other, Expression):
            scalar = _as_expression(other)
            return VectorExpression([scalar] * self.size, scalar.text)
        values = np.asarray(other)
        if values.ndim != 1 or values.size != self.size:
            raise ValueError(
                f"Expected a vector of length {self.size}; received shape {values.shape}."
            )
        return VectorExpression(
            [_as_expression(value) for value in values.tolist()],
            repr(values.tolist()),
        )

    def _binary(self, other: Any, operator: Callable, symbol: str) -> "VectorExpression":
        right = self._coerce(other)
        return VectorExpression(
            [operator(left, rhs) for left, rhs in zip(self, right, strict=True)],
            f"({self.text} {symbol} {right.text})",
        )

    def __add__(self, other: Any) -> "VectorExpression":
        return self._binary(other, lambda left, right: left + right, "+")

    def __radd__(self, other: Any) -> "VectorExpression":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "VectorExpression":
        return self._binary(other, lambda left, right: left - right, "-")

    def __rsub__(self, other: Any) -> "VectorExpression":
        return (-self).__add__(other)

    def __mul__(self, other: Any) -> "VectorExpression":
        return self._binary(other, lambda left, right: left * right, "*")

    def __rmul__(self, other: Any) -> "VectorExpression":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "VectorExpression":
        return self._binary(other, lambda left, right: left / right, "/")

    def __neg__(self) -> "VectorExpression":
        return VectorExpression([-component for component in self], f"(-{self.text})")

    def __le__(self, other: Any) -> list["Constraint"]:
        right = self._coerce(other)
        return [left <= rhs for left, rhs in zip(self, right, strict=True)]

    def __ge__(self, other: Any) -> list["Constraint"]:
        right = self._coerce(other)
        return [left >= rhs for left, rhs in zip(self, right, strict=True)]

    def __eq__(self, other: Any) -> list["Constraint"]:  # type: ignore[override]
        right = self._coerce(other)
        return [left == rhs for left, rhs in zip(self, right, strict=True)]

    def sum(self) -> Expression:
        return sum(self.components, _as_expression(0.0))

    def __repr__(self) -> str:
        return f"VectorExpression(size={self.size}, text={self.text!r})"


def _as_expression(value: Any) -> Expression:
    if isinstance(value, Expression):
        return value
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if not np.isscalar(value):
        raise TypeError(
            "Expected a scalar symbolic expression or numeric constant; "
            f"received {type(value).__name__}."
        )
    scalar = float(value)
    return Expression(
        lambda _context, value=scalar: jnp.asarray(value),
        repr(scalar),
    )


@dataclass(frozen=True)
class Constraint:
    """Normalized scalar constraint."""

    residual: Expression
    kind: str
    text: str

    def __post_init__(self) -> None:
        if self.kind not in {"eq", "ineq"}:
            raise ValueError(f"Constraint kind must be 'eq' or 'ineq'; got {self.kind!r}.")


class ConstraintList:
    """Mutable collection used by ``problem.constraints.add(...)``."""

    def __init__(self) -> None:
        self.items: list[Constraint] = []

    def add(self, *constraints: Constraint | Sequence[Constraint]) -> "ConstraintList":
        for constraint in constraints:
            if isinstance(constraint, (list, tuple)):
                self.add(*constraint)
                continue
            if not isinstance(constraint, Constraint):
                raise TypeError(
                    "constraints.add(...) expects symbolic comparisons created "
                    "with ==, <=, or >=."
                )
            self.items.append(constraint)
        return self

    def equality(self, expression: Any, *, name: str | None = None) -> "ConstraintList":
        residual = _as_expression(expression)
        self.items.append(Constraint(residual, "eq", name or f"{residual.text} == 0"))
        return self

    def inequality(self, expression: Any, *, name: str | None = None) -> "ConstraintList":
        residual = _as_expression(expression)
        self.items.append(Constraint(residual, "ineq", name or f"{residual.text} <= 0"))
        return self

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[Constraint]:
        return iter(self.items)


class _SymbolNamespace:
    def __init__(self, problem: "Problem", kind: str) -> None:
        self._problem = problem
        self._kind = kind

    @property
    def names(self) -> tuple[str, ...]:
        if self._kind == "parameter":
            return tuple(self._problem.parameter_names)
        return tuple(self._problem.variable_names)

    def __getattr__(self, name: str) -> Expression:
        return self[name]

    def __getitem__(self, name_or_index: str | int) -> Expression:
        if isinstance(name_or_index, int):
            name = self.names[name_or_index]
        else:
            name = str(name_or_index)
        self._problem._check_symbol(self._kind, name)
        affine_coefficients = None
        affine_offset = None
        if self._kind == "variable":
            affine_coefficients = {name: _constant_affine_scalar(1.0)}
            affine_offset = _constant_affine_scalar(0.0)
        return Expression(
            lambda context, kind=self._kind, name=name: context.value(kind, name),
            name,
            variable_dependencies=(name,) if self._kind == "variable" else (),
            parameter_dependencies=(name,) if self._kind == "parameter" else (),
            affine_in_variables=True,
            symbol_kind=self._kind,
            symbol_name=name,
            affine_coefficients=affine_coefficients,
            affine_offset=affine_offset,
        )

    def vector(self) -> VectorExpression:
        token = "p" if self._kind == "parameter" else "y"
        return VectorExpression([self[name] for name in self.names], token)

    def __len__(self) -> int:
        return len(self.names)

    def __iter__(self) -> Iterator[Expression]:
        return iter(self.vector())


class NamedValues(Mapping[str, float]):
    """Read-only mapping with attribute access for named numerical values."""

    def __init__(self, names: Sequence[str], values: ArrayLike) -> None:
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if vector.size != len(names):
            raise ValueError(
                f"Expected {len(names)} values for {list(names)}; received {vector.size}."
            )
        self._data = dict(zip(names, vector.tolist(), strict=True))

    def __getitem__(self, key: str) -> float:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, name: str) -> float:
        try:
            return self._data[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def vector(self) -> np.ndarray:
        return np.asarray(list(self._data.values()), dtype=np.float64)

    def __repr__(self) -> str:
        return f"NamedValues({self._data!r})"


class BatchNamedValues(Mapping[str, np.ndarray]):
    """Read-only named columns for a batch of numerical values."""

    def __init__(self, names: Sequence[str], values: ArrayLike) -> None:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(names):
            raise ValueError(
                "Expected a matrix with shape (batch, number_of_names); "
                f"received {matrix.shape} for {len(names)} names."
            )
        self._names = tuple(names)
        self._matrix = matrix
        self._index = {name: index for index, name in enumerate(self._names)}

    def __getitem__(self, key: str) -> np.ndarray:
        return self._matrix[:, self._index[key]]

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def __getattr__(self, name: str) -> np.ndarray:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    @property
    def batch_size(self) -> int:
        return int(self._matrix.shape[0])

    def matrix(self, *, copy: bool = True) -> np.ndarray:
        """Return the batch matrix, optionally as a zero-copy read-only view."""
        if copy:
            return self._matrix.copy()
        view = self._matrix.view()
        view.flags.writeable = False
        return view

    def vector(self) -> np.ndarray:
        """Return the full ``(batch, width)`` matrix for API symmetry."""
        return self.matrix()

    def row(self, index: int) -> NamedValues:
        return NamedValues(self._names, self._matrix[index])

    def __repr__(self) -> str:
        return f"BatchNamedValues(names={self._names!r}, shape={self._matrix.shape!r})"


@dataclass(frozen=True)
class ProjectionStructure:
    """Static sparse structure shared by every problem in a projection batch.

    The padded supports and masks are computed once in Python before JIT
    compilation.  Numerical iterations update only local derivative values.
    """

    target_parameter_indices: tuple[int, ...]
    equality_variable_indices: tuple[tuple[int, ...], ...]
    inequality_variable_indices: tuple[tuple[int, ...], ...]
    equality_jacobian_rows: tuple[int, ...]
    equality_jacobian_cols: tuple[int, ...]
    inequality_jacobian_rows: tuple[int, ...]
    inequality_jacobian_cols: tuple[int, ...]
    max_row_nonzeros: int
    equality_support: tuple[tuple[int, ...], ...]
    equality_support_mask: tuple[tuple[bool, ...], ...]
    inequality_support: tuple[tuple[int, ...], ...]
    inequality_support_mask: tuple[tuple[bool, ...], ...]
    constraints_affine_in_variables: bool
    estimated_reduced_kkt_density: float
    estimated_condensed_kkt_density: float

    @property
    def jacobian_nonzeros(self) -> int:
        return sum(map(len, self.equality_variable_indices)) + sum(
            map(len, self.inequality_variable_indices)
        )


@dataclass(frozen=True)
class CompiledProblem:
    """Low-level objects generated from one parameter binding."""

    name: str
    variable_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    parameter_values: np.ndarray
    initial_state: PrimalDualState
    indices: BoundIndices
    bounds: BoundValues
    evaluate_nlp: Callable[[PrimalDualState], NLPQuantities]
    objective_function: ScalarFunction
    equality_function: VectorFunction
    inequality_function: VectorFunction

    def objective(self, variables: ArrayLike) -> float:
        value = self.objective_function(
            jnp.asarray(variables, dtype=jnp.float64),
            jnp.asarray(self.parameter_values, dtype=jnp.float64),
        )
        return float(np.asarray(value))

    def equalities(self, variables: ArrayLike) -> np.ndarray:
        value = self.equality_function(
            jnp.asarray(variables, dtype=jnp.float64),
            jnp.asarray(self.parameter_values, dtype=jnp.float64),
        )
        return np.asarray(value, dtype=np.float64)

    def inequalities(self, variables: ArrayLike) -> np.ndarray:
        value = self.inequality_function(
            jnp.asarray(variables, dtype=jnp.float64),
            jnp.asarray(self.parameter_values, dtype=jnp.float64),
        )
        return np.asarray(value, dtype=np.float64)


@dataclass(frozen=True)
class CompiledBatchProblem:
    """JAX-compiled same-structure batch of parameterized problems."""

    name: str
    variable_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    parameter_values: np.ndarray
    initial_state: JaxPrimalDualState
    indices: Any
    bounds: Any
    evaluate_nlp_single: Callable[[JaxPrimalDualState, jax.Array], JaxNLPQuantities]
    objective_function: ScalarFunction
    equality_function: VectorFunction
    inequality_function: VectorFunction
    initialize_state_function: Callable[[jax.Array, jax.Array, jax.Array], JaxPrimalDualState]
    objective_batch_function: Callable[[jax.Array, jax.Array], jax.Array]
    equality_batch_function: Callable[[jax.Array, jax.Array], jax.Array]
    inequality_batch_function: Callable[[jax.Array, jax.Array], jax.Array]
    equality_jacobian_function: Callable[[jax.Array, jax.Array], jax.Array]
    inequality_jacobian_function: Callable[[jax.Array, jax.Array], jax.Array]
    equality_sparse_jacobian_values_function: Callable | None = None
    inequality_sparse_jacobian_values_function: Callable | None = None
    constraint_hvp_function: Callable | None = None
    equality_local_derivatives_function: Callable | None = None
    inequality_local_derivatives_function: Callable | None = None
    equality_affine_offset_function: Callable | None = None
    inequality_affine_offset_function: Callable | None = None
    equality_constant_local_jacobian: np.ndarray | None = None
    inequality_constant_local_jacobian: np.ndarray | None = None
    projection_structure: ProjectionStructure | None = None

    @property
    def batch_size(self) -> int:
        return int(self.parameter_values.shape[0])

    def evaluate_nlp(self, state: JaxPrimalDualState) -> JaxNLPQuantities:
        return jax.vmap(self.evaluate_nlp_single, in_axes=(0, 0))(
            state, jnp.asarray(self.parameter_values, dtype=jnp.float64)
        )

    def objectives(self, variables: ArrayLike) -> np.ndarray:
        values = self.objective_batch_function(
            jnp.asarray(variables, dtype=jnp.float64),
            jnp.asarray(self.parameter_values, dtype=jnp.float64),
        )
        return np.asarray(values, dtype=np.float64)

    def equalities(self, variables: ArrayLike) -> np.ndarray:
        values = self.equality_batch_function(
            jnp.asarray(variables, dtype=jnp.float64),
            jnp.asarray(self.parameter_values, dtype=jnp.float64),
        )
        return np.asarray(values, dtype=np.float64)

    def inequalities(self, variables: ArrayLike) -> np.ndarray:
        values = self.inequality_batch_function(
            jnp.asarray(variables, dtype=jnp.float64),
            jnp.asarray(self.parameter_values, dtype=jnp.float64),
        )
        return np.asarray(values, dtype=np.float64)


@dataclass
class PreparedSparseProjectionBatchSolver:
    """Persistent device-resident sparse projection executor.

    The symbolic structure, initialization graph, sparse solver executable and
    static bound data are prepared once for a fixed batch shape.  Calls to
    :meth:`solve` accept NumPy or JAX arrays and return JAX arrays on the
    selected device without forcing a host synchronization.
    """

    compiled: CompiledBatchProblem
    sparse_solver: Any
    device: Any
    interior_margin: float
    target_parameter_indices: jax.Array
    physical_parameter_indices: jax.Array
    batch_size: int
    parameter_count: int
    transpose_mode: str = "scatter"
    gmres_unroll: int = 1
    arnoldi_prefix_block: int = 0
    gpu_kkt_mode: str = "xla"
    pallas_kkt_block_size: int = 64
    store_residual_history: bool = True
    adjoint_solver_mode: str = "gmres"
    adjoint_gmres_restart: int = 30
    adjoint_gmres_maxiter: int = 50
    adjoint_gmres_unroll: int = 1
    adjoint_arnoldi_prefix_block: int = 0
    solve_default_function: Any | None = None
    solve_default_with_linearization_function: Any | None = None
    solve_default_loss_gradient_functions: dict[str, Any] | None = None

    @property
    def backend(self) -> str:
        return "cuda" if self.device.platform == "gpu" else self.device.platform

    def put_parameters(self, parameters: Any) -> jax.Array:
        array = jnp.asarray(parameters, dtype=jnp.float64)
        if array.ndim != 2:
            raise ValueError("parameters must be a rank-two array.")
        expected = (self.batch_size, self.parameter_count)
        if tuple(array.shape) != expected:
            raise ValueError(
                f"Prepared solver expected parameter shape {expected}; "
                f"received {tuple(array.shape)}."
            )
        return jax.device_put(array, self.device)

    def put_mu(self, mu: float | Sequence[float] | jax.Array) -> jax.Array:
        mu_array = jax.device_put(jnp.asarray(mu, dtype=jnp.float64), self.device)
        if mu_array.ndim == 0:
            mu_array = jnp.broadcast_to(mu_array, (self.batch_size,))
        if tuple(mu_array.shape) != (self.batch_size,):
            raise ValueError(
                f"mu must be scalar or have shape ({self.batch_size},)."
            )
        return mu_array

    def pack_projection_parameters_jax(
        self,
        problem_parameters: jax.Array,
        y_hat: jax.Array,
    ) -> jax.Array:
        """Pack physical parameters and a projection target without host work.

        This method is intended for use inside larger ``jax.jit`` graphs.  The
        target columns are placed at ``target_parameter_indices`` and all
        remaining problem parameters are placed at
        ``physical_parameter_indices``.
        """

        problem_parameters = jnp.asarray(problem_parameters, dtype=jnp.float64)
        y_hat = jnp.asarray(y_hat, dtype=jnp.float64)
        packed = jnp.zeros(
            (self.batch_size, self.parameter_count), dtype=y_hat.dtype
        )
        packed = packed.at[:, self.target_parameter_indices].set(y_hat)
        return packed.at[:, self.physical_parameter_indices].set(
            problem_parameters
        )

    def solve_with_linearization_jax(
        self,
        parameter_values: jax.Array,
        mu: float | jax.Array = 1e-4,
    ):
        """Trace-safe fixed-shape projection solve for composed JAX graphs."""

        parameter_values = jnp.asarray(parameter_values, dtype=jnp.float64)
        mu_values = jnp.asarray(mu, dtype=jnp.float64)
        if mu_values.ndim == 0:
            mu_values = jnp.broadcast_to(mu_values, (self.batch_size,))
        if self.solve_default_with_linearization_function is not None:
            return self.solve_default_with_linearization_function(
                parameter_values, mu_values
            )
        variable_values = parameter_values[:, self.target_parameter_indices]
        margin = jnp.asarray(self.interior_margin, dtype=jnp.float64)
        initial_state = self.compiled.initialize_state_function(
            variable_values, parameter_values, margin
        )
        return self.sparse_solver._solve_all_with_linearization_function(
            initial_state, parameter_values, mu_values
        )

    def solve_yhat_adjoint_jax(
        self,
        final_state: JaxPrimalDualState,
        final_linearization: Any,
        grad_y_tilde: jax.Array,
        initial_adjoint: jax.Array | None = None,
    ):
        """Trace-safe final-KKT adjoint used by projected neural models."""

        grad_y_tilde = jnp.asarray(grad_y_tilde, dtype=final_state.x.dtype)
        width = (
            final_state.x.shape[1]
            + final_state.y_c.shape[1]
            + final_state.y_d.shape[1]
        )
        if initial_adjoint is None:
            initial_adjoint = jnp.zeros(
                (self.batch_size, width), dtype=final_state.x.dtype
            )
        return self.sparse_solver._solve_yhat_adjoint_function(
            final_state,
            final_linearization,
            grad_y_tilde,
            initial_adjoint,
        )

    def parameter_vjp_from_adjoint_jax(
        self,
        final_state: JaxPrimalDualState,
        parameter_values: jax.Array,
        adjoint: Any,
    ) -> jax.Array:
        """Return the full projection-parameter VJP without another solve.

        The final sparse adjoint supplies the cotangent blocks for stationarity,
        equalities, and inequalities.  Differentiating their contracted KKT
        residual with respect to the packed problem parameters produces
        ``-F_p.T @ lambda``.  Static variable bounds do not contribute.
        """

        def one_sample(state, parameters, lambda_x, lambda_c, lambda_d):
            state = jax.tree_util.tree_map(jax.lax.stop_gradient, state)
            lambda_x = jax.lax.stop_gradient(lambda_x)
            lambda_c = jax.lax.stop_gradient(lambda_c)
            lambda_d = jax.lax.stop_gradient(lambda_d)

            def contracted_residual(current_parameters):
                nlp = self.compiled.evaluate_nlp_single(
                    state, current_parameters
                )
                stationarity = nlp.grad_f
                if nlp.c.shape[0]:
                    stationarity = stationarity + nlp.jac_c.T @ state.y_c
                if nlp.d.shape[0]:
                    stationarity = stationarity + nlp.jac_d.T @ state.y_d
                value = jnp.vdot(lambda_x, stationarity)
                if nlp.c.shape[0]:
                    value = value + jnp.vdot(lambda_c, nlp.c)
                if nlp.d.shape[0]:
                    value = value + jnp.vdot(lambda_d, nlp.d)
                return value

            return -jax.grad(contracted_residual)(parameters)

        return jax.vmap(one_sample)(
            final_state,
            parameter_values,
            adjoint.x,
            adjoint.y_c,
            adjoint.y_d,
        )

    def solve(
        self,
        parameters: Any,
        *,
        mu: float | Sequence[float] | jax.Array = 1e-4,
        initial_values: Any | None = None,
    ) -> BatchedFixedMuResult:
        parameter_array = self.put_parameters(parameters)
        mu_array = self.put_mu(mu)
        if initial_values is None and self.solve_default_function is not None:
            # CUDA and repeated CPU workloads use one outer compiled entry point
            # for target initialization plus the complete Newton solve.
            result = self.solve_default_function(parameter_array, mu_array)
            self.sparse_solver.solve_calls += 1
            return result

        if initial_values is None:
            variable_array = parameter_array[:, self.target_parameter_indices]
        else:
            variable_array = jax.device_put(
                jnp.asarray(initial_values, dtype=jnp.float64), self.device
            )
            expected = (self.batch_size, len(self.compiled.variable_names))
            if tuple(variable_array.shape) != expected:
                raise ValueError(
                    f"initial_values must have shape {expected}; "
                    f"received {tuple(variable_array.shape)}."
                )
        margin = jax.device_put(
            jnp.asarray(self.interior_margin, dtype=jnp.float64), self.device
        )
        initial_state = self.compiled.initialize_state_function(
            variable_array, parameter_array, margin
        )
        return self.sparse_solver.solve(
            initial_state=initial_state,
            parameter_values=parameter_array,
            mu=mu_array,
        )

    def solve_with_linearization(
        self,
        parameters: Any,
        *,
        mu: float | Sequence[float] | jax.Array = 1e-4,
        initial_values: Any | None = None,
    ):
        """Solve and retain the final sparse KKT data for an adjoint call."""

        parameter_array = self.put_parameters(parameters)
        mu_array = self.put_mu(mu)
        if (
            initial_values is None
            and self.solve_default_with_linearization_function is not None
        ):
            result = self.solve_default_with_linearization_function(
                parameter_array, mu_array
            )
            self.sparse_solver.solve_calls += 1
            return result
        if initial_values is None:
            variable_array = parameter_array[:, self.target_parameter_indices]
        else:
            variable_array = jax.device_put(
                jnp.asarray(initial_values, dtype=jnp.float64), self.device
            )
        margin = jax.device_put(
            jnp.asarray(self.interior_margin, dtype=jnp.float64), self.device
        )
        initial_state = self.compiled.initialize_state_function(
            variable_array, parameter_array, margin
        )
        return self.sparse_solver.solve_with_linearization(
            initial_state=initial_state,
            parameter_values=parameter_array,
            mu=mu_array,
        )

    def solve_yhat_adjoint(
        self,
        final_state: JaxPrimalDualState,
        parameters: Any,
        grad_y_tilde: Any,
        *,
        mu: float | Sequence[float] | jax.Array = 1e-4,
        final_linearization: Any | None = None,
        initial_adjoint: Any | None = None,
    ):
        """Return the final-KKT adjoint for projection-input gradients.

        When ``final_linearization`` comes from :meth:`solve`, no final
        derivative or preconditioner data is reevaluated.  ``parameters`` and
        ``mu`` remain accepted for API compatibility but are unused in that
        optimized path.
        """

        gradient_array = jax.device_put(
            jnp.asarray(grad_y_tilde, dtype=jnp.float64), self.device
        )
        expected = (self.batch_size, len(self.compiled.variable_names))
        if tuple(gradient_array.shape) != expected:
            raise ValueError(
                f"grad_y_tilde must have shape {expected}; "
                f"received {tuple(gradient_array.shape)}."
            )
        parameter_array = None
        mu_array = None
        if final_linearization is None:
            parameter_array = self.put_parameters(parameters)
            mu_array = self.put_mu(mu)
        width = (
            final_state.x.shape[1]
            + final_state.y_c.shape[1]
            + final_state.y_d.shape[1]
        )
        initial_solution = None
        if initial_adjoint is not None:
            if hasattr(initial_adjoint, "x"):
                initial_solution = jnp.concatenate(
                    (initial_adjoint.x, initial_adjoint.y_c, initial_adjoint.y_d),
                    axis=1,
                )
            else:
                initial_solution = jax.device_put(
                    jnp.asarray(initial_adjoint, dtype=jnp.float64), self.device
                )
            if tuple(initial_solution.shape) != (self.batch_size, width):
                raise ValueError(
                    "initial_adjoint must be a packed array with shape "
                    f"{(self.batch_size, width)}."
                )
        return self.sparse_solver.solve_yhat_adjoint(
            final_state=final_state,
            final_linearization=final_linearization,
            parameter_values=parameter_array,
            grad_x=gradient_array,
            mu=mu_array,
            initial_solution=initial_solution,
        )

    def solve_loss_yhat_gradient(
        self,
        parameters: Any,
        q_data: Any,
        *,
        q_mode: str = "diagonal",
        valid_mask: Any | None = None,
        mu: float | Sequence[float] | jax.Array = 1e-4,
        initial_values: Any | None = None,
        initial_adjoint: Any | None = None,
    ):
        """Fuse projection, quadratic loss, and implicit ``dL/dy_hat``."""

        if q_mode not in {"identity", "diagonal", "dense"}:
            raise ValueError("q_mode must be identity, diagonal, or dense.")
        parameter_array = self.put_parameters(parameters)
        mu_array = self.put_mu(mu)
        if valid_mask is None:
            valid_mask_array = jnp.ones((self.batch_size,), dtype=bool)
        else:
            valid_mask_array = jax.device_put(
                jnp.asarray(valid_mask, dtype=bool), self.device
            )
            if tuple(valid_mask_array.shape) != (self.batch_size,):
                raise ValueError(
                    f"valid_mask must have shape ({self.batch_size},)."
                )
        q_array = jax.device_put(jnp.asarray(q_data, dtype=jnp.float64), self.device)
        expected_q = (
            (len(self.compiled.variable_names), len(self.compiled.variable_names))
            if q_mode == "dense"
            else (len(self.compiled.variable_names),)
        )
        if tuple(q_array.shape) != expected_q:
            raise ValueError(f"q_data must have shape {expected_q} for {q_mode} mode.")

        # The condensed width is x + equality dual + inequality dual.
        structure = self.compiled.projection_structure
        if structure is None:
            raise RuntimeError("Prepared sparse solver is missing projection structure.")
        n_eq = len(structure.equality_support)
        n_ineq = len(structure.inequality_support)
        width = len(self.compiled.variable_names) + n_eq + n_ineq
        if initial_adjoint is None:
            initial_adjoint_array = jnp.zeros(
                (self.batch_size, width), dtype=jnp.float64
            )
        elif hasattr(initial_adjoint, "x"):
            initial_adjoint_array = jnp.concatenate(
                (initial_adjoint.x, initial_adjoint.y_c, initial_adjoint.y_d),
                axis=1,
            )
        else:
            initial_adjoint_array = jax.device_put(
                jnp.asarray(initial_adjoint, dtype=jnp.float64), self.device
            )
        if tuple(initial_adjoint_array.shape) != (self.batch_size, width):
            raise ValueError(
                f"initial_adjoint must have shape {(self.batch_size, width)}."
            )

        if (
            initial_values is None
            and self.solve_default_loss_gradient_functions is not None
        ):
            function = self.solve_default_loss_gradient_functions[
                "dense" if q_mode == "dense" else "diagonal"
            ]
            result = function(
                parameter_array,
                mu_array,
                q_array,
                valid_mask_array,
                initial_adjoint_array,
            )
            self.sparse_solver.solve_calls += 1
            return result

        if initial_values is None:
            variable_array = parameter_array[:, self.target_parameter_indices]
        else:
            variable_array = jax.device_put(
                jnp.asarray(initial_values, dtype=jnp.float64), self.device
            )
        margin = jax.device_put(
            jnp.asarray(self.interior_margin, dtype=jnp.float64), self.device
        )
        initial_state = self.compiled.initialize_state_function(
            variable_array, parameter_array, margin
        )
        return self.sparse_solver.solve_loss_yhat_gradient(
            initial_state=initial_state,
            parameter_values=parameter_array,
            mu_values=mu_array,
            q_data=q_array,
            valid_mask=valid_mask_array,
            q_mode=q_mode,
            initial_adjoint=initial_adjoint_array,
        )



@dataclass(frozen=True)
class ProblemSolution:
    """Named high-level view of a fixed-barrier solver result."""

    problem_name: str
    variables: NamedValues
    parameters: NamedValues
    objective: float
    equalities: np.ndarray
    inequalities: np.ndarray
    raw_result: FixedMuResult

    @property
    def state(self) -> PrimalDualState:
        return self.raw_result.state

    @property
    def status(self) -> str:
        return self.raw_result.status

    @property
    def converged(self) -> bool:
        return self.raw_result.converged

    @property
    def iterations(self) -> int:
        return self.raw_result.iterations

    @property
    def residual_inf_norm(self) -> float:
        return self.raw_result.final_residual_inf_norm

    def __getitem__(self, variable_name: str) -> float:
        return self.variables[variable_name]


@dataclass(frozen=True)
class BatchProblemSolution:
    """High-level result of a true batched JAX primal--dual solve."""

    problem_name: str
    variables: BatchNamedValues
    parameters: BatchNamedValues
    objectives: np.ndarray
    equalities: np.ndarray
    inequalities: np.ndarray
    raw_result: BatchedFixedMuResult
    projection_mode: str = "generic_dense"
    device_platform: str = "cpu"
    status_codes_host: np.ndarray | None = None
    converged_host: np.ndarray | None = None
    iterations_host: np.ndarray | None = None
    residual_inf_norm_host: np.ndarray | None = None

    @property
    def batch_size(self) -> int:
        return self.variables.batch_size

    @property
    def objective(self) -> np.ndarray:
        """Alias returning the per-problem objective vector."""
        return self.objectives

    @property
    def status(self) -> tuple[str, ...]:
        codes = (
            self.status_codes_host
            if self.status_codes_host is not None
            else np.asarray(self.raw_result.status_codes, dtype=np.int32)
        )
        return tuple(BATCH_STATUS_NAMES[int(code)] for code in codes)

    @property
    def converged(self) -> np.ndarray:
        if self.converged_host is not None:
            return self.converged_host
        return np.asarray(self.raw_result.converged, dtype=bool)

    @property
    def all_converged(self) -> bool:
        return bool(np.all(self.converged))

    @property
    def iterations(self) -> np.ndarray:
        if self.iterations_host is not None:
            return self.iterations_host
        return np.asarray(self.raw_result.iterations, dtype=np.int32)

    @property
    def residual_inf_norm(self) -> np.ndarray:
        if self.residual_inf_norm_host is not None:
            return self.residual_inf_norm_host
        return np.asarray(self.raw_result.final_residual_inf_norm, dtype=np.float64)

    @property
    def state(self) -> JaxPrimalDualState:
        return self.raw_result.state

    def __getitem__(self, variable_name: str) -> np.ndarray:
        return self.variables[variable_name]


class Problem:
    """Symbolic high-level problem builder for the existing solver core."""

    def __init__(self, name: str = "problem") -> None:
        self.name = str(name)
        self.parameter_names: list[str] = []
        self.variable_names: list[str] = []
        self.parameter = _SymbolNamespace(self, "parameter")
        self.variable = _SymbolNamespace(self, "variable")
        self.parameters = self.parameter
        self.variables = self.variable
        self.constraints = ConstraintList()
        self._objective: Expression | None = None
        self._projection_target_names: tuple[str, ...] | None = None
        self._compiled_batch_structure_cache: dict[tuple[Any, ...], CompiledBatchProblem] = {}
        self._compiled_batch_structure_cache_hits = 0
        self._compiled_batch_structure_cache_misses = 0
        self._prepared_sparse_solver_cache: dict[tuple[Any, ...], Any] = {}
        self._prepared_sparse_solver_cache_hits = 0
        self._prepared_sparse_solver_cache_misses = 0

    @property
    def objective(self) -> Expression | None:
        return self._objective

    @objective.setter
    def objective(self, expression: Any) -> None:
        symbolic = _as_expression(expression)
        self._objective = symbolic
        self._projection_target_names = symbolic.projection_target_names

    def squared_distance(
        self,
        variables: VectorExpression | _SymbolNamespace,
        target: VectorExpression | _SymbolNamespace,
    ) -> Expression:
        """Return ``0.5 * ||variables - target||_2^2`` with projection metadata.

        The specialized projection solvers require direct decision-variable
        symbols and direct parameter symbols in matching order.
        """
        variable_vector = variables.vector() if isinstance(variables, _SymbolNamespace) else variables
        target_vector = target.vector() if isinstance(target, _SymbolNamespace) else target
        if isinstance(variable_vector, Sequence) and not isinstance(variable_vector, VectorExpression):
            variable_vector = VectorExpression(
                [_as_expression(item) for item in variable_vector], "projection_variables"
            )
        if isinstance(target_vector, Sequence) and not isinstance(target_vector, VectorExpression):
            target_vector = VectorExpression(
                [_as_expression(item) for item in target_vector], "projection_target"
            )
        if not isinstance(variable_vector, VectorExpression) or not isinstance(target_vector, VectorExpression):
            raise TypeError(
                "squared_distance expects symbolic vectors, namespaces, or sequences "
                "of scalar symbolic expressions."
            )
        if variable_vector.size != target_vector.size:
            raise ValueError(
                "Projection variables and target parameters must have the same size; "
                f"received {variable_vector.size} and {target_vector.size}."
            )
        if variable_vector.size != len(self.variable_names):
            raise ValueError(
                "The projection objective must include every decision variable exactly once."
            )
        variable_names = []
        target_names = []
        for index, (variable, parameter) in enumerate(
            zip(variable_vector, target_vector, strict=True)
        ):
            if variable.symbol_kind != "variable" or variable.symbol_name is None:
                raise ValueError(
                    f"Projection component {index} is not a direct decision-variable symbol."
                )
            if parameter.symbol_kind != "parameter" or parameter.symbol_name is None:
                raise ValueError(
                    f"Projection target component {index} is not a direct parameter symbol."
                )
            variable_names.append(variable.symbol_name)
            target_names.append(parameter.symbol_name)
        if tuple(variable_names) != tuple(self.variable_names):
            raise ValueError(
                "Projection variables must follow the declaration order "
                f"{self.variable_names}; received {variable_names}."
            )

        components = [
            variable - parameter
            for variable, parameter in zip(variable_vector, target_vector, strict=True)
        ]
        expression = 0.5 * sum(
            (component**2 for component in components),
            _as_expression(0.0),
        )
        expression.projection_target_names = tuple(target_names)
        return expression

    def set_projection_objective(
        self,
        *,
        variables: VectorExpression | _SymbolNamespace,
        target: VectorExpression | _SymbolNamespace,
    ) -> "Problem":
        """Assign the shared identity-Hessian distance objective."""
        self.objective = self.squared_distance(variables, target)
        return self

    def add_parameter(self, names: str | Iterable[str]) -> _SymbolNamespace:
        for name in _coerce_names(names):
            self._register_symbol(name, "parameter")
        return self.parameter

    def add_variable(self, names: str | Iterable[str]) -> _SymbolNamespace:
        for name in _coerce_names(names):
            self._register_symbol(name, "variable")
        return self.variable

    def _register_symbol(self, name: str, kind: str) -> None:
        if name in self.parameter_names or name in self.variable_names:
            raise ValueError(f"Duplicate symbol name {name!r}.")
        if kind == "parameter":
            self.parameter_names.append(name)
        elif kind == "variable":
            self.variable_names.append(name)
        else:
            raise ValueError(kind)

    def _check_symbol(self, kind: str, name: str) -> None:
        names = self.parameter_names if kind == "parameter" else self.variable_names
        if name not in names:
            raise AttributeError(f"Unknown {kind} symbol {name!r}.")

    # Common nonlinear expression helpers.
    def _unary(self, expression: Any, function: Callable, name: str) -> Expression:
        symbolic = _as_expression(expression)
        return Expression(
            lambda context, symbolic=symbolic: function(symbolic.evaluate(context)),
            f"{name}({symbolic.text})",
            variable_dependencies=symbolic.variable_dependencies,
            parameter_dependencies=symbolic.parameter_dependencies,
            affine_in_variables=not symbolic.variable_dependencies,
        )

    def sin(self, expression: Any) -> Expression:
        return self._unary(expression, jnp.sin, "sin")

    def cos(self, expression: Any) -> Expression:
        return self._unary(expression, jnp.cos, "cos")

    def exp(self, expression: Any) -> Expression:
        return self._unary(expression, jnp.exp, "exp")

    def log(self, expression: Any) -> Expression:
        return self._unary(expression, jnp.log, "log")

    def sqrt(self, expression: Any) -> Expression:
        return self._unary(expression, jnp.sqrt, "sqrt")

    def abs(self, expression: Any) -> Expression:
        return self._unary(expression, jnp.abs, "abs")

    def matrix(self, values: ArrayLike) -> np.ndarray:
        """Return a validated constant matrix for use in symbolic helpers."""
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(f"matrix(...) expects a two-dimensional array; got {array.shape}.")
        return array

    def vector(self, values: ArrayLike) -> np.ndarray:
        """Return a validated constant vector for comparisons or linear maps."""
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1:
            raise ValueError(f"vector(...) expects a one-dimensional array; got {array.shape}.")
        return array

    def tensor(self, values: ArrayLike) -> np.ndarray:
        """Return a validated rank-three constant tensor."""
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 3:
            raise ValueError(f"tensor(...) expects a three-dimensional array; got {array.shape}.")
        return array

    def lin(self, matrix: ArrayLike, vector: VectorExpression | _SymbolNamespace) -> VectorExpression:
        symbolic_vector = vector.vector() if isinstance(vector, _SymbolNamespace) else vector
        if not isinstance(symbolic_vector, VectorExpression):
            raise TypeError("lin(matrix, vector) expects a symbolic vector or namespace.")
        coefficient = np.asarray(matrix, dtype=np.float64)
        if coefficient.ndim == 1:
            coefficient = coefficient.reshape(1, -1)
        if coefficient.ndim != 2 or coefficient.shape[1] != symbolic_vector.size:
            raise ValueError(
                "Coefficient matrix shape must be (rows, vector_size); received "
                f"{coefficient.shape} for vector size {symbolic_vector.size}."
            )
        rows: list[Expression] = []
        for row in coefficient:
            rows.append(
                sum(
                    (float(value) * component for value, component in zip(row, symbolic_vector, strict=True)),
                    _as_expression(0.0),
                )
            )
        return VectorExpression(rows, "A @ vector")

    def quad(self, matrix: ArrayLike, vector: VectorExpression | _SymbolNamespace) -> Expression:
        symbolic_vector = vector.vector() if isinstance(vector, _SymbolNamespace) else vector
        if not isinstance(symbolic_vector, VectorExpression):
            raise TypeError("quad(matrix, vector) expects a symbolic vector or namespace.")
        coefficient = np.asarray(matrix, dtype=np.float64)
        expected = (symbolic_vector.size, symbolic_vector.size)
        if coefficient.shape != expected:
            raise ValueError(f"Quadratic matrix must have shape {expected}; got {coefficient.shape}.")
        result = _as_expression(0.0)
        for i, left in enumerate(symbolic_vector):
            for j, right in enumerate(symbolic_vector):
                value = coefficient[i, j]
                if value != 0.0:
                    result = result + float(value) * left * right
        return result

    def _build_projection_structure(self) -> ProjectionStructure | None:
        if self._projection_target_names is None:
            return None
        if len(self._projection_target_names) != len(self.variable_names):
            raise ValueError(
                "Projection target dimension must match the decision-variable dimension."
            )
        parameter_index = {name: index for index, name in enumerate(self.parameter_names)}
        try:
            target_indices = tuple(
                parameter_index[name] for name in self._projection_target_names
            )
        except KeyError as error:
            raise ValueError(
                f"Unknown projection target parameter {error.args[0]!r}."
            ) from error

        variable_index = {name: index for index, name in enumerate(self.variable_names)}
        equalities = [constraint for constraint in self.constraints if constraint.kind == "eq"]
        inequalities = [constraint for constraint in self.constraints if constraint.kind == "ineq"]

        def dependency_indices(constraint: Constraint) -> tuple[int, ...]:
            return tuple(
                sorted(variable_index[name] for name in constraint.residual.variable_dependencies)
            )

        equality_dependencies = tuple(dependency_indices(item) for item in equalities)
        inequality_dependencies = tuple(dependency_indices(item) for item in inequalities)
        affine = all(
            constraint.residual.affine_in_variables for constraint in self.constraints
        )

        n_x = len(self.variable_names)
        n_c = len(equalities)
        n_d = len(inequalities)
        n_s = n_d
        reduced_dimension = n_x + n_s + n_c + n_d
        condensed_dimension = n_x + n_c + n_d
        jacobian_nonzeros = sum(map(len, equality_dependencies)) + sum(
            map(len, inequality_dependencies)
        )
        nonlinear_hessian_nonzeros = sum(
            len(dependencies) ** 2
            for constraint, dependencies in zip(
                [*equalities, *inequalities],
                [*equality_dependencies, *inequality_dependencies],
                strict=True,
            )
            if not constraint.residual.affine_in_variables
        )
        approximate_nonzeros = (
            n_x
            + nonlinear_hessian_nonzeros
            + 2 * jacobian_nonzeros
            + n_s
            + 2 * n_d
            + n_c
            + n_d
        )
        density = (
            min(1.0, float(approximate_nonzeros) / float(reduced_dimension**2))
            if reduced_dimension
            else 0.0
        )
        condensed_nonzeros = (
            n_x
            + nonlinear_hessian_nonzeros
            + 2 * jacobian_nonzeros
            + n_c
            + n_d
        )
        condensed_density = (
            min(1.0, float(condensed_nonzeros) / float(condensed_dimension**2))
            if condensed_dimension
            else 0.0
        )

        max_row_nonzeros = max(
            [1, *(len(item) for item in equality_dependencies), *(len(item) for item in inequality_dependencies)]
        )

        def padded_support(
            dependencies: tuple[tuple[int, ...], ...],
        ) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[bool, ...], ...]]:
            support_rows = []
            mask_rows = []
            for dependency in dependencies:
                padding = max_row_nonzeros - len(dependency)
                support_rows.append(tuple(dependency) + (0,) * padding)
                mask_rows.append((True,) * len(dependency) + (False,) * padding)
            return tuple(support_rows), tuple(mask_rows)

        equality_support, equality_support_mask = padded_support(equality_dependencies)
        inequality_support, inequality_support_mask = padded_support(inequality_dependencies)
        equality_rows = tuple(
            row
            for row, dependencies in enumerate(equality_dependencies)
            for _ in dependencies
        )
        equality_cols = tuple(
            column
            for dependencies in equality_dependencies
            for column in dependencies
        )
        inequality_rows = tuple(
            row
            for row, dependencies in enumerate(inequality_dependencies)
            for _ in dependencies
        )
        inequality_cols = tuple(
            column
            for dependencies in inequality_dependencies
            for column in dependencies
        )
        return ProjectionStructure(
            target_parameter_indices=target_indices,
            equality_variable_indices=equality_dependencies,
            inequality_variable_indices=inequality_dependencies,
            equality_jacobian_rows=equality_rows,
            equality_jacobian_cols=equality_cols,
            inequality_jacobian_rows=inequality_rows,
            inequality_jacobian_cols=inequality_cols,
            max_row_nonzeros=max_row_nonzeros,
            equality_support=equality_support,
            equality_support_mask=equality_support_mask,
            inequality_support=inequality_support,
            inequality_support_mask=inequality_support_mask,
            constraints_affine_in_variables=affine,
            estimated_reduced_kkt_density=density,
            estimated_condensed_kkt_density=condensed_density,
        )

    def _build_sparse_projection_derivatives(
        self,
        structure: ProjectionStructure,
    ) -> tuple[Callable, Callable, Callable]:
        """Compile nonzero Jacobian values and local constraint HVPs."""
        variable_index = {name: index for index, name in enumerate(self.variable_names)}
        parameter_index = {name: index for index, name in enumerate(self.parameter_names)}
        equality_constraints = [
            constraint for constraint in self.constraints if constraint.kind == "eq"
        ]
        inequality_constraints = [
            constraint for constraint in self.constraints if constraint.kind == "ineq"
        ]

        def make_context(variables, parameters):
            return _EvaluationContext(
                variables=variables,
                parameters=parameters,
                variable_index=variable_index,
                parameter_index=parameter_index,
            )

        def sparse_values(constraints, dependencies, variables, parameters):
            blocks = []
            for constraint, dependency_tuple in zip(
                constraints, dependencies, strict=True
            ):
                if not dependency_tuple:
                    continue
                indices = jnp.asarray(dependency_tuple, dtype=jnp.int32)
                local_variables = variables[indices]

                def local_residual(local_values):
                    expanded = variables.at[indices].set(local_values)
                    return constraint.residual.evaluate(
                        make_context(expanded, parameters)
                    )

                blocks.append(jax.grad(local_residual)(local_variables))
            if not blocks:
                return jnp.empty((0,), dtype=variables.dtype)
            return jnp.concatenate(blocks)

        def equality_sparse_values(variables, parameters):
            return sparse_values(
                equality_constraints,
                structure.equality_variable_indices,
                variables,
                parameters,
            )

        def inequality_sparse_values(variables, parameters):
            return sparse_values(
                inequality_constraints,
                structure.inequality_variable_indices,
                variables,
                parameters,
            )

        all_constraints = [*equality_constraints, *inequality_constraints]
        all_dependencies = [
            *structure.equality_variable_indices,
            *structure.inequality_variable_indices,
        ]

        def constraint_hvp(
            variables,
            parameters,
            equality_multipliers,
            inequality_multipliers,
            vector,
        ):
            multipliers = jnp.concatenate(
                (equality_multipliers, inequality_multipliers)
            )
            result = jnp.zeros_like(variables)
            for row, (constraint, dependency_tuple) in enumerate(
                zip(all_constraints, all_dependencies, strict=True)
            ):
                if constraint.residual.affine_in_variables or not dependency_tuple:
                    continue
                indices = jnp.asarray(dependency_tuple, dtype=jnp.int32)
                local_variables = variables[indices]
                local_vector = vector[indices]

                def local_residual(local_values):
                    expanded = variables.at[indices].set(local_values)
                    return constraint.residual.evaluate(
                        make_context(expanded, parameters)
                    )

                local_hvp = jax.jvp(
                    jax.grad(local_residual),
                    (local_variables,),
                    (local_vector,),
                )[1]
                result = result.at[indices].add(multipliers[row] * local_hvp)
            return result

        return equality_sparse_values, inequality_sparse_values, constraint_hvp

    def _build_local_projection_derivatives(
        self,
        structure: ProjectionStructure,
    ) -> tuple[
        Callable,
        Callable,
        Callable | None,
        Callable | None,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        """Compile fixed-width local Jacobian and Hessian blocks.

        Affine groups use the symbolic affine decomposition and therefore do
        not differentiate a dense Jacobian.  Nonlinear rows are differentiated
        only on their local supports, so their Hessian blocks never have shape
        ``(n, n)``.
        """

        variable_index = {name: index for index, name in enumerate(self.variable_names)}
        parameter_index = {name: index for index, name in enumerate(self.parameter_names)}
        equality_constraints = [
            constraint for constraint in self.constraints if constraint.kind == "eq"
        ]
        inequality_constraints = [
            constraint for constraint in self.constraints if constraint.kind == "ineq"
        ]
        width = int(structure.max_row_nonzeros)

        def make_context(variables, parameters):
            return _EvaluationContext(
                variables=variables,
                parameters=parameters,
                variable_index=variable_index,
                parameter_index=parameter_index,
            )

        def build_function(constraints, dependencies, padded_support, padded_mask):
            count = len(constraints)
            all_affine = all(
                constraint.residual.affine_in_variables for constraint in constraints
            )
            support_array = jnp.asarray(padded_support, dtype=jnp.int32)
            mask_array = jnp.asarray(padded_mask, dtype=bool)

            if all_affine:
                expressions = [constraint.residual for constraint in constraints]

                metadata_available = all(
                    expression.affine_coefficients is not None
                    and expression.affine_offset is not None
                    for expression in expressions
                )

                if metadata_available:
                    def affine_local_derivatives(variables, parameters):
                        if count == 0:
                            return (
                                jnp.empty((0, width), dtype=variables.dtype),
                                jnp.empty((0, width, width), dtype=variables.dtype),
                            )
                        context = make_context(variables, parameters)
                        rows = []
                        for expression, dependency_tuple in zip(
                            expressions, dependencies, strict=True
                        ):
                            row = [
                                expression.affine_coefficients[
                                    self.variable_names[index]
                                ].evaluate(context)
                                for index in dependency_tuple
                            ]
                            if row:
                                local = jnp.stack(row)
                            else:
                                local = jnp.empty((0,), dtype=variables.dtype)
                            rows.append(
                                jnp.pad(local, (0, width - len(dependency_tuple)))
                            )
                        local = jnp.stack(rows)
                        hessian = jnp.zeros(
                            (count, width, width), dtype=variables.dtype
                        )
                        return local, hessian

                    def affine_offsets(variables, parameters):
                        if count == 0:
                            return jnp.empty((0,), dtype=variables.dtype)
                        context = make_context(variables, parameters)
                        return jnp.stack(
                            [
                                expression.affine_offset.evaluate(context)
                                for expression in expressions
                            ]
                        )

                    coefficients_are_constant = all(
                        not expression.affine_coefficients[
                            self.variable_names[index]
                        ].parameter_dependencies
                        for expression, dependency_tuple in zip(
                            expressions, dependencies, strict=True
                        )
                        for index in dependency_tuple
                    )
                    constant_local = None
                    if coefficients_are_constant:
                        zero_context = make_context(
                            jnp.zeros((len(self.variable_names),), dtype=jnp.float64),
                            jnp.zeros((len(self.parameter_names),), dtype=jnp.float64),
                        )
                        constant_local = np.zeros((count, width), dtype=np.float64)
                        for row_index, (expression, dependency_tuple) in enumerate(
                            zip(expressions, dependencies, strict=True)
                        ):
                            for local_index, variable_position in enumerate(
                                dependency_tuple
                            ):
                                constant_local[row_index, local_index] = float(
                                    np.asarray(
                                        expression.affine_coefficients[
                                            self.variable_names[variable_position]
                                        ].evaluate(zero_context)
                                    )
                                )

                    return affine_local_derivatives, affine_offsets, constant_local

                def vector_function(variables, parameters):
                    if not expressions:
                        return jnp.empty((0,), dtype=variables.dtype)
                    context = make_context(variables, parameters)
                    return jnp.stack(
                        [expression.evaluate(context) for expression in expressions]
                    )

                full_jacobian = jax.jacfwd(vector_function, argnums=0)

                def affine_local_derivatives(variables, parameters):
                    if count == 0:
                        return (
                            jnp.empty((0, width), dtype=variables.dtype),
                            jnp.empty((0, width, width), dtype=variables.dtype),
                        )
                    dense = full_jacobian(variables, parameters)
                    rows = jnp.arange(count, dtype=jnp.int32)[:, None]
                    local = dense[rows, support_array]
                    local = jnp.where(mask_array, local, 0.0)
                    hessian = jnp.zeros(
                        (count, width, width), dtype=variables.dtype
                    )
                    return local, hessian

                return affine_local_derivatives, None, None

            def nonlinear_local_derivatives(variables, parameters):
                jacobian_rows = []
                hessian_rows = []
                for constraint, dependency_tuple in zip(
                    constraints, dependencies, strict=True
                ):
                    if not dependency_tuple:
                        jacobian_rows.append(jnp.zeros((width,), dtype=variables.dtype))
                        hessian_rows.append(
                            jnp.zeros((width, width), dtype=variables.dtype)
                        )
                        continue
                    indices = jnp.asarray(dependency_tuple, dtype=jnp.int32)
                    local_variables = variables[indices]

                    def local_residual(local_values):
                        expanded = variables.at[indices].set(local_values)
                        return constraint.residual.evaluate(
                            make_context(expanded, parameters)
                        )

                    local_gradient = jax.grad(local_residual)(local_variables)
                    if constraint.residual.affine_in_variables:
                        local_hessian = jnp.zeros(
                            (len(dependency_tuple), len(dependency_tuple)),
                            dtype=variables.dtype,
                        )
                    else:
                        local_hessian = jax.hessian(local_residual)(local_variables)
                    padding = width - len(dependency_tuple)
                    jacobian_rows.append(jnp.pad(local_gradient, (0, padding)))
                    hessian_rows.append(
                        jnp.pad(local_hessian, ((0, padding), (0, padding)))
                    )
                if not jacobian_rows:
                    return (
                        jnp.empty((0, width), dtype=variables.dtype),
                        jnp.empty((0, width, width), dtype=variables.dtype),
                    )
                return jnp.stack(jacobian_rows), jnp.stack(hessian_rows)

            return nonlinear_local_derivatives, None, None

        equality_derivatives, equality_offsets, equality_constant = build_function(
            equality_constraints,
            structure.equality_variable_indices,
            structure.equality_support,
            structure.equality_support_mask,
        )
        inequality_derivatives, inequality_offsets, inequality_constant = build_function(
            inequality_constraints,
            structure.inequality_variable_indices,
            structure.inequality_support,
            structure.inequality_support_mask,
        )
        return (
            equality_derivatives,
            inequality_derivatives,
            equality_offsets,
            inequality_offsets,
            equality_constant,
            inequality_constant,
        )

    def _build_functions(self) -> tuple[ScalarFunction, VectorFunction, VectorFunction]:
        if not self.variable_names:
            raise ValueError("At least one decision variable must be declared.")
        if self._objective is None:
            raise ValueError("An objective must be assigned before compiling or optimizing.")

        variable_index = {name: index for index, name in enumerate(self.variable_names)}
        parameter_index = {name: index for index, name in enumerate(self.parameter_names)}
        equality_expressions = [
            constraint.residual for constraint in self.constraints if constraint.kind == "eq"
        ]
        inequality_expressions = [
            constraint.residual for constraint in self.constraints if constraint.kind == "ineq"
        ]
        objective_expression = self._objective
        projection_target_indices = (
            None
            if self._projection_target_names is None
            else jnp.asarray(
                [parameter_index[name] for name in self._projection_target_names],
                dtype=jnp.int32,
            )
        )

        def context(variables, parameters):
            return _EvaluationContext(
                variables=variables,
                parameters=parameters,
                variable_index=variable_index,
                parameter_index=parameter_index,
            )

        def objective_function(variables, parameters):
            if projection_target_indices is not None:
                difference = variables - parameters[projection_target_indices]
                return 0.5 * jnp.vdot(difference, difference)
            return jnp.asarray(objective_expression.evaluate(context(variables, parameters)))

        def evaluate_vector(expressions: Sequence[Expression], variables, parameters):
            if not expressions:
                return jnp.empty((0,), dtype=variables.dtype)
            current_context = context(variables, parameters)
            return jnp.stack([expression.evaluate(current_context) for expression in expressions])

        def equality_function(variables, parameters):
            return evaluate_vector(equality_expressions, variables, parameters)

        def inequality_function(variables, parameters):
            return evaluate_vector(inequality_expressions, variables, parameters)

        return objective_function, equality_function, inequality_function

    def _bind_parameters(self, values: Mapping[str, float] | Sequence[float] | None) -> np.ndarray:
        count = len(self.parameter_names)
        if values is None:
            if count:
                raise ValueError(
                    f"Parameter values are required for {self.parameter_names}."
                )
            return np.empty((0,), dtype=np.float64)
        if isinstance(values, Mapping):
            missing = [name for name in self.parameter_names if name not in values]
            extra = [name for name in values if name not in self.parameter_names]
            if missing or extra:
                raise ValueError(
                    f"Parameter binding mismatch; missing={missing}, extra={extra}."
                )
            return np.asarray([values[name] for name in self.parameter_names], dtype=np.float64)
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if vector.size != count:
            raise ValueError(f"Expected {count} parameter values; received {vector.size}.")
        return vector

    def _bind_initial_values(
        self,
        values: Mapping[str, float] | Sequence[float] | None,
    ) -> np.ndarray:
        count = len(self.variable_names)
        if values is None:
            return np.zeros((count,), dtype=np.float64)
        if isinstance(values, Mapping):
            unknown = [name for name in values if name not in self.variable_names]
            if unknown:
                raise ValueError(f"Unknown initial-value names: {unknown}.")
            return np.asarray(
                [values.get(name, 0.0) for name in self.variable_names],
                dtype=np.float64,
            )
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if vector.size != count:
            raise ValueError(f"Expected {count} initial values; received {vector.size}.")
        return vector

    def _bind_parameter_batch(
        self,
        values: Any,
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Normalize parameter batches to ``(batch, n_parameter)``."""
        count = len(self.parameter_names)
        requested = None if batch_size is None else int(batch_size)
        if requested is not None and requested <= 0:
            raise ValueError("batch_size must be positive.")

        if values is None:
            if count:
                raise ValueError(
                    f"Batched parameter values are required for {self.parameter_names}."
                )
            size = requested or 1
            return np.empty((size, 0), dtype=np.float64)

        if isinstance(values, Sequence) and not isinstance(values, (str, bytes, np.ndarray)):
            rows = list(values)
            if rows and isinstance(rows[0], Mapping):
                matrix = np.stack([self._bind_parameters(row) for row in rows], axis=0)
                if requested is not None and matrix.shape[0] != requested:
                    raise ValueError(
                        f"Expected batch_size={requested}; received {matrix.shape[0]} rows."
                    )
                return matrix

        if isinstance(values, Mapping):
            missing = [name for name in self.parameter_names if name not in values]
            extra = [name for name in values if name not in self.parameter_names]
            if missing or extra:
                raise ValueError(
                    f"Parameter binding mismatch; missing={missing}, extra={extra}."
                )
            columns: list[np.ndarray] = []
            inferred_sizes: set[int] = set()
            for name in self.parameter_names:
                column = np.asarray(values[name], dtype=np.float64)
                if column.ndim == 0:
                    columns.append(column.reshape(1))
                elif column.ndim == 1:
                    columns.append(column)
                    if column.size > 1:
                        inferred_sizes.add(int(column.size))
                else:
                    raise ValueError(
                        f"Batched parameter {name!r} must be scalar or one-dimensional; "
                        f"received {column.shape}."
                    )
            if len(inferred_sizes) > 1:
                raise ValueError(
                    "All non-scalar parameter columns must have the same batch length; "
                    f"received {sorted(inferred_sizes)}."
                )
            size = requested or (next(iter(inferred_sizes)) if inferred_sizes else 1)
            broadcasted = []
            for name, column in zip(self.parameter_names, columns, strict=True):
                if column.size == 1:
                    broadcasted.append(np.full((size,), float(column[0]), dtype=np.float64))
                elif column.size == size:
                    broadcasted.append(column.astype(np.float64, copy=False))
                else:
                    raise ValueError(
                        f"Parameter column {name!r} has length {column.size}; expected {size}."
                    )
            if not broadcasted:
                return np.empty((size, 0), dtype=np.float64)
            return np.stack(broadcasted, axis=1)

        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1:
            if count == 1:
                array = array.reshape(-1, 1)
            elif array.size == count:
                array = array.reshape(1, count)
            else:
                raise ValueError(
                    f"Expected shape (batch, {count}); received {array.shape}."
                )
        if array.ndim != 2 or array.shape[1] != count:
            raise ValueError(
                f"Expected parameter batch shape (batch, {count}); received {array.shape}."
            )
        if requested is not None:
            if array.shape[0] == 1 and requested > 1:
                array = np.broadcast_to(array, (requested, count)).copy()
            elif array.shape[0] != requested:
                raise ValueError(
                    f"Expected batch_size={requested}; received {array.shape[0]} rows."
                )
        return array

    def _bind_initial_value_batch(self, values: Any, batch_size: int) -> np.ndarray:
        """Normalize initial decision values to ``(batch, n_variable)``."""
        count = len(self.variable_names)
        if values is None:
            return np.zeros((batch_size, count), dtype=np.float64)

        if isinstance(values, Sequence) and not isinstance(values, (str, bytes, np.ndarray)):
            rows = list(values)
            if rows and isinstance(rows[0], Mapping):
                matrix = np.stack([self._bind_initial_values(row) for row in rows], axis=0)
                if matrix.shape[0] != batch_size:
                    raise ValueError(
                        f"Expected {batch_size} initial-value rows; received {matrix.shape[0]}."
                    )
                return matrix

        if isinstance(values, Mapping):
            unknown = [name for name in values if name not in self.variable_names]
            if unknown:
                raise ValueError(f"Unknown initial-value names: {unknown}.")
            columns = []
            for name in self.variable_names:
                column = np.asarray(values.get(name, 0.0), dtype=np.float64)
                if column.ndim == 0:
                    columns.append(np.full((batch_size,), float(column), dtype=np.float64))
                elif column.ndim == 1 and column.size == batch_size:
                    columns.append(column)
                elif column.ndim == 1 and column.size == 1:
                    columns.append(np.full((batch_size,), float(column[0]), dtype=np.float64))
                else:
                    raise ValueError(
                        f"Initial-value column {name!r} must be scalar or length {batch_size}; "
                        f"received {column.shape}."
                    )
            return np.stack(columns, axis=1)

        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1:
            if count == 1 and array.size == batch_size:
                array = array.reshape(batch_size, 1)
            elif array.size == count:
                array = np.broadcast_to(array.reshape(1, count), (batch_size, count)).copy()
            else:
                raise ValueError(
                    f"Expected one initial vector of length {count} or shape "
                    f"({batch_size}, {count}); received {array.shape}."
                )
        if array.ndim != 2 or array.shape != (batch_size, count):
            raise ValueError(
                f"Expected initial-value batch shape ({batch_size}, {count}); "
                f"received {array.shape}."
            )
        return array

    def compile(
        self,
        *,
        parameters: Mapping[str, float] | Sequence[float] | None = None,
        initial_values: Mapping[str, float] | Sequence[float] | None = None,
        interior_margin: float = 1.0,
    ) -> CompiledProblem:
        """Compile symbolic expressions into the existing solver inputs."""
        parameter_values = self._bind_parameters(parameters)
        variable_values = self._bind_initial_values(initial_values)
        margin = float(interior_margin)
        if not np.isfinite(margin) or margin <= 0.0:
            raise ValueError(f"interior_margin must be finite and positive; got {margin}.")

        objective_function, equality_function, inequality_function = self._build_functions()
        projection_structure = self._build_projection_structure()
        if projection_structure is not None and initial_values is None:
            variable_values = parameter_values[
                np.asarray(projection_structure.target_parameter_indices, dtype=np.int64)
            ].copy()
        equality_jacobian = jax.jacobian(equality_function, argnums=0)
        inequality_jacobian = jax.jacobian(inequality_function, argnums=0)

        if projection_structure is None:
            objective_gradient = jax.grad(objective_function, argnums=0)

            def lagrangian(variables, parameters_, equality_multipliers, inequality_multipliers):
                return (
                    objective_function(variables, parameters_)
                    + jnp.vdot(equality_multipliers, equality_function(variables, parameters_))
                    + jnp.vdot(inequality_multipliers, inequality_function(variables, parameters_))
                )

            lagrangian_hessian = jax.hessian(lagrangian, argnums=0)
        else:
            target_indices = jnp.asarray(
                projection_structure.target_parameter_indices, dtype=jnp.int32
            )

            def objective_gradient(variables, parameters_):
                return variables - parameters_[target_indices]

            def constraint_lagrangian(
                variables, parameters_, equality_multipliers, inequality_multipliers
            ):
                return (
                    jnp.vdot(equality_multipliers, equality_function(variables, parameters_))
                    + jnp.vdot(inequality_multipliers, inequality_function(variables, parameters_))
                )

            constraint_hessian = jax.hessian(constraint_lagrangian, argnums=0)

            def lagrangian_hessian(
                variables, parameters_, equality_multipliers, inequality_multipliers
            ):
                return jnp.eye(variables.shape[0], dtype=variables.dtype) + constraint_hessian(
                    variables,
                    parameters_,
                    equality_multipliers,
                    inequality_multipliers,
                )

        parameter_jax = jnp.asarray(parameter_values, dtype=jnp.float64)

        def evaluate_nlp(state: PrimalDualState) -> NLPQuantities:
            variables = jnp.asarray(state.x, dtype=jnp.float64)
            equality_multipliers = jnp.asarray(state.y_c, dtype=jnp.float64)
            inequality_multipliers = jnp.asarray(state.y_d, dtype=jnp.float64)
            return NLPQuantities(
                grad_f=np.asarray(
                    objective_gradient(variables, parameter_jax), dtype=np.float64
                ),
                c=np.asarray(equality_function(variables, parameter_jax), dtype=np.float64),
                d=np.asarray(inequality_function(variables, parameter_jax), dtype=np.float64),
                jac_c=np.asarray(
                    equality_jacobian(variables, parameter_jax), dtype=np.float64
                ).reshape((state.y_c.size, state.x.size)),
                jac_d=np.asarray(
                    inequality_jacobian(variables, parameter_jax), dtype=np.float64
                ).reshape((state.y_d.size, state.x.size)),
                hessian_lagrangian=np.asarray(
                    lagrangian_hessian(
                        variables,
                        parameter_jax,
                        equality_multipliers,
                        inequality_multipliers,
                    ),
                    dtype=np.float64,
                ),
                objective_value=float(
                    np.asarray(objective_function(variables, parameter_jax))
                ),
            )

        initial_jax = jnp.asarray(variable_values, dtype=jnp.float64)
        equality_initial = np.asarray(
            equality_function(initial_jax, parameter_jax), dtype=np.float64
        )
        inequality_initial = np.asarray(
            inequality_function(initial_jax, parameter_jax), dtype=np.float64
        )
        n_equalities = equality_initial.size
        n_inequalities = inequality_initial.size

        # Each normalized inequality d(y,p) <= 0 becomes d(y,p) - s = 0 with
        # an upper bound s <= 0.  The initial s is strictly interior even when
        # the initial decision vector is infeasible.
        slack_initial = np.minimum(inequality_initial, -margin)
        empty_float = np.empty((0,), dtype=np.float64)
        empty_index = np.empty((0,), dtype=np.int64)
        indices = BoundIndices(
            x_lower=empty_index.copy(),
            x_upper=empty_index.copy(),
            s_lower=empty_index.copy(),
            s_upper=np.arange(n_inequalities, dtype=np.int64),
        )
        bounds = BoundValues(
            x_lower=empty_float.copy(),
            x_upper=empty_float.copy(),
            s_lower=empty_float.copy(),
            s_upper=np.zeros((n_inequalities,), dtype=np.float64),
        )
        initial_state = PrimalDualState(
            x=variable_values,
            s=slack_initial,
            y_c=np.zeros((n_equalities,), dtype=np.float64),
            y_d=np.ones((n_inequalities,), dtype=np.float64),
            z_L=empty_float.copy(),
            z_U=empty_float.copy(),
            v_L=empty_float.copy(),
            v_U=np.ones((n_inequalities,), dtype=np.float64),
        )

        return CompiledProblem(
            name=self.name,
            variable_names=tuple(self.variable_names),
            parameter_names=tuple(self.parameter_names),
            parameter_values=parameter_values,
            initial_state=initial_state,
            indices=indices,
            bounds=bounds,
            evaluate_nlp=evaluate_nlp,
            objective_function=objective_function,
            equality_function=equality_function,
            inequality_function=inequality_function,
        )

    def compile_batch(
        self,
        *,
        parameters: Any,
        initial_values: Any = None,
        batch_size: int | None = None,
        interior_margin: float = 1.0,
        device: Any | None = None,
    ) -> CompiledBatchProblem:
        """Compile a same-structure parameter batch for vectorized JAX solves."""
        selected_device = _resolve_jax_device(device)
        parameter_values = self._bind_parameter_batch(parameters, batch_size=batch_size)
        batch_count = int(parameter_values.shape[0])
        variable_values = self._bind_initial_value_batch(initial_values, batch_count)
        margin = float(interior_margin)
        if not np.isfinite(margin) or margin <= 0.0:
            raise ValueError(f"interior_margin must be finite and positive; got {margin}.")

        model_signature = (
            tuple(self.parameter_names),
            tuple(self.variable_names),
            None if self._objective is None else self._objective.text,
            tuple((item.kind, item.text) for item in self.constraints),
        )
        device_key = (
            (selected_device.platform, selected_device.id)
            if selected_device is not None
            else ("default", -1)
        )
        structure_cache_key = (model_signature, batch_count, device_key)
        cached = self._compiled_batch_structure_cache.get(structure_cache_key)
        if cached is not None:
            projection_structure = cached.projection_structure
            if projection_structure is not None and initial_values is None:
                variable_values = parameter_values[
                    :,
                    np.asarray(
                        projection_structure.target_parameter_indices,
                        dtype=np.int64,
                    ),
                ].copy()
            variable_jax = jnp.asarray(variable_values, dtype=jnp.float64)
            parameter_jax = jnp.asarray(parameter_values, dtype=jnp.float64)
            margin_jax = jnp.asarray(margin, dtype=jnp.float64)
            if selected_device is not None:
                variable_jax = jax.device_put(variable_jax, selected_device)
                parameter_jax = jax.device_put(parameter_jax, selected_device)
                margin_jax = jax.device_put(margin_jax, selected_device)
            initial_state = cached.initialize_state_function(
                variable_jax,
                parameter_jax,
                margin_jax,
            )
            self._compiled_batch_structure_cache_hits += 1
            return replace(
                cached,
                parameter_values=parameter_values,
                initial_state=initial_state,
            )

        self._compiled_batch_structure_cache_misses += 1
        objective_function, equality_function, inequality_function = self._build_functions()
        projection_structure = self._build_projection_structure()
        if projection_structure is not None and initial_values is None:
            variable_values = parameter_values[
                :, np.asarray(projection_structure.target_parameter_indices, dtype=np.int64)
            ].copy()
        if projection_structure is None:
            equality_sparse_values = None
            inequality_sparse_values = None
            constraint_hvp = None
            equality_local_derivatives = None
            inequality_local_derivatives = None
            equality_affine_offsets = None
            inequality_affine_offsets = None
            equality_constant_local_jacobian = None
            inequality_constant_local_jacobian = None
        else:
            (
                equality_sparse_values,
                inequality_sparse_values,
                constraint_hvp,
            ) = self._build_sparse_projection_derivatives(projection_structure)
            (
                equality_local_derivatives,
                inequality_local_derivatives,
                equality_affine_offsets,
                inequality_affine_offsets,
                equality_constant_local_jacobian,
                inequality_constant_local_jacobian,
            ) = self._build_local_projection_derivatives(projection_structure)
        equality_jacobian = jax.jacobian(equality_function, argnums=0)
        inequality_jacobian = jax.jacobian(inequality_function, argnums=0)

        if projection_structure is None:
            objective_gradient = jax.grad(objective_function, argnums=0)

            def lagrangian(variables, parameters_, equality_multipliers, inequality_multipliers):
                return (
                    objective_function(variables, parameters_)
                    + jnp.vdot(equality_multipliers, equality_function(variables, parameters_))
                    + jnp.vdot(inequality_multipliers, inequality_function(variables, parameters_))
                )

            lagrangian_hessian = jax.hessian(lagrangian, argnums=0)
        else:
            target_indices = jnp.asarray(
                projection_structure.target_parameter_indices, dtype=jnp.int32
            )

            def objective_gradient(variables, parameters_):
                return variables - parameters_[target_indices]

            def constraint_lagrangian(
                variables, parameters_, equality_multipliers, inequality_multipliers
            ):
                return (
                    jnp.vdot(equality_multipliers, equality_function(variables, parameters_))
                    + jnp.vdot(inequality_multipliers, inequality_function(variables, parameters_))
                )

            constraint_hessian = jax.hessian(constraint_lagrangian, argnums=0)

            def lagrangian_hessian(
                variables, parameters_, equality_multipliers, inequality_multipliers
            ):
                return jnp.eye(variables.shape[0], dtype=variables.dtype) + constraint_hessian(
                    variables,
                    parameters_,
                    equality_multipliers,
                    inequality_multipliers,
                )

        def evaluate_nlp_single(
            state: JaxPrimalDualState,
            parameter_vector: jax.Array,
        ) -> JaxNLPQuantities:
            variables = state.x
            return JaxNLPQuantities(
                grad_f=objective_gradient(variables, parameter_vector),
                c=equality_function(variables, parameter_vector),
                d=inequality_function(variables, parameter_vector),
                jac_c=equality_jacobian(variables, parameter_vector),
                jac_d=inequality_jacobian(variables, parameter_vector),
                hessian_lagrangian=lagrangian_hessian(
                    variables,
                    parameter_vector,
                    state.y_c,
                    state.y_d,
                ),
                objective_value=objective_function(variables, parameter_vector),
            )

        variable_jax = jnp.asarray(variable_values, dtype=jnp.float64)
        parameter_jax = jnp.asarray(parameter_values, dtype=jnp.float64)
        margin_jax = jnp.asarray(margin, dtype=jnp.float64)
        if selected_device is not None:
            variable_jax = jax.device_put(variable_jax, selected_device)
            parameter_jax = jax.device_put(parameter_jax, selected_device)
            margin_jax = jax.device_put(margin_jax, selected_device)
        n_equalities = sum(item.kind == "eq" for item in self.constraints)
        n_inequalities = sum(item.kind == "ineq" for item in self.constraints)

        empty_float = np.empty((0,), dtype=np.float64)
        empty_index = np.empty((0,), dtype=np.int64)
        numpy_indices = BoundIndices(
            x_lower=empty_index.copy(),
            x_upper=empty_index.copy(),
            s_lower=empty_index.copy(),
            s_upper=np.arange(n_inequalities, dtype=np.int64),
        )
        numpy_bounds = BoundValues(
            x_lower=empty_float.copy(),
            x_upper=empty_float.copy(),
            s_lower=empty_float.copy(),
            s_upper=np.zeros((n_inequalities,), dtype=np.float64),
        )

        equality_batch_function = jax.jit(jax.vmap(equality_function))
        inequality_batch_function = jax.jit(jax.vmap(inequality_function))
        objective_batch_function = jax.jit(jax.vmap(objective_function))

        def initialize_state_function(variable_batch, parameter_batch, margin_value):
            inequality_initial = inequality_batch_function(
                variable_batch, parameter_batch
            )
            slack_initial = jnp.minimum(inequality_initial, -margin_value)
            return JaxPrimalDualState(
                x=variable_batch,
                s=slack_initial,
                y_c=jnp.zeros(
                    (variable_batch.shape[0], n_equalities),
                    dtype=variable_batch.dtype,
                ),
                y_d=jnp.ones(
                    (variable_batch.shape[0], n_inequalities),
                    dtype=variable_batch.dtype,
                ),
                z_L=jnp.empty(
                    (variable_batch.shape[0], 0), dtype=variable_batch.dtype
                ),
                z_U=jnp.empty(
                    (variable_batch.shape[0], 0), dtype=variable_batch.dtype
                ),
                v_L=jnp.empty(
                    (variable_batch.shape[0], 0), dtype=variable_batch.dtype
                ),
                v_U=jnp.ones(
                    (variable_batch.shape[0], n_inequalities),
                    dtype=variable_batch.dtype,
                ),
            )

        initialize_state_function = jax.jit(initialize_state_function)
        initial_state = initialize_state_function(
            variable_jax,
            parameter_jax,
            margin_jax,
        )

        compiled_indices = indices_from_numpy(numpy_indices)
        compiled_bounds = bounds_from_numpy(numpy_bounds)
        if selected_device is not None:
            compiled_indices = jax.tree_util.tree_map(
                lambda value: jax.device_put(value, selected_device),
                compiled_indices,
            )
            compiled_bounds = jax.tree_util.tree_map(
                lambda value: jax.device_put(value, selected_device),
                compiled_bounds,
            )

        compiled = CompiledBatchProblem(
            name=self.name,
            variable_names=tuple(self.variable_names),
            parameter_names=tuple(self.parameter_names),
            parameter_values=parameter_values,
            initial_state=initial_state,
            indices=compiled_indices,
            bounds=compiled_bounds,
            evaluate_nlp_single=evaluate_nlp_single,
            objective_function=objective_function,
            equality_function=equality_function,
            inequality_function=inequality_function,
            initialize_state_function=initialize_state_function,
            objective_batch_function=objective_batch_function,
            equality_batch_function=equality_batch_function,
            inequality_batch_function=inequality_batch_function,
            equality_jacobian_function=equality_jacobian,
            inequality_jacobian_function=inequality_jacobian,
            equality_sparse_jacobian_values_function=equality_sparse_values,
            inequality_sparse_jacobian_values_function=inequality_sparse_values,
            constraint_hvp_function=constraint_hvp,
            equality_local_derivatives_function=equality_local_derivatives,
            inequality_local_derivatives_function=inequality_local_derivatives,
            equality_affine_offset_function=equality_affine_offsets,
            inequality_affine_offset_function=inequality_affine_offsets,
            equality_constant_local_jacobian=equality_constant_local_jacobian,
            inequality_constant_local_jacobian=inequality_constant_local_jacobian,
            projection_structure=projection_structure,
        )
        self._compiled_batch_structure_cache[structure_cache_key] = compiled
        return compiled

    def prepare_sparse_projection_batch_solver(
        self,
        *,
        parameters: Any,
        batch_size: int | None = None,
        options: FixedMuOptions | None = None,
        interior_margin: float = 1.0,
        gmres_tolerance: float = 1e-9,
        gmres_atol: float = 0.0,
        gmres_restart: int = 30,
        gmres_maxiter: int = 50,
        use_preconditioner: bool = True,
        backtracking: bool = False,
        max_backtracking_steps: int = 8,
        backtracking_factor: float = 0.5,
        backend: Any | None = None,
        device: Any | None = None,
        transpose_mode: str = "auto",
        gmres_unroll: int | None = None,
        arnoldi_prefix_block: int | None = None,
        gpu_kkt_mode: str = "auto",
        pallas_kkt_block_size: int = 64,
        pallas_interpret: bool = False,
        store_residual_history: bool = True,
        fuse_initialization: bool | None = None,
        adjoint_gmres_tolerance: float | None = None,
        adjoint_gmres_atol: float | None = None,
        adjoint_gmres_restart: int | None = None,
        adjoint_gmres_maxiter: int | None = None,
        adjoint_gmres_unroll: int | None = None,
        adjoint_arnoldi_prefix_block: int | None = None,
        adjoint_solver_mode: str = "auto",
        adjoint_dense_threshold: int = 64,
    ) -> PreparedSparseProjectionBatchSolver:
        """Prepare one fixed-shape sparse projection solver on a device.

        This API is intended for repeated CPU/CUDA batches.  It avoids the
        per-batch high-level compilation/binding path and keeps inputs and
        outputs device-resident until the caller explicitly uses
        :func:`jax.device_get`.
        """

        if backend is not None and device is not None:
            backend_device = _resolve_jax_device(backend)
            explicit_device = _resolve_jax_device(device)
            if (
                backend_device.platform,
                backend_device.id,
            ) != (explicit_device.platform, explicit_device.id):
                raise ValueError("backend and device select different JAX devices.")
            selected_device = backend_device
        else:
            selected_device = _resolve_jax_device(
                backend if backend is not None else device
            )
        if selected_device is None:
            selected_device = jax.devices()[0]

        mode = str(transpose_mode).strip().lower()
        if mode not in {"auto", "scatter", "gather", "bucketed"}:
            raise ValueError(
                "transpose_mode must be one of 'auto', 'scatter', 'gather', "
                "or 'bucketed'."
            )
        requested_gpu_kkt_mode = str(gpu_kkt_mode).strip().lower()
        if requested_gpu_kkt_mode not in {"auto", "xla", "pallas"}:
            raise ValueError(
                "gpu_kkt_mode must be one of 'auto', 'xla', or 'pallas'."
            )
        if gmres_unroll is None:
            gmres_unroll = 4 if selected_device.platform == "gpu" else 1
        if not isinstance(gmres_unroll, int) or gmres_unroll < 1:
            raise ValueError("gmres_unroll must be a positive integer or None.")
        if arnoldi_prefix_block is None:
            arnoldi_prefix_block = 8 if selected_device.platform == "gpu" else 0
        if (
            not isinstance(arnoldi_prefix_block, int)
            or arnoldi_prefix_block < 0
        ):
            raise ValueError(
                "arnoldi_prefix_block must be a nonnegative integer or None."
            )
        if not isinstance(pallas_kkt_block_size, int) or pallas_kkt_block_size < 1:
            raise ValueError("pallas_kkt_block_size must be a positive integer.")
        if fuse_initialization is None:
            fuse_initialization = selected_device.platform == "gpu"

        compiled = self.compile_batch(
            parameters=parameters,
            batch_size=batch_size,
            interior_margin=interior_margin,
            device=selected_device,
        )
        structure = compiled.projection_structure
        if structure is None:
            raise ValueError(
                "prepare_sparse_projection_batch_solver requires an objective "
                "created with set_projection_objective(...)."
            )
        if compiled.equality_local_derivatives_function is None or compiled.inequality_local_derivatives_function is None:
            raise RuntimeError("Sparse local derivative functions were not prepared.")

        device_kind_text = str(getattr(selected_device, "device_kind", "")).lower()
        if mode == "auto":
            if selected_device.platform == "cpu":
                selected_transpose_mode = "scatter"
            else:
                padding_fraction = _transpose_gather_padding_fraction(
                    structure.equality_support,
                    structure.equality_support_mask,
                    structure.inequality_support,
                    structure.inequality_support_mask,
                    len(self.variable_names),
                )
                bucket_padding_fraction = _transpose_bucket_padding_fraction(
                    structure.equality_support,
                    structure.equality_support_mask,
                    structure.inequality_support,
                    structure.inequality_support_mask,
                    len(self.variable_names),
                )
                # T4 retains the measured scatter path.  Newer accelerators may
                # use compact gather/reduce layouts that avoid atomic scatters.
                if "t4" in device_kind_text:
                    selected_transpose_mode = "scatter"
                elif padding_fraction <= 0.25:
                    selected_transpose_mode = "gather"
                elif bucket_padding_fraction <= 0.25:
                    selected_transpose_mode = "bucketed"
                else:
                    selected_transpose_mode = "scatter"
        else:
            selected_transpose_mode = mode
        use_column_gather_transpose = selected_transpose_mode == "gather"
        use_column_bucketed_transpose = selected_transpose_mode == "bucketed"

        pallas_compatible = (
            selected_device.platform == "gpu"
            and structure.constraints_affine_in_variables
            and compiled.equality_constant_local_jacobian is not None
            and compiled.inequality_constant_local_jacobian is not None
        )
        if requested_gpu_kkt_mode == "auto":
            # Keep the proven XLA path as the portable default.  The Pallas
            # kernel is available explicitly for GPU benchmarking and can be
            # promoted after device-specific parity/performance validation.
            selected_gpu_kkt_mode = "xla"
        else:
            selected_gpu_kkt_mode = requested_gpu_kkt_mode
        if selected_gpu_kkt_mode == "pallas" and not pallas_compatible:
            raise ValueError(
                "gpu_kkt_mode='pallas' requires a CUDA device and affine "
                "constraints with a parameter-independent Jacobian."
            )

        opts = options or FixedMuOptions(line_search=None)
        if opts.line_search is not None:
            raise ValueError(
                "Prepared batched sparse solves require FixedMuOptions(line_search=None)."
            )
        if opts.regularization is None:
            regularization = JaxNewtonRegularization()
        else:
            regularization = JaxNewtonRegularization(
                delta_x=opts.regularization.delta_x,
                delta_s=opts.regularization.delta_s,
                delta_c=opts.regularization.delta_c,
                delta_d=opts.regularization.delta_d,
            )

        if adjoint_solver_mode not in {"auto", "gmres", "dense"}:
            raise ValueError(
                "adjoint_solver_mode must be auto, gmres, or dense."
            )
        if not isinstance(adjoint_dense_threshold, int) or adjoint_dense_threshold < 1:
            raise ValueError("adjoint_dense_threshold must be positive.")
        condensed_width = (
            len(self.variable_names)
            + len(structure.equality_support)
            + len(structure.inequality_support)
        )
        if adjoint_solver_mode == "auto":
            selected_adjoint_solver_mode = (
                "dense"
                if selected_device.platform == "cpu"
                and condensed_width <= adjoint_dense_threshold
                and compiled.batch_size <= 32
                else "gmres"
            )
        else:
            selected_adjoint_solver_mode = adjoint_solver_mode
        selected_adjoint_gmres_tolerance = (
            gmres_tolerance
            if adjoint_gmres_tolerance is None
            else float(adjoint_gmres_tolerance)
        )
        selected_adjoint_gmres_atol = (
            gmres_atol if adjoint_gmres_atol is None else float(adjoint_gmres_atol)
        )
        selected_adjoint_gmres_restart = (
            gmres_restart
            if adjoint_gmres_restart is None
            else int(adjoint_gmres_restart)
        )
        selected_adjoint_gmres_maxiter = (
            gmres_maxiter
            if adjoint_gmres_maxiter is None
            else int(adjoint_gmres_maxiter)
        )
        selected_adjoint_gmres_unroll = (
            gmres_unroll
            if adjoint_gmres_unroll is None
            else int(adjoint_gmres_unroll)
        )
        selected_adjoint_prefix = (
            arnoldi_prefix_block
            if adjoint_arnoldi_prefix_block is None
            else int(adjoint_arnoldi_prefix_block)
        )

        indices = jax.tree_util.tree_map(
            lambda value: jax.device_put(value, selected_device), compiled.indices
        )
        bounds = jax.tree_util.tree_map(
            lambda value: jax.device_put(value, selected_device), compiled.bounds
        )
        model_signature = (
            tuple(self.parameter_names),
            tuple(self.variable_names),
            None if self._objective is None else self._objective.text,
            tuple((item.kind, item.text) for item in self.constraints),
        )
        device_key = (selected_device.platform, selected_device.id)
        sparse_cache_key = (
            model_signature,
            int(compiled.batch_size),
            device_key,
            int(opts.max_iterations),
            float(opts.residual_tolerance),
            float(opts.tau),
            float(opts.step_scale),
            float(opts.minimum_step),
            tuple(float(value) for value in regularization),
            float(gmres_tolerance),
            float(gmres_atol),
            int(gmres_restart),
            int(gmres_maxiter),
            bool(use_preconditioner),
            bool(backtracking),
            int(max_backtracking_steps),
            float(backtracking_factor),
            str(selected_transpose_mode),
            bool(selected_device.platform == "gpu"),
            int(gmres_unroll),
            int(arnoldi_prefix_block),
            str(selected_gpu_kkt_mode),
            int(pallas_kkt_block_size),
            bool(pallas_interpret),
            bool(store_residual_history),
            str(selected_adjoint_solver_mode),
            float(selected_adjoint_gmres_tolerance),
            float(selected_adjoint_gmres_atol),
            int(selected_adjoint_gmres_restart),
            int(selected_adjoint_gmres_maxiter),
            int(selected_adjoint_gmres_unroll),
            int(selected_adjoint_prefix),
        )
        sparse_solver = self._prepared_sparse_solver_cache.get(sparse_cache_key)
        if sparse_solver is None:
            with jax.default_device(selected_device):
                sparse_solver = prepare_batched_projection_condensed_sparse_solver(
                    batch_size=int(compiled.batch_size),
                    equality_function=compiled.equality_function,
                    inequality_function=compiled.inequality_function,
                    equality_local_derivatives=compiled.equality_local_derivatives_function,
                    inequality_local_derivatives=compiled.inequality_local_derivatives_function,
                    equality_affine_offsets=compiled.equality_affine_offset_function,
                    inequality_affine_offsets=compiled.inequality_affine_offset_function,
                    equality_constant_local_jacobian=compiled.equality_constant_local_jacobian,
                    inequality_constant_local_jacobian=compiled.inequality_constant_local_jacobian,
                    equality_support=np.asarray(structure.equality_support, dtype=np.int32),
                    equality_mask=np.asarray(structure.equality_support_mask, dtype=bool),
                    inequality_support=np.asarray(structure.inequality_support, dtype=np.int32),
                    inequality_mask=np.asarray(structure.inequality_support_mask, dtype=bool),
                    target_parameter_indices=np.asarray(
                        structure.target_parameter_indices, dtype=np.int32
                    ),
                    indices=indices,
                    bounds=bounds,
                    max_iterations=opts.max_iterations,
                    residual_tolerance=opts.residual_tolerance,
                    tau=opts.tau,
                    step_scale=opts.step_scale,
                    minimum_step=opts.minimum_step,
                    regularization=regularization,
                    gmres_tolerance=gmres_tolerance,
                    gmres_atol=gmres_atol,
                    gmres_restart=gmres_restart,
                    gmres_maxiter=gmres_maxiter,
                    use_preconditioner=use_preconditioner,
                    backtracking=backtracking,
                    max_backtracking_steps=max_backtracking_steps,
                    backtracking_factor=backtracking_factor,
                    constraints_affine=structure.constraints_affine_in_variables,
                    use_column_gather_transpose=use_column_gather_transpose,
                    use_column_bucketed_transpose=use_column_bucketed_transpose,
                    use_shared_constant_jacobian=(
                        selected_device.platform == "gpu"
                    ),
                    gmres_unroll=gmres_unroll,
                    arnoldi_prefix_block=arnoldi_prefix_block,
                    gpu_kkt_mode=selected_gpu_kkt_mode,
                    pallas_kkt_block_size=pallas_kkt_block_size,
                    pallas_interpret=pallas_interpret,
                    store_residual_history=store_residual_history,
                    adjoint_gmres_tolerance=selected_adjoint_gmres_tolerance,
                    adjoint_gmres_atol=selected_adjoint_gmres_atol,
                    adjoint_gmres_restart=selected_adjoint_gmres_restart,
                    adjoint_gmres_maxiter=selected_adjoint_gmres_maxiter,
                    adjoint_gmres_unroll=selected_adjoint_gmres_unroll,
                    adjoint_arnoldi_prefix_block=selected_adjoint_prefix,
                    adjoint_solver_mode=selected_adjoint_solver_mode,
                )
            self._prepared_sparse_solver_cache[sparse_cache_key] = sparse_solver
            self._prepared_sparse_solver_cache_misses += 1
        else:
            self._prepared_sparse_solver_cache_hits += 1

        target_parameter_indices = jax.device_put(
            jnp.asarray(structure.target_parameter_indices, dtype=jnp.int32),
            selected_device,
        )
        target_index_host = np.asarray(
            structure.target_parameter_indices, dtype=np.int32
        )
        physical_index_host = np.asarray(
            [
                index
                for index in range(len(self.parameter_names))
                if index not in set(target_index_host.tolist())
            ],
            dtype=np.int32,
        )
        physical_parameter_indices = jax.device_put(
            jnp.asarray(physical_index_host, dtype=jnp.int32),
            selected_device,
        )
        solve_default_function = None
        solve_default_with_linearization_function = None
        solve_default_loss_gradient_functions = None
        if fuse_initialization:
            margin = jax.device_put(
                jnp.asarray(interior_margin, dtype=jnp.float64), selected_device
            )

            def solve_from_parameters(parameter_array, mu_values):
                variable_array = parameter_array[:, target_parameter_indices]
                initial_state = compiled.initialize_state_function(
                    variable_array, parameter_array, margin
                )
                return sparse_solver._solve_all_function(
                    initial_state, parameter_array, mu_values
                )

            def solve_from_parameters_with_linearization(
                parameter_array, mu_values
            ):
                variable_array = parameter_array[:, target_parameter_indices]
                initial_state = compiled.initialize_state_function(
                    variable_array, parameter_array, margin
                )
                return sparse_solver._solve_all_with_linearization_function(
                    initial_state, parameter_array, mu_values
                )

            def make_default_loss_gradient(q_mode):
                def solve_from_parameters_loss_gradient(
                    parameter_array,
                    mu_values,
                    q_data,
                    valid_mask,
                    initial_adjoint,
                ):
                    variable_array = parameter_array[:, target_parameter_indices]
                    initial_state = compiled.initialize_state_function(
                        variable_array, parameter_array, margin
                    )
                    return sparse_solver.solve_loss_yhat_gradient(
                        initial_state=initial_state,
                        parameter_values=parameter_array,
                        mu_values=mu_values,
                        q_data=q_data,
                        valid_mask=valid_mask,
                        q_mode=q_mode,
                        initial_adjoint=initial_adjoint,
                    )
                return solve_from_parameters_loss_gradient

            with jax.default_device(selected_device):
                solve_default_function = jax.jit(solve_from_parameters)
                solve_default_with_linearization_function = jax.jit(
                    solve_from_parameters_with_linearization
                )
                solve_default_loss_gradient_functions = {
                    "diagonal": jax.jit(make_default_loss_gradient("diagonal")),
                    "dense": jax.jit(make_default_loss_gradient("dense")),
                }

        return PreparedSparseProjectionBatchSolver(
            compiled=compiled,
            sparse_solver=sparse_solver,
            device=selected_device,
            interior_margin=float(interior_margin),
            target_parameter_indices=target_parameter_indices,
            physical_parameter_indices=physical_parameter_indices,
            batch_size=int(compiled.batch_size),
            parameter_count=len(self.parameter_names),
            transpose_mode=selected_transpose_mode,
            gmres_unroll=int(gmres_unroll),
            arnoldi_prefix_block=int(arnoldi_prefix_block),
            gpu_kkt_mode=selected_gpu_kkt_mode,
            pallas_kkt_block_size=int(pallas_kkt_block_size),
            store_residual_history=bool(store_residual_history),
            adjoint_solver_mode=selected_adjoint_solver_mode,
            adjoint_gmres_restart=int(selected_adjoint_gmres_restart),
            adjoint_gmres_maxiter=int(selected_adjoint_gmres_maxiter),
            adjoint_gmres_unroll=int(selected_adjoint_gmres_unroll),
            adjoint_arnoldi_prefix_block=int(selected_adjoint_prefix),
            solve_default_function=solve_default_function,
            solve_default_with_linearization_function=solve_default_with_linearization_function,
            solve_default_loss_gradient_functions=solve_default_loss_gradient_functions,
        )


    def solve_batch(
        self,
        *,
        parameters: Any,
        initial_values: Any = None,
        batch_size: int | None = None,
        mu: float | Sequence[float] = 1e-4,
        options: FixedMuOptions | None = None,
        interior_margin: float = 1.0,
        projection_mode: str = "auto",
        sparse_density_threshold: float = 0.12,
        dense_dimension_threshold: int = 192,
        gmres_tolerance: float = 1e-9,
        gmres_atol: float = 0.0,
        gmres_restart: int = 30,
        gmres_maxiter: int = 50,
        use_preconditioner: bool = True,
        backtracking: bool = False,
        max_backtracking_steps: int = 8,
        backtracking_factor: float = 0.5,
        backend: Any | None = None,
        device: Any | None = None,
        raise_on_failure: bool = False,
        evaluate_outputs: bool = True,
        jit: bool = True,
    ) -> BatchProblemSolution:
        """Solve a parameter batch with a generic or projection-specialized mode.

        ``projection_mode`` may be ``"auto"``, ``"dense_batched"``,
        ``"sparse_iterative"``, ``"shared_linear"``, or ``"generic_dense"``.
        Projection modes require an objective created by
        :meth:`set_projection_objective` or :meth:`squared_distance`.
        """
        if backend is not None and device is not None:
            backend_device = _resolve_jax_device(backend)
            explicit_device = _resolve_jax_device(device)
            if (
                backend_device.platform,
                backend_device.id,
            ) != (explicit_device.platform, explicit_device.id):
                raise ValueError("backend and device select different JAX devices.")
            selected_device = backend_device
        else:
            selected_device = _resolve_jax_device(
                backend if backend is not None else device
            )
        compiled = self.compile_batch(
            parameters=parameters,
            initial_values=initial_values,
            batch_size=batch_size,
            interior_margin=interior_margin,
            device=selected_device,
        )
        opts = options or FixedMuOptions(line_search=None)
        if opts.line_search is not None:
            raise ValueError(
                "solve_batch requires FixedMuOptions(line_search=None); the batched "
                "kernels use vectorized fraction-to-the-boundary updates."
            )
        if opts.regularization is None:
            regularization = JaxNewtonRegularization()
        else:
            regularization = JaxNewtonRegularization(
                delta_x=opts.regularization.delta_x,
                delta_s=opts.regularization.delta_s,
                delta_c=opts.regularization.delta_c,
                delta_d=opts.regularization.delta_d,
            )

        aliases = {
            "dense": "dense_batched",
            "sparse": "sparse_iterative",
            "sparse_condensed": "sparse_iterative",
            "condensed_sparse": "sparse_iterative",
            "shared": "shared_linear",
            "generic": "generic_dense",
        }
        requested_mode = aliases.get(str(projection_mode).lower(), str(projection_mode).lower())
        valid_modes = {
            "auto",
            "dense_batched",
            "sparse_iterative",
            "shared_linear",
            "generic_dense",
        }
        if requested_mode not in valid_modes:
            raise ValueError(
                f"Unknown projection_mode {projection_mode!r}; expected one of {sorted(valid_modes)}."
            )

        structure = compiled.projection_structure
        if structure is None and requested_mode not in {"auto", "generic_dense", "dense_batched"}:
            raise ValueError(
                f"projection_mode={requested_mode!r} requires an identity-Hessian "
                "distance objective created with set_projection_objective(...)."
            )

        initial_state = compiled.initial_state
        parameter_array = jnp.asarray(compiled.parameter_values, dtype=jnp.float64)
        indices = compiled.indices
        bounds = compiled.bounds
        if selected_device is not None:
            initial_state = jax.tree_util.tree_map(
                lambda value: jax.device_put(value, selected_device), initial_state
            )
            parameter_array = jax.device_put(parameter_array, selected_device)
            indices = jax.tree_util.tree_map(
                lambda value: jax.device_put(value, selected_device), indices
            )
            bounds = jax.tree_util.tree_map(
                lambda value: jax.device_put(value, selected_device), bounds
            )

        equality_jacobians = None
        shared_linear_available = False
        if (
            requested_mode in {"auto", "shared_linear"}
            and structure is not None
            and initial_state.y_d.shape[1] == 0
            and structure.constraints_affine_in_variables
        ):
            equality_jacobians = jax.vmap(compiled.equality_jacobian_function)(
                initial_state.x, parameter_array
            )
            jacobian_host = np.asarray(equality_jacobians, dtype=np.float64)
            shared_linear_available = bool(
                jacobian_host.shape[0] == 0
                or np.allclose(
                    jacobian_host,
                    jacobian_host[:1],
                    rtol=1e-12,
                    atol=1e-12,
                )
            )

        if requested_mode == "auto":
            if structure is None:
                selected_mode = "generic_dense"
            elif shared_linear_available:
                selected_mode = "shared_linear"
            else:
                reduced_dimension = (
                    initial_state.x.shape[1]
                    + initial_state.s.shape[1]
                    + initial_state.y_c.shape[1]
                    + initial_state.y_d.shape[1]
                )
                if (
                    reduced_dimension >= int(dense_dimension_threshold)
                    and structure.estimated_condensed_kkt_density
                    <= float(sparse_density_threshold)
                ):
                    selected_mode = "sparse_iterative"
                else:
                    selected_mode = "dense_batched"
        else:
            selected_mode = requested_mode

        if selected_mode == "generic_dense" and structure is not None:
            selected_mode = "dense_batched"
        if selected_mode == "shared_linear" and not shared_linear_available:
            raise ValueError(
                "shared_linear requires an equality-only affine projection problem "
                "whose equality Jacobian is identical for every batch row."
            )
        if selected_mode in {"dense_batched", "sparse_iterative", "shared_linear"} and structure is None:
            if selected_mode == "dense_batched":
                selected_mode = "generic_dense"
            else:
                raise ValueError(
                    f"{selected_mode} requires set_projection_objective(...)."
                )

        if selected_mode == "shared_linear":
            assert equality_jacobians is not None
            result = run_shared_linear_projection_batch(
                initial_state=initial_state,
                parameter_values=parameter_array,
                equality_function=compiled.equality_function,
                equality_jacobian=equality_jacobians[0],
                target_parameter_indices=jnp.asarray(
                    structure.target_parameter_indices, dtype=jnp.int32
                ),
                residual_tolerance=opts.residual_tolerance,
                jit=jit,
            )
        elif selected_mode == "sparse_iterative":
            assert compiled.equality_local_derivatives_function is not None
            assert compiled.inequality_local_derivatives_function is not None
            model_signature = (
                tuple(self.parameter_names),
                tuple(self.variable_names),
                None if self._objective is None else self._objective.text,
                tuple((item.kind, item.text) for item in self.constraints),
            )
            result_device_key = next(iter(initial_state.x.devices()), None)
            device_key = (
                (result_device_key.platform, result_device_key.id)
                if result_device_key is not None
                else ("default", -1)
            )
            sparse_cache_key = (
                model_signature,
                int(parameter_array.shape[0]),
                device_key,
                int(opts.max_iterations),
                float(opts.residual_tolerance),
                float(opts.tau),
                float(opts.step_scale),
                float(opts.minimum_step),
                tuple(float(value) for value in regularization),
                float(gmres_tolerance),
                float(gmres_atol),
                int(gmres_restart),
                int(gmres_maxiter),
                bool(use_preconditioner),
                bool(backtracking),
                int(max_backtracking_steps),
                float(backtracking_factor),
            )
            sparse_solver = self._prepared_sparse_solver_cache.get(sparse_cache_key)
            if sparse_solver is None:
                sparse_solver = prepare_batched_projection_condensed_sparse_solver(
                    batch_size=int(parameter_array.shape[0]),
                    equality_function=compiled.equality_function,
                    inequality_function=compiled.inequality_function,
                    equality_local_derivatives=compiled.equality_local_derivatives_function,
                    inequality_local_derivatives=compiled.inequality_local_derivatives_function,
                    equality_affine_offsets=compiled.equality_affine_offset_function,
                    inequality_affine_offsets=compiled.inequality_affine_offset_function,
                    equality_constant_local_jacobian=compiled.equality_constant_local_jacobian,
                    inequality_constant_local_jacobian=compiled.inequality_constant_local_jacobian,
                    equality_support=jnp.asarray(structure.equality_support, dtype=jnp.int32),
                    equality_mask=jnp.asarray(structure.equality_support_mask, dtype=bool),
                    inequality_support=jnp.asarray(structure.inequality_support, dtype=jnp.int32),
                    inequality_mask=jnp.asarray(structure.inequality_support_mask, dtype=bool),
                    target_parameter_indices=jnp.asarray(
                        structure.target_parameter_indices, dtype=jnp.int32
                    ),
                    indices=indices,
                    bounds=bounds,
                    max_iterations=opts.max_iterations,
                    residual_tolerance=opts.residual_tolerance,
                    tau=opts.tau,
                    step_scale=opts.step_scale,
                    minimum_step=opts.minimum_step,
                    regularization=regularization,
                    gmres_tolerance=gmres_tolerance,
                    gmres_atol=gmres_atol,
                    gmres_restart=gmres_restart,
                    gmres_maxiter=gmres_maxiter,
                    use_preconditioner=use_preconditioner,
                    backtracking=backtracking,
                    max_backtracking_steps=max_backtracking_steps,
                    backtracking_factor=backtracking_factor,
                    constraints_affine=structure.constraints_affine_in_variables,
                    use_column_gather_transpose=device_key[0] != "cpu",
                )
                self._prepared_sparse_solver_cache[sparse_cache_key] = sparse_solver
                self._prepared_sparse_solver_cache_misses += 1
            else:
                self._prepared_sparse_solver_cache_hits += 1
            result = sparse_solver.solve(
                initial_state=initial_state,
                parameter_values=parameter_array,
                mu=mu,
            )
        else:
            result = run_batched_fixed_mu_iterations(
                initial_state=initial_state,
                parameter_values=parameter_array,
                evaluate_nlp=compiled.evaluate_nlp_single,
                indices=indices,
                bounds=bounds,
                mu=mu,
                max_iterations=opts.max_iterations,
                residual_tolerance=opts.residual_tolerance,
                tau=opts.tau,
                step_scale=opts.step_scale,
                minimum_step=opts.minimum_step,
                regularization=regularization,
                check_newton_residual=opts.check_newton_residual,
                jit=jit,
            )

        result_device = next(iter(result.state.x.devices()))
        (
            status_device,
            converged_device,
            iterations_device,
            residual_device,
            final_variables_device,
        ) = jax.device_get(
            (
                result.status_codes,
                result.converged,
                result.iterations,
                result.final_residual_inf_norm,
                result.state.x,
            )
        )
        status_codes = np.asarray(status_device, dtype=np.int32)
        converged = np.asarray(converged_device, dtype=bool)
        iterations_host = np.asarray(iterations_device, dtype=np.int32)
        residual_host = np.asarray(residual_device, dtype=np.float64)
        if raise_on_failure and not np.all(converged):
            failures = [
                (
                    int(index),
                    BATCH_STATUS_NAMES[int(status_codes[index])],
                    float(result.final_residual_inf_norm[index]),
                )
                for index in np.flatnonzero(~converged)
            ]
            raise RuntimeError(
                f"Batched problem {self.name!r} had nonconverged rows in "
                f"mode {selected_mode!r}: {failures}."
            )

        final_variables = np.asarray(final_variables_device, dtype=np.float64)
        if evaluate_outputs:
            objectives = compiled.objectives(final_variables)
            equalities = compiled.equalities(final_variables)
            inequalities = compiled.inequalities(final_variables)
        else:
            objectives = np.full(
                (final_variables.shape[0],), np.nan, dtype=np.float64
            )
            equalities = np.full(
                (final_variables.shape[0], result.state.y_c.shape[1]),
                np.nan,
                dtype=np.float64,
            )
            inequalities = np.full(
                (final_variables.shape[0], result.state.y_d.shape[1]),
                np.nan,
                dtype=np.float64,
            )
        return BatchProblemSolution(
            problem_name=self.name,
            variables=BatchNamedValues(self.variable_names, final_variables),
            parameters=BatchNamedValues(self.parameter_names, compiled.parameter_values),
            objectives=objectives,
            equalities=equalities,
            inequalities=inequalities,
            raw_result=result,
            projection_mode=selected_mode,
            device_platform=result_device.platform,
            status_codes_host=status_codes,
            converged_host=converged,
            iterations_host=iterations_host,
            residual_inf_norm_host=residual_host,
        )

    def batch_solver_cache_info(self) -> dict[str, int]:
        """Return persistent sparse-solver cache statistics."""

        return {
            "compiled_structure_size": len(self._compiled_batch_structure_cache),
            "compiled_structure_hits": self._compiled_batch_structure_cache_hits,
            "compiled_structure_misses": self._compiled_batch_structure_cache_misses,
            "sparse_solver_size": len(self._prepared_sparse_solver_cache),
            "sparse_solver_hits": self._prepared_sparse_solver_cache_hits,
            "sparse_solver_misses": self._prepared_sparse_solver_cache_misses,
        }

    def optimize_batch(self, **kwargs: Any) -> BatchProblemSolution:
        """Alias for :meth:`solve_batch`."""
        return self.solve_batch(**kwargs)

    def solve(
        self,
        *,
        parameters: Mapping[str, float] | Sequence[float] | None = None,
        initial_values: Mapping[str, float] | Sequence[float] | None = None,
        mu: float = 1e-4,
        options: FixedMuOptions | None = None,
        interior_margin: float = 1.0,
        use_filter_line_search: bool = True,
        raise_on_failure: bool = False,
    ) -> ProblemSolution:
        """Compile and solve one parameterized problem instance."""
        compiled = self.compile(
            parameters=parameters,
            initial_values=initial_values,
            interior_margin=interior_margin,
        )
        if options is None:
            options = FixedMuOptions(
                line_search=(
                    FilterLineSearchOptions() if use_filter_line_search else None
                )
            )
        result = run_fixed_mu_iterations(
            initial_state=compiled.initial_state,
            evaluate_nlp=compiled.evaluate_nlp,
            indices=compiled.indices,
            bounds=compiled.bounds,
            mu=mu,
            options=options,
        )
        if raise_on_failure and not result.converged:
            raise RuntimeError(
                f"Problem {self.name!r} did not converge: status={result.status}, "
                f"residual={result.final_residual_inf_norm:.3e}."
            )
        final_variables = result.state.x
        return ProblemSolution(
            problem_name=self.name,
            variables=NamedValues(self.variable_names, final_variables),
            parameters=NamedValues(self.parameter_names, compiled.parameter_values),
            objective=compiled.objective(final_variables),
            equalities=compiled.equalities(final_variables),
            inequalities=compiled.inequalities(final_variables),
            raw_result=result,
        )

    def optimize(self, **kwargs: Any) -> ProblemSolution:
        """Alias for :meth:`solve`, matching the reference modeling API."""
        return self.solve(**kwargs)


# Descriptive aliases for different user preferences.
PrimalDualProblem = Problem
OptimizationProblem = Problem
