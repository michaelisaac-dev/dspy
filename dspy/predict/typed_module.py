import dataclasses
import logging
import typing
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar, overload

from pydantic import BaseModel

from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _is_concrete_class(obj: Any) -> bool:
    """True if `obj` is a real class (not a subscripted generic like `list[str]`).

    `isinstance(list[str], type)` is True on Python 3.10+, but
    `isinstance(x, list[str])` raises at runtime — so we reject anything
    with a typing origin even though it passes the `type` check.
    """
    return isinstance(obj, type) and typing.get_origin(obj) is None


def _infer_output_type(adapter: Callable[..., Any]) -> type:
    """Infer the target output class produced by `adapter`.

    A class adapter is its own output type. For function adapters, the
    return annotation must resolve (via `typing.get_type_hints`) to a
    concrete class.
    """
    if isinstance(adapter, type):
        if not _is_concrete_class(adapter):
            raise TypeError(
                f"Subscripted generics like {adapter!r} are not supported as "
                "output_adapter. Pass a concrete class, or use a callable adapter."
            )
        return adapter

    try:
        hints = typing.get_type_hints(adapter)
    except Exception as e:
        raise TypeError(
            "Cannot infer output type from output_adapter: failed to read "
            f"return annotation ({e}). Annotate the adapter's return type, or "
            "pass a class as the adapter."
        ) from e

    if "return" not in hints:
        raise TypeError(
            "Cannot infer output type from output_adapter: no return annotation "
            "found. Annotate the adapter's return type, or pass a class as the "
            "adapter."
        )

    ret = hints["return"]
    if not _is_concrete_class(ret):
        raise TypeError(
            f"Cannot infer output type from output_adapter: return annotation "
            f"{ret!r} is not a concrete class. Use a function annotated with a "
            "concrete return type, or pass a class as the adapter."
        )

    return ret


class TypedModule(Module, Generic[T]):
    """Wrapper that casts a sub-module's `Prediction` output to a typed object.

    `TypedModule` runs an inner `dspy.Module` and converts its `Prediction`
    output into an instance of the type specified by `output_adapter`. The
    result satisfies `isinstance(result, output_type)`, so downstream code
    can treat the wrapper as if it returned a strongly-typed value.

    The class is generic in its output type, so IDEs and static type
    checkers (PyCharm, Pylance/pyright, mypy) infer the correct return
    type for calls. The type parameter is deduced from `output_adapter`:

        ```python
        typed = dspy.TypedModule(predict, output_adapter=Answer)
        result = typed(question="?")            # reveal_type(result) -> Answer

        def to_answer(pred) -> Answer: ...
        typed = dspy.TypedModule(predict, output_adapter=to_answer)
        result = typed(question="?")            # reveal_type(result) -> Answer
        ```

    The wrapper is transparent to optimizers: it stores the inner module as
    `self.module`, so `named_parameters`, `predictors`, `set_lm`,
    `dump_state`, `deepcopy`, etc. recurse into it.

    Args:
        submodule: A `dspy.Module` instance to wrap, or a `dspy.Module`
            subclass to instantiate (using `**submodule_kwargs`).
        output_adapter: How to produce the typed output. Pass either:

            - A class. Field-mapping construction:
                * `pydantic.BaseModel` subclass → `cls(**matching_fields)`.
                * `@dataclass` → `cls(**matching_fields)`.
                * Other class (e.g. `int`, `str`) → `cls(value)`. Requires
                  the signature to have exactly one output field.
            - A callable `(prediction) -> T`. The wrapper calls it with
              the raw `Prediction` and returns its result. The output type
              is read from the callable's return annotation.

            Subscripted generics (e.g. `list[str]`) are rejected.
        **submodule_kwargs: Forwarded to `submodule(**submodule_kwargs)`
            when `submodule` is a class.

    Examples:
        Pydantic model with multi-field signature:

        >>> import dspy
        >>> from pydantic import BaseModel
        >>>
        >>> class Answer(BaseModel):
        ...     answer: str
        ...     confidence: float
        >>>
        >>> typed = dspy.TypedModule(
        ...     dspy.Predict("question -> answer: str, confidence: float"),
        ...     output_adapter=Answer,
        ... )
        >>> result = typed(question="What is 1+1?")
        >>> isinstance(result, Answer)
        True

        Primitive output:

        >>> typed = dspy.TypedModule(
        ...     dspy.Predict("question -> answer: int"),
        ...     output_adapter=int,
        ... )
        >>> result = typed(question="What is 2+2?")
        >>> isinstance(result, int)
        True

        Custom conversion via a callable (output type inferred from the
        return annotation):

        >>> def to_answer(pred) -> Answer:
        ...     return Answer(answer=pred.answer, confidence=float(pred.confidence))
        >>> typed = dspy.TypedModule(predict, output_adapter=to_answer)

        Instantiate the inner module from a class:

        >>> typed = dspy.TypedModule(
        ...     dspy.ChainOfThought,
        ...     output_adapter=Answer,
        ...     signature="question -> answer: str, confidence: float",
        ... )
    """

    output_type: type[T]
    output_adapter: type[T] | Callable[[Prediction], T]
    module: Module

    @overload
    def __init__(
        self,
        submodule: Module | type[Module],
        output_adapter: type[T],
        **submodule_kwargs: Any,
    ) -> None: ...

    @overload
    def __init__(
        self,
        submodule: Module | type[Module],
        output_adapter: Callable[[Prediction], T],
        **submodule_kwargs: Any,
    ) -> None: ...

    def __init__(
        self,
        submodule: Module | type[Module],
        output_adapter: type[T] | Callable[[Prediction], T],
        **submodule_kwargs: Any,
    ) -> None:
        super().__init__()

        if isinstance(submodule, type):
            if not issubclass(submodule, Module):
                raise TypeError(
                    f"submodule class must be a subclass of dspy.Module, got {submodule.__name__}."
                )
            self.module = submodule(**submodule_kwargs)
        else:
            if not isinstance(submodule, Module):
                raise TypeError(
                    f"submodule must be a dspy.Module instance or class, got {type(submodule).__name__}."
                )
            if submodule_kwargs:
                raise ValueError(
                    "submodule_kwargs are only allowed when submodule is a class. "
                    f"Got an instance of {type(submodule).__name__} with kwargs={list(submodule_kwargs)}."
                )
            self.module = submodule

        if not callable(output_adapter):
            raise TypeError(
                f"output_adapter must be a class or a callable, got {type(output_adapter).__name__}."
            )

        self.output_type = _infer_output_type(output_adapter)
        self.output_adapter = output_adapter

    # Type-only declarations so that static analyzers (PyCharm, Pylance, mypy)
    # see `TypedModule[T].__call__(...) -> T`. At runtime, `Module.__call__`
    # is used (so the `@with_callbacks` decorator on the parent is preserved
    # and not duplicated).
    if TYPE_CHECKING:
        def __call__(self, *args: Any, **kwargs: Any) -> T: ...  # type: ignore[override]
        async def acall(self, *args: Any, **kwargs: Any) -> T: ...  # type: ignore[override]

    @property
    def signature(self):
        """Forward the inner module's signature so optimizers can introspect it."""
        return getattr(self.module, "signature", None)

    def forward(self, **kwargs: Any) -> T:
        prediction = self.module(**kwargs)
        return self._cast(prediction)

    async def aforward(self, **kwargs: Any) -> T:
        prediction = await self.module.acall(**kwargs)
        return self._cast(prediction)

    def _cast(self, prediction: Prediction) -> T:
        if not isinstance(prediction, Prediction):
            raise TypeError(
                f"Expected the inner module to return a dspy.Prediction, "
                f"got {type(prediction).__name__}."
            )

        # A class adapter goes through the default field-mapping cast. A
        # non-class callable adapter is invoked directly on the Prediction.
        if isinstance(self.output_adapter, type):
            result = self._default_cast(prediction)
        else:
            result = self.output_adapter(prediction)

        if not isinstance(result, self.output_type):
            raise TypeError(
                f"output_adapter must return an instance of {self.output_type.__name__}, "
                f"got {type(result).__name__}."
            )

        self._preserve_metadata(prediction, result)
        return result

    def _default_cast(self, prediction: Prediction) -> T:
        if isinstance(prediction, self.output_type):
            return prediction  # type: ignore[return-value]

        fields = dict(prediction.items())

        if issubclass(self.output_type, BaseModel):
            model_fields = self.output_type.model_fields
            relevant = {k: v for k, v in fields.items() if k in model_fields}
            missing = [k for k in model_fields if k not in fields and model_fields[k].is_required()]
            if missing:
                raise ValueError(
                    f"Cannot construct {self.output_type.__name__}: prediction is missing "
                    f"required field(s) {missing}. Prediction has fields {list(fields)}."
                )
            return self.output_type(**relevant)  # type: ignore[return-value]

        if dataclasses.is_dataclass(self.output_type):
            dc_fields = {f.name for f in dataclasses.fields(self.output_type)}
            relevant = {k: v for k, v in fields.items() if k in dc_fields}
            missing = [
                f.name
                for f in dataclasses.fields(self.output_type)
                if f.name not in fields
                and f.default is dataclasses.MISSING
                and f.default_factory is dataclasses.MISSING
            ]
            if missing:
                raise ValueError(
                    f"Cannot construct {self.output_type.__name__}: prediction is missing "
                    f"required field(s) {missing}. Prediction has fields {list(fields)}."
                )
            return self.output_type(**relevant)  # type: ignore[return-value]

        if len(fields) == 1:
            value = next(iter(fields.values()))
        elif len(fields) == 0:
            raise ValueError(
                f"Cannot cast empty prediction to {self.output_type.__name__}."
            )
        else:
            raise TypeError(
                f"Cannot cast multi-field prediction (fields={list(fields)}) to "
                f"{self.output_type.__name__}. Use a pydantic BaseModel or a callable "
                "adapter to pick the right field(s)."
            )

        if isinstance(value, self.output_type):
            return value  # type: ignore[return-value]

        return self.output_type(value)  # type: ignore[call-arg]

    def _preserve_metadata(self, prediction: Prediction, result: Any) -> None:
        usage = prediction.get_lm_usage() if hasattr(prediction, "get_lm_usage") else None
        if not usage:
            return
        try:
            object.__setattr__(result, "_lm_usage", usage)
        except (AttributeError, TypeError, ValueError):
            pass

    def _set_lm_usage(self, tokens: dict[str, Any], output: Any) -> None:
        if isinstance(output, Prediction):
            super()._set_lm_usage(tokens, output)
            return
        try:
            object.__setattr__(output, "_lm_usage", tokens)
        except (AttributeError, TypeError, ValueError):
            super()._set_lm_usage(tokens, output)

    def __repr__(self) -> str:  # type: ignore[override]
        return (
            f"TypedModule(module={self.module!r}, "
            f"output_type={self.output_type.__name__})"
        )
