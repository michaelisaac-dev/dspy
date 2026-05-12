from typing import get_type_hints

import pytest
from pydantic import BaseModel

import dspy
from dspy.predict.typed_module import TypedModule
from dspy.primitives.prediction import Prediction
from dspy.utils.dummies import DummyLM


class Answer(BaseModel):
    answer: str
    confidence: float


class AnswerOnly(BaseModel):
    answer: str


def _set_lm(answers):
    lm = DummyLM(answers)
    dspy.configure(lm=lm)
    return lm


# ---------------------------------------------------------------------------
# Default casting via class adapter
# ---------------------------------------------------------------------------


def test_cast_to_pydantic_basemodel():
    _set_lm([{"answer": "Paris", "confidence": "0.95"}])
    typed = TypedModule(
        dspy.Predict("question -> answer: str, confidence: float"),
        output_adapter=Answer,
    )

    result = typed(question="Capital of France?")

    assert isinstance(result, Answer)
    assert result.answer == "Paris"
    assert result.confidence == pytest.approx(0.95)


def test_cast_to_primitive_with_single_field():
    _set_lm([{"answer": "4"}])
    typed = TypedModule(dspy.Predict("question -> answer: int"), output_adapter=int)

    result = typed(question="2+2?")

    assert isinstance(result, int)
    assert result == 4


def test_cast_to_primitive_multifield_raises():
    _set_lm([{"answer": "Paris", "reasoning": "France's capital"}])
    typed = TypedModule(
        dspy.Predict("question -> reasoning, answer"),
        output_adapter=str,
    )

    with pytest.raises(TypeError, match="Cannot cast multi-field prediction"):
        typed(question="Capital of France?")


def test_extra_prediction_fields_are_ignored_for_basemodel():
    _set_lm([{"reasoning": "1+1=2", "answer": "2", "confidence": "0.9"}])
    typed = TypedModule(
        dspy.ChainOfThought,
        output_adapter=Answer,
        signature="question -> answer: str, confidence: float",
    )

    result = typed(question="What is 1+1?")

    assert isinstance(result, Answer)
    assert result.answer == "2"
    assert result.confidence == pytest.approx(0.9)


def test_basemodel_missing_required_field_raises():
    _set_lm([{"answer": "Paris"}])
    typed = TypedModule(
        dspy.Predict("question -> answer"),
        output_adapter=Answer,
    )

    with pytest.raises(ValueError, match="missing required field"):
        typed(question="Capital of France?")


def test_basemodel_with_optional_fields_uses_defaults():
    class WithDefault(BaseModel):
        answer: str
        score: float = 0.5

    _set_lm([{"answer": "Paris"}])
    typed = TypedModule(dspy.Predict("question -> answer"), output_adapter=WithDefault)

    result = typed(question="Capital of France?")

    assert isinstance(result, WithDefault)
    assert result.answer == "Paris"
    assert result.score == 0.5


def test_output_type_inferred_from_class_adapter():
    _set_lm([{"answer": "X"}])

    class MyResult:
        def __init__(self, value):
            self.value = value

    typed = TypedModule(dspy.Predict("q -> answer"), output_adapter=MyResult)

    assert typed.output_type is MyResult
    result = typed(q="?")
    assert isinstance(result, MyResult)
    assert result.value == "X"


def test_custom_function_adapter_with_return_annotation():
    _set_lm([{"answer": "paris", "confidence": "0.4"}])

    def adapter(pred) -> Answer:
        return Answer(answer=pred.answer.upper(), confidence=float(pred.confidence) * 100)

    typed = TypedModule(
        dspy.Predict("question -> answer, confidence: float"),
        output_adapter=adapter,
    )

    result = typed(question="Capital of France?")

    assert isinstance(result, Answer)
    assert result.answer == "PARIS"
    assert result.confidence == pytest.approx(40.0)


def test_class_with_prediction_constructor_via_callable_wrapper():
    """Classes that need the raw Prediction in their constructor should be
    wrapped in a callable so they bypass the default field-mapping cast."""
    _set_lm([{"answer": "X"}])

    class MyResult:
        def __init__(self, prediction):
            self.answer = prediction.answer

    def adapter(pred) -> MyResult:
        return MyResult(pred)

    typed = TypedModule(dspy.Predict("q -> answer"), output_adapter=adapter)
    result = typed(q="?")
    assert isinstance(result, MyResult)
    assert result.answer == "X"


def test_function_adapter_returning_wrong_type_raises():
    _set_lm([{"answer": "Paris"}])

    def bad_adapter(pred) -> Answer:
        return pred.answer  # actually returns a str, violating the annotation

    typed = TypedModule(
        dspy.Predict("question -> answer"),
        output_adapter=bad_adapter,
    )

    with pytest.raises(TypeError, match="must return an instance of Answer"):
        typed(question="?")


def test_adapter_without_return_annotation_raises():
    def adapter(pred):
        return Answer(answer=pred.answer, confidence=0.0)

    with pytest.raises(TypeError, match="no return annotation"):
        TypedModule(dspy.Predict("q -> answer"), output_adapter=adapter)


def test_adapter_with_non_class_return_annotation_raises():
    def adapter(pred) -> "Answer | None":
        return Answer(answer=pred.answer, confidence=0.0)

    with pytest.raises(TypeError, match="not a concrete class"):
        TypedModule(dspy.Predict("q -> answer"), output_adapter=adapter)


def test_instantiate_submodule_class_with_kwargs():
    _set_lm([{"reasoning": "1+1=2", "answer": "2", "confidence": "0.9"}])
    typed = TypedModule(
        dspy.ChainOfThought,
        output_adapter=Answer,
        signature="question -> answer: str, confidence: float",
    )

    assert isinstance(typed.module, dspy.ChainOfThought)
    result = typed(question="1+1?")
    assert isinstance(result, Answer)
    assert result.answer == "2"
    assert result.confidence == pytest.approx(0.9)


def test_submodule_kwargs_with_instance_raises():
    instance = dspy.Predict("q -> answer")
    with pytest.raises(ValueError, match="submodule_kwargs are only allowed when submodule is a class"):
        TypedModule(instance, output_adapter=AnswerOnly, signature="q -> answer")


def test_invalid_submodule_class_raises():
    class NotAModule:
        pass

    with pytest.raises(TypeError, match="must be a subclass of dspy.Module"):
        TypedModule(NotAModule, output_adapter=str)


def test_invalid_submodule_instance_raises():
    with pytest.raises(TypeError, match="must be a dspy.Module instance or class"):
        TypedModule("not-a-module", output_adapter=str)


def test_invalid_output_adapter_raises():
    with pytest.raises(TypeError, match="must be a class or a callable"):
        TypedModule(dspy.Predict("q -> answer"), output_adapter="not-callable")


@pytest.mark.asyncio
async def test_aforward_casts_output():
    _set_lm([{"answer": "Paris", "confidence": "0.9"}])
    typed = TypedModule(
        dspy.Predict("question -> answer: str, confidence: float"),
        output_adapter=Answer,
    )

    result = await typed.acall(question="Capital of France?")

    assert isinstance(result, Answer)
    assert result.answer == "Paris"


def test_named_predictors_recurse_into_inner_module():
    typed = TypedModule(
        dspy.ChainOfThought,
        output_adapter=AnswerOnly,
        signature="question -> answer",
    )

    names = [n for n, _ in typed.named_predictors()]
    assert names == ["module.predict"]
    assert len(typed.predictors()) == 1


def test_named_parameters_finds_predict_directly():
    typed = TypedModule(dspy.Predict("q -> answer"), output_adapter=AnswerOnly)
    names = [n for n, _ in typed.named_predictors()]
    assert names == ["module"]


def test_set_lm_propagates_to_inner_predictors():
    typed = TypedModule(
        dspy.ChainOfThought,
        output_adapter=AnswerOnly,
        signature="question -> answer",
    )
    lm = DummyLM([{"answer": "x"}])
    typed.set_lm(lm)

    for _, predictor in typed.named_predictors():
        assert predictor.lm is lm


def test_get_lm_works_through_wrapper():
    typed = TypedModule(
        dspy.ChainOfThought,
        output_adapter=AnswerOnly,
        signature="question -> answer",
    )
    lm = DummyLM([{"answer": "x"}])
    typed.set_lm(lm)
    assert typed.get_lm() is lm


def test_deepcopy_creates_independent_inner_module():
    typed = TypedModule(dspy.Predict("q -> answer"), output_adapter=AnswerOnly)
    clone = typed.deepcopy()

    assert clone.module is not typed.module
    assert clone.output_type is typed.output_type
    assert isinstance(clone, TypedModule)


def test_reset_copy_resets_inner_predictor_state():
    typed = TypedModule(dspy.Predict("q -> answer"), output_adapter=AnswerOnly)
    typed.module.demos = [dspy.Example(q="hi", answer="ok").with_inputs("q")]
    typed.module.lm = DummyLM([{"answer": "x"}])

    fresh = typed.reset_copy()

    assert fresh.module.demos == []
    assert fresh.module.lm is None


def test_dump_and_load_state_roundtrip():
    typed = TypedModule(dspy.Predict("q -> answer"), output_adapter=AnswerOnly)
    typed.module.demos = [dspy.Example(q="hi", answer="ok").with_inputs("q")]

    state = typed.dump_state()
    fresh = TypedModule(dspy.Predict("q -> answer"), output_adapter=AnswerOnly)
    fresh.load_state(state)

    assert len(fresh.module.demos) == 1
    assert fresh.module.demos[0]["answer"] == "ok"


def test_signature_property_forwards_inner_signature():
    inner = dspy.Predict("question -> answer")
    typed = TypedModule(inner, output_adapter=AnswerOnly)
    assert typed.signature is inner.signature


def test_inner_module_must_return_prediction():
    class BadModule(dspy.Module):
        def forward(self, **kwargs):
            return "not a prediction"

    typed = TypedModule(BadModule(), output_adapter=str)
    with pytest.raises(TypeError, match="Expected the inner module to return a dspy.Prediction"):
        typed()


def test_passthrough_when_prediction_already_is_output_type():
    class PredictionSubclass(Prediction):
        pass

    class ReturnsSubclass(dspy.Module):
        def forward(self, **kwargs):
            return PredictionSubclass(answer="x")

    typed = TypedModule(ReturnsSubclass(), output_adapter=PredictionSubclass)
    result = typed()
    assert isinstance(result, PredictionSubclass)


def test_works_inside_user_module_with_optimizer():
    """End-to-end: TypedModule wrapped in a user module can be optimized by BootstrapFewShot."""

    class QA(dspy.Module):
        def __init__(self):
            super().__init__()
            self.qa = TypedModule(dspy.Predict("question -> answer"), output_adapter=AnswerOnly)

        def forward(self, question):
            typed = self.qa(question=question)
            return dspy.Prediction(answer=typed.answer)

    lm = DummyLM(
        [
            {"answer": "4"},
            {"answer": "6"},
            {"answer": "8"},
        ]
        * 6
    )
    dspy.configure(lm=lm)

    trainset = [
        dspy.Example(question="2+2?", answer="4").with_inputs("question"),
        dspy.Example(question="3+3?", answer="6").with_inputs("question"),
        dspy.Example(question="4+4?", answer="8").with_inputs("question"),
    ]

    def metric(ex, pred, trace=None):
        return ex.answer == pred.answer

    from dspy.teleprompt import BootstrapFewShot

    optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=2)
    program = QA()
    compiled = optimizer.compile(program, trainset=trainset)
    assert compiled is not None
    assert [n for n, _ in compiled.named_predictors()] == ["qa.module"]


def test_lm_usage_metadata_is_preserved_when_possible():
    class CustomOutput:
        def __init__(self, value):
            self.value = value

    def adapter(pred) -> CustomOutput:
        return CustomOutput(pred.value)

    class FakeModule(dspy.Module):
        def forward(self, **kwargs):
            pred = Prediction(value="hello")
            pred.set_lm_usage({"prompt_tokens": 5, "completion_tokens": 3})
            return pred

    typed = TypedModule(FakeModule(), output_adapter=adapter)

    result = typed()
    assert isinstance(result, CustomOutput)
    assert getattr(result, "_lm_usage", None) == {
        "prompt_tokens": 5,
        "completion_tokens": 3,
    }


def test_repr_includes_inner_module_and_output_type():
    typed = TypedModule(dspy.Predict("q -> answer"), output_adapter=AnswerOnly)
    representation = repr(typed)
    assert "TypedModule" in representation
    assert "AnswerOnly" in representation


def test_typed_module_is_generic():
    """`TypedModule[Answer]` should produce a parametrized generic alias."""
    alias = TypedModule[Answer]
    assert getattr(alias, "__origin__", None) is TypedModule


def test_explicit_type_parameter_annotation_does_not_break_runtime():
    _set_lm([{"answer": "Paris", "confidence": "0.9"}])

    typed: TypedModule[Answer] = TypedModule(
        dspy.Predict("question -> answer: str, confidence: float"),
        output_adapter=Answer,
    )
    result = typed(question="Capital of France?")
    assert isinstance(result, Answer)


def test_forward_aforward_have_typevar_in_annotations():
    """`forward`/`aforward`/`_cast` return annotations must remain in place —
    this is what IDEs and static type checkers use to track the output type."""
    assert "return" in TypedModule.forward.__annotations__
    assert "return" in TypedModule.aforward.__annotations__
    assert "return" in get_type_hints(TypedModule._cast)


def test_dataclass_adapter():
    from dataclasses import dataclass

    @dataclass
    class DCAnswer:
        answer: str
        confidence: float

    _set_lm([{"answer": "Paris", "confidence": "0.9"}])
    typed = TypedModule(
        dspy.Predict("q -> answer, confidence: float"),
        output_adapter=DCAnswer,
    )

    result = typed(q="?")
    assert isinstance(result, DCAnswer)
    assert result.answer == "Paris"
    assert result.confidence == pytest.approx(0.9)


def test_dataclass_with_defaults_uses_them():
    from dataclasses import dataclass, field

    @dataclass
    class WithDefaults:
        answer: str
        score: float = 0.5
        tags: list = field(default_factory=list)

    _set_lm([{"answer": "Paris"}])
    typed = TypedModule(dspy.Predict("q -> answer"), output_adapter=WithDefaults)

    result = typed(q="?")
    assert isinstance(result, WithDefaults)
    assert result.answer == "Paris"
    assert result.score == 0.5
    assert result.tags == []


def test_dataclass_missing_required_field_raises():
    from dataclasses import dataclass

    @dataclass
    class DC:
        answer: str
        confidence: float  # required, no default

    _set_lm([{"answer": "Paris"}])
    typed = TypedModule(dspy.Predict("q -> answer"), output_adapter=DC)

    with pytest.raises(ValueError, match="missing required field"):
        typed(q="?")


def test_subscripted_generic_rejected():
    """`list[str]` and similar subscripted generics should be rejected with
    a clear error, since `isinstance(x, list[str])` raises at runtime."""
    with pytest.raises(TypeError, match="Subscripted generics"):
        TypedModule(dspy.Predict("q -> answer"), output_adapter=list[str])


def test_adapter_with_subscripted_generic_return_rejected():
    def adapter(pred) -> list[str]:
        return [pred.answer]

    with pytest.raises(TypeError, match="not a concrete class"):
        TypedModule(dspy.Predict("q -> answer"), output_adapter=adapter)


def test_forward_ref_annotation_resolves():
    """String/forward-ref return annotations like `-> "Answer"` should be
    resolved by `typing.get_type_hints` against the function's globals."""

    def adapter(pred) -> "Answer":
        return Answer(answer=pred.answer, confidence=float(pred.confidence))

    _set_lm([{"answer": "Paris", "confidence": "0.9"}])
    typed = TypedModule(
        dspy.Predict("q -> answer, confidence: float"),
        output_adapter=adapter,
    )
    assert typed.output_type is Answer

    result = typed(q="?")
    assert isinstance(result, Answer)
    assert result.answer == "Paris"


def test_subclass_of_typed_module_works():
    """Users should be able to subclass `TypedModule[Answer]` directly."""

    class MyTyped(TypedModule[Answer]):
        pass

    _set_lm([{"answer": "Paris", "confidence": "0.9"}])
    sub = MyTyped(
        dspy.Predict("q -> answer, confidence: float"),
        output_adapter=Answer,
    )
    result = sub(q="?")
    assert isinstance(result, Answer)
    assert isinstance(sub, MyTyped)
    assert isinstance(sub, TypedModule)


def test_cloudpickle_round_trip_preserves_typed_module():
    """`save_program=True` uses cloudpickle. The class and its generic
    parameterization must survive a serialize/deserialize round-trip."""
    import cloudpickle

    typed = TypedModule(dspy.Predict("q -> answer"), output_adapter=AnswerOnly)
    blob = cloudpickle.dumps(typed)
    restored = cloudpickle.loads(blob)

    assert isinstance(restored, TypedModule)
    assert restored.output_type is AnswerOnly
    assert [n for n, _ in restored.named_predictors()] == ["module"]


def test_forward_can_be_monkeypatched_to_return_tuple():
    """GEPA's `bootstrap_trace` wraps `program.forward` to return
    `(prediction, trace)`. TypedModule's cast must still happen via the
    original forward, and the outer tuple must propagate cleanly."""
    from types import MethodType

    _set_lm([{"answer": "Paris"}])
    typed = TypedModule(dspy.Predict("q -> answer"), output_adapter=AnswerOnly)
    original_forward = object.__getattribute__(typed, "forward")

    def patched(self, **kwargs):
        return original_forward(**kwargs), "fake_trace"

    typed.forward = MethodType(patched, typed)

    result = typed(q="?")
    assert isinstance(result, tuple)
    assert isinstance(result[0], AnswerOnly)
    assert result[1] == "fake_trace"


def test_callable_adapter_returning_none_raises():
    """If a callable adapter returns None (e.g. forgotten return), surface
    a clear TypeError instead of letting a None leak downstream."""

    def adapter(pred) -> AnswerOnly:
        return None  # type: ignore[return-value]

    _set_lm([{"answer": "Paris"}])
    typed = TypedModule(dspy.Predict("q -> answer"), output_adapter=adapter)

    with pytest.raises(TypeError, match="must return an instance of AnswerOnly"):
        typed(q="?")

