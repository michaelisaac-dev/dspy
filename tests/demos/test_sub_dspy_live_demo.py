"""Live demo (not for commit): a real LM drives RLM and must write code that imports and uses dspy.

Two backends:
* SUB_DSPY on a dspy-interpreters SubprocessInterpreter — native dspy runs inside the worker.
* The default PythonInterpreter (Deno/Pyodide) — the bridged dspy facade, out of the box.
"""

import os

import pytest
from dotenv import load_dotenv

import dspy
from dspy.predict.rlm import RLM
from dspy.primitives.code_interpreter import SUB_DSPY_FACTORY_NAME, InterpreterCapability
from dspy.primitives.python_interpreter import PythonInterpreter

load_dotenv()  # credentials from the repo's git-ignored .env

# Point DEMO_LM at any litellm model string with its API key in .env. No sampling params: Opus 5 rejects them.
LM = dspy.LM(os.environ.get("DEMO_LM", "anthropic/claude-opus-5"), cache=False)

TASK = dspy.Signature(
    "passage -> summary",
    "Summarize the passage in one sentence. You MUST do this by building a dspy.Predict sub-agent "
    "in the REPL: import dspy, construct dspy.Predict('passage -> summary'), call it on the passage, "
    "print the sub-agent's summary, and then SUBMIT that summary.",
)
PASSAGE = (
    "The Recursive Language Model treats a long context as an external environment: the model writes "
    "Python code to inspect the text, decomposes it into pieces, and delegates the pieces to sub-agents "
    "before combining their answers into a final result."
)


def dspy_steps(trajectory):
    """Trajectory steps whose code uses dspy, with their outputs, printed for inspection."""
    steps = [step for step in trajectory if "dspy" in step["code"]]
    for step in trajectory:
        print("\n--- code ---\n" + step["code"] + "\n--- output ---\n" + str(step["output"]))
    return steps


def assert_dspy_used_successfully(result):
    steps = dspy_steps(result.trajectory)
    assert steps, "the model never wrote code using dspy"
    assert any("import dspy" in step["code"] for step in steps), "the model never imported dspy"
    # RLM recovers from bad attempts by design; what must hold is that a dspy.Predict step ran successfully.
    assert any(
        "dspy.Predict" in step["code"] and not str(step["output"]).startswith("[Error]") for step in steps
    ), "no dspy.Predict step succeeded"
    assert isinstance(result.summary, str) and len(result.summary) > 10
    print("\nFINAL SUMMARY:", result.summary)


def test_sub_dspy_native_dspy_in_a_worker_process():
    dspy_interpreters = pytest.importorskip("dspy_interpreters")

    class SubDspyWorker(dspy_interpreters.SubprocessInterpreter):
        capabilities = InterpreterCapability.SUB_DSPY

    interpreter = SubDspyWorker()
    try:
        interpreter.execute(f"from dspy_interpreters import SubprocessInterpreter as {SUB_DSPY_FACTORY_NAME}")
        # The worker really has dspy: this is the real package, not a facade.
        probe = str(interpreter.execute("import dspy\nprint(dspy.__file__, hasattr(dspy, 'LM'))"))
        print("\nWORKER PROBE:", probe.strip())
        assert "dspy/__init__.py" in probe and "True" in probe

        rlm = RLM(TASK, max_iters=6, interpreter_factory=SubDspyWorker)
        assert rlm._sub_dspy  # native mode: no facade is installed, so any dspy call ran in the worker
        with dspy.context(lm=LM):
            result = rlm(interpreter, passage=PASSAGE)
    finally:
        interpreter.shutdown()

    assert_dspy_used_successfully(result)


def test_default_python_interpreter_facade():
    interpreter = PythonInterpreter()
    try:
        rlm = RLM(TASK, max_iters=6)  # default interpreter_factory is PythonInterpreter
        assert not rlm._sub_dspy  # facade mode
        with dspy.context(lm=LM):
            result = rlm(interpreter, passage=PASSAGE)
        # After the run, the sandbox's importable dspy is the facade: has Predict, has no LM.
        probe = str(interpreter.execute("import dspy\nprint(type(dspy).__name__, hasattr(dspy, 'Predict'), hasattr(dspy, 'LM'))"))
        print("\nSANDBOX PROBE:", probe.strip())
        assert "module True False" in probe
    finally:
        interpreter.shutdown()

    assert_dspy_used_successfully(result)

