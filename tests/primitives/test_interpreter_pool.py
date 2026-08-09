"""Tests for dspy.PooledInterpreterFactory (warm interpreter reuse)."""

import copy
import shutil
import threading

import pytest

import dspy
from dspy.primitives.code_interpreter import CodeInterpreter, CodeInterpreterError
from dspy.primitives.interpreter_pool import PooledInterpreterFactory, _PooledInterpreter

deno_required = pytest.mark.skipif(shutil.which("deno") is None, reason="Deno is not installed")


class FakeReusableInterpreter:
    """Emulates PythonInterpreter's pooling contract (prepare_for_reuse + sandbox reset)."""

    def __init__(self, log):
        self.log = log
        self.log["boots"] += 1
        self.tools = {}
        self.output_fields = None
        self._tools_registered = False
        self.namespace = {}
        self.shut_down = False
        self.fail_reset = False
        self.fail_start = False

    def start(self):
        if self.shut_down or self.fail_start:
            raise CodeInterpreterError("interpreter session has ended")

    def execute(self, code, variables=None):
        if self.shut_down:
            raise CodeInterpreterError("interpreter session has ended")
        if code.strip() == "print(_dspy_pool_reset())":
            if self.fail_reset:
                return "something went wrong"
            self.namespace.clear()
            return "_dspy_pool_reset_ok\n"
        self.namespace[code] = variables
        return ""

    def shutdown(self):
        self.shut_down = True

    def prepare_for_reuse(self):
        self.tools = {}
        self.output_fields = None
        self._tools_registered = False


class PlainInterpreter:
    """A CodeInterpreter without prepare_for_reuse; must never be pooled."""

    def __init__(self, log):
        log["boots"] += 1
        self.tools = {}
        self.shut_down = False

    def start(self):
        pass

    def execute(self, code, variables=None):
        return ""

    def shutdown(self):
        self.shut_down = True


def test_sequential_leases_reuse_one_interpreter():
    log = {"boots": 0}
    factory = PooledInterpreterFactory(lambda: FakeReusableInterpreter(log))

    underlyings = []
    for _ in range(3):
        interp = factory()
        interp.execute("x = 1")
        underlyings.append(interp._interpreter)
        interp.shutdown()

    assert log["boots"] == 1
    assert underlyings[0] is underlyings[1] is underlyings[2]
    # The sandbox namespace was reset between leases.
    assert underlyings[0].namespace == {}


def test_concurrent_leases_get_distinct_interpreters():
    log = {"boots": 0}
    factory = PooledInterpreterFactory(lambda: FakeReusableInterpreter(log))

    first, second = factory(), factory()
    assert first._interpreter is not second._interpreter
    assert log["boots"] == 2
    first.shutdown()
    second.shutdown()

    # Both are now warm and get handed out again.
    third = factory()
    fourth = factory()
    assert log["boots"] == 2
    third.shutdown()
    fourth.shutdown()


def test_failed_reset_discards_interpreter():
    log = {"boots": 0}
    factory = PooledInterpreterFactory(lambda: FakeReusableInterpreter(log))

    interp = factory()
    underlying = interp._interpreter
    underlying.fail_reset = True
    interp.shutdown()

    assert underlying.shut_down
    replacement = factory()
    assert replacement._interpreter is not underlying
    assert log["boots"] == 2
    replacement.shutdown()


def test_interpreter_that_dies_while_idle_is_replaced():
    log = {"boots": 0}
    factory = PooledInterpreterFactory(lambda: FakeReusableInterpreter(log))

    interp = factory()
    underlying = interp._interpreter
    interp.shutdown()
    underlying.fail_start = True

    replacement = factory()
    assert replacement._interpreter is not underlying
    assert underlying.shut_down
    replacement.shutdown()


def test_non_reusable_interpreter_passes_through():
    log = {"boots": 0}
    factory = PooledInterpreterFactory(lambda: PlainInterpreter(log))

    interp = factory()
    assert isinstance(interp, PlainInterpreter)
    assert not isinstance(interp, _PooledInterpreter)
    interp.shutdown()
    assert interp.shut_down

    factory().shutdown()
    assert log["boots"] == 2


def test_lease_rejects_use_after_shutdown():
    log = {"boots": 0}
    factory = PooledInterpreterFactory(lambda: FakeReusableInterpreter(log))

    interp = factory()
    interp.shutdown()
    interp.shutdown()  # Idempotent.
    with pytest.raises(RuntimeError, match="already shut down"):
        interp.execute("x = 1")


def test_lease_forwards_tools_and_attributes():
    log = {"boots": 0}
    factory = PooledInterpreterFactory(lambda: FakeReusableInterpreter(log))

    interp = factory()
    underlying = interp._interpreter
    assert isinstance(interp, CodeInterpreter)

    interp.tools.update({"my_tool": lambda: "hi"})
    interp.output_fields = [{"name": "answer"}]
    assert "my_tool" in underlying.tools
    assert underlying.output_fields == [{"name": "answer"}]

    interp.shutdown()
    # Returning the lease clears per-session bindings for the next user.
    assert underlying.tools == {}
    assert underlying.output_fields is None


def test_max_idle_cap():
    log = {"boots": 0}
    factory = PooledInterpreterFactory(lambda: FakeReusableInterpreter(log), max_idle=1)

    first, second = factory(), factory()
    kept, capped = first._interpreter, second._interpreter
    first.shutdown()
    second.shutdown()

    assert not kept.shut_down
    assert capped.shut_down


def test_close_shuts_down_idle_interpreters():
    log = {"boots": 0}
    factory = PooledInterpreterFactory(lambda: FakeReusableInterpreter(log))

    interp = factory()
    underlying = interp._interpreter
    interp.shutdown()
    factory.close()
    assert underlying.shut_down


def test_deepcopy_shares_the_pool():
    factory = PooledInterpreterFactory(lambda: FakeReusableInterpreter({"boots": 0}))
    assert copy.deepcopy(factory) is factory


def test_threaded_leases_are_safe():
    log = {"boots": 0}
    factory = PooledInterpreterFactory(lambda: FakeReusableInterpreter(log))

    def use_pool():
        for _ in range(5):
            interp = factory()
            interp.execute("x = 1")
            interp.shutdown()

    threads = [threading.Thread(target=use_pool) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Never more interpreters than peak concurrency.
    assert log["boots"] <= 4


@deno_required
def test_python_interpreter_reuse_and_isolation():
    """Real Deno/Pyodide: leases share one process but see isolated sandbox state."""
    factory = PooledInterpreterFactory(dspy.PythonInterpreter)

    first = factory()
    first.tools.update({"greet": lambda name: f"hi {name}"})
    first_underlying = first._interpreter
    assert first.execute("leak = 41\nprint(greet(name='a'))").strip() == "hi a"
    first.shutdown()

    second = factory()
    try:
        assert second._interpreter is first_underlying  # Warm process reused.
        # As with a fresh interpreter, tools are set before the first execute.
        second.tools.update({"shout": lambda text: text.upper()})
        # Globals from the previous lease are gone; this lease's own tool works.
        out = second.execute("print('leak' in dir())\nprint(shout(text='ok'))")
        assert out.strip().splitlines() == ["False", "OK"]
        # The previous lease's tool is gone.
        with pytest.raises(CodeInterpreterError):
            second.execute("print(greet(name='b'))")
    finally:
        second.shutdown()
        factory.close()
