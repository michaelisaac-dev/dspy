"""Warm interpreter reuse for code-executing modules.

Booting a sandbox interpreter can dominate wall-clock time: a fresh
``dspy.PythonInterpreter`` spawns a Deno subprocess and loads Pyodide (roughly one to two
seconds), while a warm ``execute()`` takes a few milliseconds. Modules such as ``dspy.Flex``,
``dspy.RLM``, and ``dspy.CodeAct`` create an interpreter from a factory each ``forward`` and shut
it down afterwards, so every forward pays the boot cost.

``PooledInterpreterFactory`` keeps that per-forward lifecycle but makes it cheap: it is itself a
zero-argument interpreter factory, and each call returns a lease over a warm interpreter drawn
from a pool. Calling ``shutdown()`` on the lease resets the sandbox back to its just-booted state
and returns the warm interpreter to the pool instead of terminating it.

An interpreter is only recycled when it exposes ``prepare_for_reuse()``
(``dspy.PythonInterpreter`` does); any other interpreter is passed through unchanged, with
``shutdown()`` really shutting it down.

The sandbox reset is best-effort isolation, not a security boundary: globals and modules imported
after boot are removed, but mutations to preloaded modules or builtins survive. The host remains
protected by the sandbox itself either way. Pass ``reuse_interpreter=False`` to ``dspy.Flex`` (or
use a plain factory) when every forward must run in a brand-new process.
"""

import logging
import threading
import weakref
from typing import Any, Callable

from dspy.primitives.code_interpreter import (
    CodeInterpreter,
    _create_interpreter,
    _validate_interpreter_factory,
)

logger = logging.getLogger(__name__)

# Runs once per pooled interpreter right after boot. Snapshots the just-booted sandbox state
# (global names and loaded modules) and defines the reset hook that restores it. If user code
# deletes or overwrites these names, the reset fails and the interpreter is discarded instead
# of being recycled — failing safe.
_BASELINE_SNAPSHOT_CODE = """
import sys as _dspy_pool_sys

def _dspy_pool_reset():
    _globals = globals()
    for _name in list(_globals):
        if _name not in _dspy_pool_baseline_globals:
            del _globals[_name]
    for _name in list(_dspy_pool_sys.modules):
        if _name not in _dspy_pool_baseline_modules:
            del _dspy_pool_sys.modules[_name]
    return "_dspy_pool_reset_ok"

_dspy_pool_baseline_globals = set(globals()) | {
    "_dspy_pool_sys",
    "_dspy_pool_reset",
    "_dspy_pool_baseline_globals",
    "_dspy_pool_baseline_modules",
}
_dspy_pool_baseline_modules = set(_dspy_pool_sys.modules)
"""

_RESET_CODE = "print(_dspy_pool_reset())"
_RESET_OK = "_dspy_pool_reset_ok"


class _PooledInterpreter:
    """A lease over a pooled interpreter.

    Implements ``CodeInterpreter`` by forwarding to the underlying interpreter, except that
    ``shutdown()`` resets the sandbox and returns the warm interpreter to the pool. A lease
    must not be used after ``shutdown()``.
    """

    _LOCAL_ATTRS = frozenset({"_pool", "_interpreter", "_returned"})

    def __init__(self, pool: "PooledInterpreterFactory", interpreter: CodeInterpreter):
        object.__setattr__(self, "_pool", pool)
        object.__setattr__(self, "_interpreter", interpreter)
        object.__setattr__(self, "_returned", False)

    def _check_active(self) -> None:
        if object.__getattribute__(self, "_returned"):
            raise RuntimeError(
                "This pooled interpreter lease was already shut down; create a new interpreter "
                "from the factory instead of reusing the lease."
            )

    @property
    def tools(self) -> dict[str, Callable[..., Any]]:
        return self._interpreter.tools

    def start(self) -> None:
        self._check_active()
        self._interpreter.start()

    def execute(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        self._check_active()
        return self._interpreter.execute(code, variables)

    def shutdown(self) -> None:
        if object.__getattribute__(self, "_returned"):
            return
        object.__setattr__(self, "_returned", True)
        self._pool._release(self._interpreter)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_interpreter"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._LOCAL_ATTRS:
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_interpreter"), name, value)


class PooledInterpreterFactory:
    """A zero-argument interpreter factory that recycles warm interpreters.

    Wraps another interpreter factory. Each call returns a warm interpreter lease whose
    ``shutdown()`` resets the sandbox and returns the interpreter to the pool, so the usual
    create/execute/shutdown lifecycle in ``dspy.Flex``, ``dspy.RLM``, etc. stops paying the
    interpreter boot cost on every forward.

    Thread-safe: concurrent calls each get their own interpreter. Copies made by
    ``copy.deepcopy`` (e.g., optimizers copying a program) share the same pool.

    Args:
        interpreter_factory: Zero-argument callable creating the underlying interpreters.
        max_idle: Maximum number of warm interpreters kept for reuse; further releases shut
            the interpreter down. ``None`` keeps as many as were ever in use concurrently.

    Example:
        ```python
        factory = dspy.PooledInterpreterFactory(dspy.PythonInterpreter)
        rlm = dspy.RLM("question -> answer", interpreter_factory=factory)
        ```
    """

    def __init__(self, interpreter_factory: Callable[[], CodeInterpreter], max_idle: int | None = None):
        _validate_interpreter_factory(interpreter_factory)
        self._factory = interpreter_factory
        self._max_idle = max_idle
        self._idle: list[CodeInterpreter] = []
        self._lock = threading.Lock()
        # Shut down idle interpreters when the pool is garbage collected or at process exit.
        self._finalizer = weakref.finalize(self, PooledInterpreterFactory._shutdown_idle, self._idle, self._lock)

    def __call__(self) -> CodeInterpreter:
        interpreter = self._acquire()
        if getattr(interpreter, "prepare_for_reuse", None) is None:
            # Not poolable; hand it out as-is so shutdown() really shuts it down.
            return interpreter
        return _PooledInterpreter(self, interpreter)

    def __deepcopy__(self, memo):
        # The pool is a shared resource; program copies keep drawing from the same pool.
        return self

    def __reduce__(self):
        # Serialized programs reconstruct an empty pool around the same inner factory.
        return (PooledInterpreterFactory, (self._factory, self._max_idle))

    def close(self) -> None:
        """Shut down all idle warm interpreters. Leased interpreters are unaffected."""
        PooledInterpreterFactory._shutdown_idle(self._idle, self._lock)

    def _acquire(self) -> CodeInterpreter:
        while True:
            with self._lock:
                if not self._idle:
                    break
                interpreter = self._idle.pop()
            try:
                interpreter.start()  # Cheap liveness check; raises if the process died while idle.
                return interpreter
            except Exception:
                logger.debug("Discarding pooled interpreter that died while idle.", exc_info=True)
                _safe_shutdown(interpreter)

        interpreter = _create_interpreter(self._factory)
        prepare_for_reuse = getattr(interpreter, "prepare_for_reuse", None)
        if prepare_for_reuse is not None:
            # Snapshot the just-booted sandbox before the caller registers tools or runs code,
            # then clear the host-side bindings the snapshot execution created.
            interpreter.start()
            interpreter.execute(_BASELINE_SNAPSHOT_CODE)
            prepare_for_reuse()
        return interpreter

    def _release(self, interpreter: CodeInterpreter) -> None:
        try:
            output = interpreter.execute(_RESET_CODE)
            healthy = isinstance(output, str) and _RESET_OK in output
        except Exception:
            logger.debug("Discarding pooled interpreter that failed its sandbox reset.", exc_info=True)
            healthy = False
        if not healthy:
            _safe_shutdown(interpreter)
            return
        # Only poolable interpreters are ever released, so the hook is present.
        interpreter.prepare_for_reuse()
        with self._lock:
            if self._max_idle is None or len(self._idle) < self._max_idle:
                self._idle.append(interpreter)
                return
        _safe_shutdown(interpreter)

    @staticmethod
    def _shutdown_idle(idle: list[CodeInterpreter], lock: threading.Lock) -> None:
        with lock:
            interpreters, idle[:] = list(idle), []
        for interpreter in interpreters:
            _safe_shutdown(interpreter)


def _safe_shutdown(interpreter: CodeInterpreter) -> None:
    try:
        interpreter.shutdown()
    except Exception:
        logger.debug("Interpreter shutdown raised while discarding it from the pool.", exc_info=True)
