"""Shared pieces for the Terminal-Bench 2.0 Flex+GEPA demo.

Same shape as `tests/demos/conflation/all/conflation_common.py` and
`tests/demos/political-fundraising-emails/n50-all/emails_common.py`: one task signature, fixed
splits, a penalized metric factory, and CAL (cost / accuracy / latency) instrumentation that
persists a per-example record so any other metric can be recomputed offline.

What is different, and why it changes the shape of the experiment
-----------------------------------------------------------------
In the other two demos the `dspy.Flex` module IS the task solver, and GEPA's job is to move work
out of the LLM and into deterministic Python. Here the Flex module is an **agent harness**: it is
handed a task instruction plus three host tools that reach into a live Docker container, and it has
to drive that container to a passing state. GEPA rewrites the harness -- the loop structure, the
sub-predictors and their instructions, how output is summarized, when to stop, whether to verify --
not the task solution. Nothing about any individual Terminal-Bench task is learnable in the harness
source; the 45 test tasks are disjoint from the 44 the optimizer sees.

Three consequences worth stating up front:

* **The label is produced by the environment, not by a matcher.** A task is resolved iff its own
  `tests/test.sh` writes `1` to `/logs/verifier/reward.txt` inside the container. There is no fuzzy
  matching and no judgment call in the metric.
* **The penalty must be normalized.** A terminal episode is tens of LLM calls, not one, so a flat
  per-call penalty at the other demos' scale would drive every score to 0. The metric charges
  `lambda * n_calls / STEP_BUDGET`, so lambda reads as "fraction of the score forfeited by a harness
  that burns its whole step budget" and stays comparable across demos.
* **An episode is expensive and can fail in the middle.** Every rollout starts a container, runs an
  agent for minutes, runs a verifier that apt-installs and pip-installs, and tears the container
  down. `EpisodeCache` therefore persists each finished record keyed by (penalty, program source,
  task), so an interrupted evaluation resumes instead of re-running.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import random
import subprocess
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

from dotenv import load_dotenv

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

load_dotenv()

DEMO_DIR = Path(__file__).parent
DATA_DIR = DEMO_DIR / "tb2_data"
MANIFEST_PATH = DATA_DIR / "tasks.jsonl"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# Unlike the other two demos, the weakest tier is NOT a candidate here. Terminal-Bench 2.0 is hard
# enough that a near-zero baseline would give GEPA no gradient at all: with a resolve rate in the
# single digits, almost every minibatch is uniformly wrong and the reflective dataset carries no
# contrast between what worked and what did not. Sonnet is the cheapest tier that clears the floor.
# `pilot.py` measures this on a fixed 8-task probe rather than assuming it -- run it before trusting
# this default, exactly as the emails demo did for Haiku.
EXEC_MODEL = "anthropic/claude-sonnet-5"
REFLECTION_MODEL = "anthropic/claude-opus-5"

# USD per 1M tokens (input, output). litellm's own per-call cost is recorded alongside as
# `cost_usd_litellm` and is authoritative; this table exists so cost can be attributed to individual
# episodes, which an aggregate history cannot do under a thread pool.
PRICES: dict[str, tuple[float, float]] = {
    "anthropic/claude-haiku-4-5": (1.00, 5.00),
    "anthropic/claude-sonnet-5": (3.00, 15.00),
    "anthropic/claude-opus-5": (5.00, 25.00),
    "anthropic/claude-opus-4-8": (5.00, 25.00),
    "anthropic/claude-opus-4-7": (5.00, 25.00),
}

# An agent step emits a shell command and sometimes a whole file, so 2000 (the emails demo's
# execution cap) truncates real work. 8000 covers every step observed in the pilot.
EXEC_MAX_TOKENS = 8000
# The emails demo lost a full run and $3.99 to silent reflection truncation at 8000. The harness
# source is longer than an extraction module (a loop, several predictors, their instructions), so
# start where that demo ended up. `meter()` counts `finish_reason == "length"` and the sweep warns.
REFLECTION_MAX_TOKENS = 32000

# ---------------------------------------------------------------------------
# Episode budgets
# ---------------------------------------------------------------------------

# The normalizer in the penalty: lambda is charged as `lambda * n_calls / STEP_BUDGET`, so
# lambda = 0.25 means "a harness that spends 30 LLM calls forfeits a quarter of the score".
# It is a unit of account, not a cap -- the cap is MAX_PREDICTOR_CALLS.
STEP_BUDGET = 30
# Hard ceiling on bridged predictor calls per forward, enforced by dspy.Flex itself. A generated
# loop with a broken exit condition would otherwise run up an unbounded bill.
MAX_PREDICTOR_CALLS = 80
# Wall-clock ceiling per episode, in seconds. Terminal-Bench allows 900-1800s per task; that is
# unaffordable across a sweep, so the demo caps it and reports how many episodes hit the cap
# (`deadline_hits` in the summary) -- a resolve rate measured under a tighter budget than the
# official one is not comparable to the public leaderboard, so it must stay visible.
EPISODE_TIMEOUT_S = 600
# Per-command ceiling. The harness may ask for less; it may not ask for more.
MAX_COMMAND_TIMEOUT_S = 300
# Characters of command output handed back to the harness. Beyond this it is head/tail clipped, and
# the harness is told how much was dropped so it can re-run through grep/head instead.
MAX_OUTPUT_CHARS = 6000

DOCKER_LABEL = "dspy_tb2_demo=1"


# ---------------------------------------------------------------------------
# Task manifest
# ---------------------------------------------------------------------------


class TaskSpec(NamedTuple):
    """One Terminal-Bench task, flattened from its `task.toml` by `fetch_data.py`."""

    name: str
    instruction: str
    docker_image: str
    cpus: int
    memory_mb: int
    allow_internet: bool
    agent_timeout_s: float
    verifier_timeout_s: float
    difficulty: str
    category: str
    task_dir: Path

    @property
    def tests_dir(self) -> Path:
        return self.task_dir / "tests"

    @property
    def solution_path(self) -> Path:
        return self.task_dir / "solution" / "solve.sh"


def load_tasks() -> dict[str, TaskSpec]:
    """Read `tb2_data/tasks.jsonl` into name -> TaskSpec."""
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"no manifest at {MANIFEST_PATH}; run `python fetch_data.py` first")
    out: dict[str, TaskSpec] = {}
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[r["name"]] = TaskSpec(
            name=r["name"],
            instruction=r["instruction"],
            docker_image=r["docker_image"],
            cpus=int(r["cpus"]),
            memory_mb=int(r["memory_mb"]),
            allow_internet=bool(r["allow_internet"]),
            agent_timeout_s=float(r["agent_timeout_s"]),
            verifier_timeout_s=float(r["verifier_timeout_s"]),
            difficulty=r["difficulty"],
            category=r["category"],
            task_dir=DATA_DIR / r["task_dir"],
        )
    return out


TASKS: dict[str, TaskSpec] = load_tasks() if MANIFEST_PATH.exists() else {}


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

# Half the benchmark is held out for reporting; the rest is split between GEPA's reflection
# minibatches and its candidate-selection val set. Fractions rather than counts, applied WITHIN each
# difficulty stratum, because the benchmark is lopsided (4 easy / 55 medium / 30 hard) and an
# unstratified shuffle can hand a split zero easy tasks -- which would make resolve rates across
# splits incomparable for reasons that have nothing to do with the harness.
TEST_FRAC, VAL_FRAC = 0.50, 0.22


def _to_example(spec: TaskSpec, source: str) -> dspy.Example:
    # Only `task_name` and `instruction` are inputs. Everything else the runner needs is looked up
    # from TASKS by name, so the Example stays small and JSON-clean in the persisted records.
    return dspy.Example(
        task_name=spec.name,
        instruction=spec.instruction,
        difficulty=spec.difficulty,
        category=spec.category,
        source=source,
    ).with_inputs("task_name", "instruction")


def load_splits(seed: int = 0) -> tuple[list, list, list]:
    """Return (gepa_train, gepa_val, test), stratified by difficulty and disjoint by construction."""
    if not TASKS:
        raise SystemExit(f"no manifest at {MANIFEST_PATH}; run `python fetch_data.py` first")
    by_difficulty: dict[str, list[TaskSpec]] = defaultdict(list)
    for spec in TASKS.values():
        by_difficulty[spec.difficulty].append(spec)

    train, val, test = [], [], []
    for difficulty in sorted(by_difficulty):
        group = sorted(by_difficulty[difficulty], key=lambda s: s.name)
        random.Random(f"{seed}:{difficulty}").shuffle(group)
        n_test = round(TEST_FRAC * len(group))
        n_val = round(VAL_FRAC * len(group))
        test += [_to_example(s, "test") for s in group[:n_test]]
        val += [_to_example(s, "gepa_val") for s in group[n_test : n_test + n_val]]
        train += [_to_example(s, "gepa_train") for s in group[n_test + n_val :]]

    rng = random.Random(seed)
    for split in (train, val, test):
        rng.shuffle(split)
    return train, val, test


# ---------------------------------------------------------------------------
# Docker session layer
# ---------------------------------------------------------------------------


class DockerError(RuntimeError):
    pass


def _docker(args: list[str], timeout: float, stdin: str | None = None) -> subprocess.CompletedProcess:
    """Run a docker command. `stdin` is sent as raw UTF-8 bytes, deliberately.

    `subprocess.run(text=True, input="a\\nb")` writes `a\\r\\nb` on Windows -- Python's text-mode
    stdin wrapper translates newlines to `os.linesep`. Every file the harness wrote into a Linux
    container would have arrived with CRLF endings, which makes shell scripts fail with
    `$'\\r': command not found` and silently corrupts anything line-oriented. Encoding here and
    decoding the output by hand keeps the byte stream exact on every platform.
    """
    proc = subprocess.run(
        ["docker", *args],
        capture_output=True,
        timeout=timeout,
        input=None if stdin is None else stdin.encode("utf-8"),
    )
    return subprocess.CompletedProcess(
        proc.args, proc.returncode,
        proc.stdout.decode("utf-8", "replace"),
        proc.stderr.decode("utf-8", "replace"),
    )


def docker_available() -> tuple[bool, str]:
    """(reachable, detail) for the Docker daemon -- used by preflight and by the sweep's fail-fast."""
    try:
        proc = _docker(["version", "--format", "{{.Server.Version}}"], timeout=30)
    except FileNotFoundError:
        return False, "`docker` is not on PATH"
    except subprocess.TimeoutExpired:
        return False, "`docker version` timed out"
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout).strip().splitlines()[-1:][0] if (proc.stderr or proc.stdout) else "unknown error"
    return True, f"server {proc.stdout.strip()}"


def crlf_tasks(limit: int = 0) -> list[str]:
    """Tasks whose `tests/test.sh` is CRLF on disk -- i.e. whose verifier cannot run in Linux.

    Git's `core.autocrlf=true` (the default on many Windows installs) rewrites every `.sh` in the
    checkout to CRLF. `docker cp` copies bytes, so the container receives a script bash cannot
    parse: `$'\\r': command not found`, then `syntax error: unexpected end of file`. Every task
    fails, the failure looks like the agent's, and the reward is 0 across the board.
    `fetch_data.py --fix-line-endings` repairs it.
    """
    bad = []
    for name, spec in sorted(TASKS.items()):
        script = spec.tests_dir / "test.sh"
        try:
            if script.exists() and b"\r\n" in script.read_bytes():
                bad.append(name)
        except OSError:
            continue
        if limit and len(bad) >= limit:
            break
    return bad


def reap_orphans() -> int:
    """Remove containers this demo left behind (a killed run cannot run its own finally block)."""
    try:
        proc = _docker(["ps", "-aq", "--filter", f"label={DOCKER_LABEL}"], timeout=60)
    except Exception:
        return 0
    ids = [x for x in proc.stdout.split() if x]
    for cid in ids:
        try:
            _docker(["rm", "-f", cid], timeout=120)
        except Exception:
            pass
    return len(ids)


def _clip(text: str, limit: int) -> str:
    """Head+tail clip, telling the reader exactly how much was removed."""
    if len(text) <= limit:
        return text
    head, tail = limit * 2 // 3, limit // 3
    return f"{text[:head]}\n...[{len(text) - limit} characters omitted]...\n{text[-tail:]}"


class ContainerSession:
    """One task's container, plus the transcript of everything the harness did to it.

    Commands run through `docker exec bash -lc`, which is stateless: each call is its own process,
    so `export FOO=1` in one call is invisible to the next. The one piece of shell state a terminal
    agent constantly relies on -- the working directory -- is carried across calls through a state
    file, so `cd src` followed by `ls` behaves as an agent expects. Environment variables, shell
    functions and background job control do NOT persist; see EXPERIMENT.md, this is a deliberate
    simplification of Terminal-Bench's tmux session and it is the demo's largest deviation from the
    official harness.
    """

    CWD_FILE = "/tmp/.dspy_tb2_cwd"

    def __init__(self, spec: TaskSpec, episode_timeout_s: float = EPISODE_TIMEOUT_S):
        self.spec = spec
        self.sid = f"tb2-{uuid.uuid4().hex[:12]}"
        self.container_id: str | None = None
        self.workdir = "/app"
        self.transcript: list[dict[str, Any]] = []
        self.episode_timeout_s = episode_timeout_s
        self.deadline = time.perf_counter() + episode_timeout_s  # reset at the end of start()
        self.deadline_hit = False
        self._has_timeout_bin = True

    # -- lifecycle ----------------------------------------------------------

    def ensure_image(self) -> None:
        if _docker(["image", "inspect", self.spec.docker_image], timeout=120).returncode == 0:
            return
        proc = _docker(["pull", self.spec.docker_image], timeout=1800)
        if proc.returncode != 0:
            raise DockerError(f"docker pull {self.spec.docker_image} failed: {proc.stderr.strip()[-300:]}")

    def start(self) -> None:
        self.ensure_image()
        args = [
            "run", "-d", "--name", self.sid, "--label", DOCKER_LABEL,
            f"--cpus={self.spec.cpus}", f"--memory={self.spec.memory_mb}m",
            # Every TB2 task declares allow_internet=true and the verifier NEEDS it (test.sh
            # apt-installs curl and downloads uv), but honor the flag if a task ever says otherwise.
            *([] if self.spec.allow_internet else ["--network", "none"]),
            "--entrypoint", "", self.spec.docker_image, "sleep", "infinity",
        ]
        proc = _docker(args, timeout=300)
        if proc.returncode != 0:
            raise DockerError(f"docker run failed for {self.spec.name}: {proc.stderr.strip()[-300:]}")
        self.container_id = proc.stdout.strip()

        inspect = _docker(["inspect", "--format", "{{.Config.WorkingDir}}", self.container_id], timeout=60)
        wd = inspect.stdout.strip()
        # test.sh refuses to run from `/`, and the agent should start where the task's files are.
        self.workdir = wd if wd and wd != "/" else "/app"
        # Bootstrap from `/`, not from self.workdir: if the image declared no WORKDIR then /app may
        # not exist yet, and `docker exec -w /app` fails outright rather than creating it.
        self._raw_exec(f"mkdir -p {_sh_quote(self.workdir)} && printf %s {_sh_quote(self.workdir)} "
                       f"> {self.CWD_FILE}", 60, workdir="/")
        self._has_timeout_bin = self._raw_exec("command -v timeout", 30).returncode == 0
        # Only now does the agent's clock start. An image pull or a slow container start would
        # otherwise be charged to the harness, making the wall-clock cap depend on whether the image
        # happened to be in the local cache -- which is exactly the kind of hidden variable that
        # makes one lambda look better than another for no real reason.
        self.deadline = time.perf_counter() + self.episode_timeout_s

    def stop(self) -> None:
        if self.container_id:
            try:
                _docker(["rm", "-f", self.container_id], timeout=180)
            except Exception:
                pass
            self.container_id = None

    # -- execution ----------------------------------------------------------

    def _raw_exec(self, script: str, timeout: float, workdir: str | None = None):
        return _docker(
            ["exec", "-w", workdir or self.workdir, self.container_id or "", "bash", "-lc", script],
            timeout=timeout,
        )

    def remaining_s(self) -> float:
        return self.deadline - time.perf_counter()

    def exec(self, command: str, timeout_s: int = 60) -> str:
        """Run one command; return a compact `exit_code / stdout / stderr` rendering."""
        if not self.container_id:
            return "ERROR: container is not running."
        remaining = self.remaining_s()
        if remaining <= 0:
            self.deadline_hit = True
            return ("ERROR: the episode wall-clock budget is exhausted. Stop issuing commands and "
                    "return your summary now.")
        budget = int(max(1, min(int(timeout_s or 60), MAX_COMMAND_TIMEOUT_S, remaining)))

        # One shell does everything, in this order: restore the last working directory, run the
        # command, record the new working directory, exit with the command's status. It has to be
        # ONE shell -- a `cd` inside a nested `bash -c` would not be visible to a `pwd` run by the
        # parent, so the directory would never actually persist.
        script = (
            f'cd "$(cat {self.CWD_FILE} 2>/dev/null || echo {self.workdir})" 2>/dev/null || cd {self.workdir}\n'
            f"{command}\n"
            f"__dspy_rc=$?\n"
            f"pwd > {self.CWD_FILE} 2>/dev/null\n"
            f"exit $__dspy_rc\n"
        )
        # Kill it container-side: a client-side timeout only kills the local `docker exec` process
        # and leaves the real work running inside the container, where it competes with the next
        # command for the task's 1 CPU. The client-side timeout below stays as a backstop.
        if self._has_timeout_bin:
            script = f"timeout -k 5 {budget}s bash -lc {_sh_quote(script)}"

        started = time.perf_counter()
        try:
            proc = self._raw_exec(script, timeout=budget + 25)
            rc, out, err, timed_out = proc.returncode, proc.stdout, proc.stderr, False
        except subprocess.TimeoutExpired as exc:
            rc, timed_out = 124, True
            out = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            err = (exc.stderr or b"").decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        elapsed = time.perf_counter() - started

        if rc == 124:
            timed_out = True
        self.transcript.append({
            "command": command[:400],
            "exit_code": rc,
            "duration_s": round(elapsed, 2),
            "timed_out": timed_out,
            "output_chars": len(out) + len(err),
            "tail": _clip((out + err).strip(), 300),
        })

        parts = [f"exit_code={rc}  ({elapsed:.1f}s elapsed, {self.remaining_s():.0f}s left in episode)"]
        if timed_out:
            parts.append(f"NOTE: the command was killed after its {budget}s timeout.")
        if out.strip():
            parts.append("--- stdout ---\n" + _clip(out.rstrip(), MAX_OUTPUT_CHARS))
        if err.strip():
            parts.append("--- stderr ---\n" + _clip(err.rstrip(), MAX_OUTPUT_CHARS // 2))
        if not out.strip() and not err.strip():
            parts.append("(no output)")
        return "\n".join(parts)

    def write(self, path: str, content: str) -> str:
        if not self.container_id:
            return "ERROR: container is not running."
        if self.remaining_s() <= 0:
            self.deadline_hit = True
            return "ERROR: the episode wall-clock budget is exhausted."
        # Content arrives on stdin and the path arrives as a positional argument, so neither is ever
        # parsed by a shell -- a heredoc would break on any content containing its own delimiter,
        # backticks or `$(`, which is exactly the content an agent writes most often (scripts).
        proc = _docker(
            ["exec", "-i", "-w", self.workdir, self.container_id, "bash", "-c",
             'mkdir -p "$(dirname "$1")" && cat > "$1"', "_", path],
            timeout=120, stdin=content,
        )
        self.transcript.append({
            "command": f"<write {len(content)} bytes to {path}>",
            "exit_code": proc.returncode, "duration_s": 0.0, "timed_out": False,
            "output_chars": 0, "tail": (proc.stderr or "")[:200],
        })
        if proc.returncode != 0:
            return f"ERROR writing {path}: {(proc.stderr or '').strip()[:300]}"
        return f"wrote {len(content)} bytes to {path}"

    def read(self, path: str, max_bytes: int = 20000) -> str:
        if not self.container_id:
            return "ERROR: container is not running."
        proc = _docker(["exec", "-w", self.workdir, self.container_id, "bash", "-c",
                        'head -c "$2" "$1"', "_", path, str(int(max_bytes))], timeout=120)
        if proc.returncode != 0:
            return f"ERROR reading {path}: {(proc.stderr or '').strip()[:300]}"
        return _clip(proc.stdout, MAX_OUTPUT_CHARS)

    # -- verification -------------------------------------------------------

    def verify(self) -> tuple[bool, str, float]:
        """Run the task's own verifier. Returns (resolved, output tail, wall seconds).

        This is Terminal-Bench's contract, unchanged: copy `tests/` to `/tests`, run
        `bash /tests/test.sh`, and read the reward the script writes to
        `/logs/verifier/reward.txt`. The tests are copied in only now -- the agent never saw them.
        """
        started = time.perf_counter()
        if not self.container_id:
            return False, "container was not running at verification time", 0.0
        try:
            self._raw_exec("mkdir -p /logs/verifier /tests", 60, workdir="/")
            cp = _docker(["cp", os.path.join(str(self.spec.tests_dir), "."),
                          f"{self.container_id}:/tests/"], timeout=300)
            if cp.returncode != 0:
                return False, f"docker cp of tests failed: {cp.stderr.strip()[-300:]}", time.perf_counter() - started
            proc = self._raw_exec("bash /tests/test.sh 2>&1", timeout=self.spec.verifier_timeout_s + 60)
            output = proc.stdout + proc.stderr
            reward = self._raw_exec("cat /logs/verifier/reward.txt 2>/dev/null", 60)
            resolved = reward.stdout.strip().startswith("1")
        except subprocess.TimeoutExpired:
            return False, f"verifier exceeded {self.spec.verifier_timeout_s:.0f}s", time.perf_counter() - started
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"[:300], time.perf_counter() - started
        return resolved, _clip(output.strip(), 4000), time.perf_counter() - started

    # -- reporting ----------------------------------------------------------

    def digest(self, max_commands: int = 18) -> str:
        """A compact rendering of what the harness did -- the core of GEPA's feedback signal."""
        if not self.transcript:
            return "(the harness ran no commands at all)"
        shown = self.transcript[-max_commands:]
        skipped = len(self.transcript) - len(shown)
        lines = [f"[{skipped} earlier command(s) omitted]"] if skipped else []
        for i, t in enumerate(shown, start=skipped + 1):
            flag = " TIMED-OUT" if t["timed_out"] else ""
            lines.append(f"{i}. $ {t['command']}\n   -> exit={t['exit_code']}{flag} "
                         f"({t['duration_s']}s, {t['output_chars']} chars)"
                         + (f"\n   {t['tail'][:200]}" if t["tail"] else ""))
        return "\n".join(lines)

    @property
    def n_commands(self) -> int:
        return len(self.transcript)

    @property
    def n_failed_commands(self) -> int:
        return sum(1 for t in self.transcript if t["exit_code"] != 0)


def _sh_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


# ---------------------------------------------------------------------------
# Host tools handed to dspy.Flex
# ---------------------------------------------------------------------------

# The generated harness runs in the Flex sandbox and cannot touch Docker directly. These three
# functions are registered as host tools: the sandbox calls them by name with keyword arguments,
# and they resolve the opaque `session` string to a live ContainerSession here in the host process.
# Keeping the registry here (rather than passing a handle) is what lets the harness code stay a pure,
# serializable string that GEPA can rewrite.
_SESSIONS: dict[str, ContainerSession] = {}
_SESSIONS_LOCK = threading.Lock()


def _register(session: ContainerSession) -> None:
    with _SESSIONS_LOCK:
        _SESSIONS[session.sid] = session


def _unregister(session: ContainerSession) -> None:
    with _SESSIONS_LOCK:
        _SESSIONS.pop(session.sid, None)


def _resolve(session: str) -> ContainerSession | None:
    with _SESSIONS_LOCK:
        return _SESSIONS.get(session)


def terminal_exec(session: str, command: str, timeout_s: int = 60) -> str:
    """Run one bash command in the task's Linux container; returns its exit code, stdout and stderr.

    `session` is the session id from the module's inputs. The working directory persists between
    calls; exported variables do not. Long output is clipped, so filter with grep/head when you
    expect a lot. Commands are killed at `timeout_s` (max 300).
    """
    sess = _resolve(session)
    if sess is None:
        return f"ERROR: unknown session {session!r}."
    return sess.exec(command, timeout_s=timeout_s)


def write_file(session: str, path: str, content: str) -> str:
    """Write `content` to `path` in the container, creating parent directories.

    Prefer this over `echo`/heredoc for anything multi-line or containing quotes, backslashes or
    `$` -- the content is streamed on stdin and is never parsed by a shell.
    """
    sess = _resolve(session)
    if sess is None:
        return f"ERROR: unknown session {session!r}."
    return sess.write(path, content)


def read_file(session: str, path: str, max_bytes: int = 20000) -> str:
    """Return the first `max_bytes` bytes of a file in the container."""
    sess = _resolve(session)
    if sess is None:
        return f"ERROR: unknown session {session!r}."
    return sess.read(path, max_bytes=max_bytes)


HARNESS_TOOLS = [terminal_exec, write_file, read_file]


# ---------------------------------------------------------------------------
# Signature + the program under optimization
# ---------------------------------------------------------------------------


class SolveTerminalTask(dspy.Signature):
    """Complete a task inside a Linux container by running shell commands, then report what you did."""

    # Deliberately minimal, for the reason the emails demo learned the hard way: anything resembling
    # strategy in the seed docstring is task-solving information the optimizer should have had to
    # discover, and it lands in the EXECUTION prompt where it also inflates the baseline. The field
    # descriptions below are mechanical API facts (what the handle is for), not strategy.
    instruction: str = dspy.InputField(desc="What the user wants done inside the container.")
    session: str = dspy.InputField(
        desc="Opaque session id. Pass it unchanged as the `session=` argument of every tool call."
    )
    summary: str = dspy.OutputField(desc="What was done, and where the result was left.")


class TerminalAgent(dspy.Module):
    """Runs one episode end to end: start container -> run the Flex harness -> verify -> tear down.

    Only `self.harness` is code-optimizable, so this wrapper is what keeps GEPA honest. The reward
    is produced by the task's own verifier *after* the harness has returned and can no longer act,
    and the tests are copied into the container only at that point -- so no harness, however it is
    rewritten, can read or edit the thing that grades it.
    """

    def __init__(self, harness: dspy.Module | None = None, episode_timeout_s: float = EPISODE_TIMEOUT_S):
        super().__init__()
        self.harness = harness if harness is not None else new_harness()
        self.episode_timeout_s = episode_timeout_s

    def forward(self, task_name: str, instruction: str) -> dspy.Prediction:
        spec = TASKS[task_name]
        session = ContainerSession(spec, episode_timeout_s=min(self.episode_timeout_s, spec.agent_timeout_s))
        summary, harness_error = "", None
        agent_wall_s = 0.0
        try:
            session.start()
            _register(session)
            started = time.perf_counter()
            try:
                out = self.harness(instruction=instruction, session=session.sid)
                summary = str(getattr(out, "summary", "") or "")
            except Exception as exc:
                # A harness crash is a harness failure, not a lost episode: verify anyway, because a
                # crash after the work was already done should still count as resolved.
                harness_error = f"{type(exc).__name__}: {exc}"[:400]
            agent_wall_s = time.perf_counter() - started
            resolved, verifier_tail, verify_wall_s = session.verify()
        except Exception as exc:
            return dspy.Prediction(
                resolved=False, summary=summary, n_commands=0, n_failed_commands=0,
                transcript_digest="(the container never started)",
                verifier_tail="", harness_error=f"{type(exc).__name__}: {exc}"[:400],
                deadline_hit=False, agent_wall_s=agent_wall_s, verify_wall_s=0.0,
                infra_error=True,
            )
        finally:
            _unregister(session)
            session.stop()

        return dspy.Prediction(
            resolved=bool(resolved),
            summary=summary,
            n_commands=session.n_commands,
            n_failed_commands=session.n_failed_commands,
            transcript_digest=session.digest(),
            verifier_tail=verifier_tail,
            harness_error=harness_error,
            deadline_hit=session.deadline_hit,
            agent_wall_s=agent_wall_s,
            verify_wall_s=verify_wall_s,
            infra_error=False,
        )


def new_harness() -> dspy.Flex:
    """A fresh, un-optimized harness.

    `dspy.Flex` with tools seeds itself as a single `dspy.RLM` over the signature -- a code-REPL
    agent that can call the three container tools. That IS the baseline: what you get from
    `dspy.Flex(sig, tools=...)` before any optimization. GEPA's job is to turn it into a harness.
    """
    return dspy.Flex(SolveTerminalTask, tools=HARNESS_TOOLS, max_predictor_calls=MAX_PREDICTOR_CALLS)


def make_lms(exec_model: str = EXEC_MODEL,
             exec_max_tokens: int = EXEC_MAX_TOKENS,
             reflection_max_tokens: int = REFLECTION_MAX_TOKENS) -> tuple[dspy.LM, dspy.LM]:
    """Return (execution LM, reflection LM) and make history unbounded for cost accounting."""
    # `meter()` slices lm.history by index; the default 10k cap pops from the FRONT, which would
    # shift those indices mid-run and silently drop calls from the totals. A terminal sweep makes
    # far more calls than the other demos, so this is not hypothetical here.
    dspy.configure(max_history_size=10**9)
    return (dspy.LM(exec_model, max_tokens=exec_max_tokens),
            dspy.LM(REFLECTION_MODEL, max_tokens=reflection_max_tokens))


def disable_cache() -> None:
    """Turn off dspy's caches so latency and cost are what a cold production call would be."""
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


def make_metric(llm_call_penalty: float, step_budget: int = STEP_BUDGET):
    """Build the GEPA metric for one penalty: reward a passing verifier, charge per LLM call.

    Score is `max(0, resolved - lambda * n_calls / step_budget)`. The feedback names the objective
    and then shows the harness what it actually did -- the command transcript with exit codes, the
    verifier's own output, whether it ran out of wall clock -- because a harness can only be
    improved from evidence about its behavior, not from a scalar.

    Note on contamination: the verifier tail quoted below contains pytest output from the task's
    hidden tests. GEPA only ever sees this for train and val tasks, never for the 45 test tasks, so
    it is a training reward signal in the ordinary sense. It is still real information about those
    44 tasks, which is why the harness source is inspected for task-specific literals in
    EXPERIMENT.md section 3.
    """

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None, program_trace=None) -> ScoreWithFeedback:
        exec_trace = program_trace if program_trace is not None else trace
        n_calls = len(exec_trace) if exec_trace else 0
        charge = llm_call_penalty * n_calls / step_budget

        resolved = bool(getattr(pred, "resolved", False))
        score = max(0.0, (1.0 if resolved else 0.0) - charge)

        n_commands = int(getattr(pred, "n_commands", 0) or 0)
        n_failed = int(getattr(pred, "n_failed_commands", 0) or 0)
        digest = str(getattr(pred, "transcript_digest", "") or "")
        verifier_tail = str(getattr(pred, "verifier_tail", "") or "")
        harness_error = getattr(pred, "harness_error", None)
        deadline_hit = bool(getattr(pred, "deadline_hit", False))

        if getattr(pred, "infra_error", False):
            # Docker could not give us a container. Nothing about the harness caused this and
            # nothing about the harness can fix it, so say so instead of teaching from noise.
            return ScoreWithFeedback(
                score=0.0,
                feedback=("INFRASTRUCTURE FAILURE, not a harness failure: the container never "
                          f"started ({harness_error}). Ignore this example; it says nothing about "
                          "the harness code."),
            )

        cost = (f"{n_calls} LLM call(s) = {charge:.3f} of the score, {n_commands} shell command(s) "
                f"({n_failed} nonzero exit), final score {score:.3f}")

        if llm_call_penalty == 0.0:
            goal = ("LLM calls are free under the current objective, so solving the task is all that "
                    "matters. Use whatever number of steps maximizes the chance the verifier passes.")
        else:
            goal = (
                f"Target: get the verifier to pass while spending as few LLM calls as the task "
                f"allows (each call costs {llm_call_penalty / step_budget:.4f} of the score). Shell "
                f"commands are FREE -- only model calls are charged -- so batch reconnaissance, "
                f"parsing, retries and checking your own work into plain Python and shell rather "
                f"than into another round of model reasoning. How to structure the loop, what to "
                f"look at first, and when to stop is for you to work out."
            )

        head: str
        if harness_error and not resolved:
            head = (f"The harness RAISED before finishing ({cost}). Error: {harness_error}. A harness "
                    f"must never crash -- catch tool errors, check the strings tools return, and "
                    f"always return dspy.Prediction(summary=<str>).")
        elif deadline_hit and not resolved:
            head = (f"FAILED by running out of wall clock ({cost}). The episode budget was exhausted "
                    f"while commands were still being issued, so the run was cut off mid-task. Budget "
                    f"the time: do cheap, broad reconnaissance first, and do not spend long commands "
                    f"on a path you have not confirmed.")
        elif not resolved and n_commands == 0:
            head = (f"FAILED having run NO commands at all ({cost}) -- the worst outcome, because it "
                    f"spent model calls without touching the container. The harness must actually "
                    f"call terminal_exec/write_file with session=<the session input>.")
        elif not resolved:
            head = (f"FAILED: the task's verifier returned 0 ({cost}). The work in the container did "
                    f"not satisfy the task's hidden tests.")
        elif charge > 0:
            head = (f"SOLVED, but at a cost ({cost}). Full credit needs this same result with fewer "
                    f"model calls.")
        else:
            head = f"SOLVED ({cost}). This is the target -- keep whatever produced it."

        parts = [head, "", "What the harness did, in order:", digest]
        if verifier_tail:
            parts += ["", "What the verifier said (the agent never sees this during the episode):",
                      _clip(verifier_tail, 1800)]
        parts += ["", goal]
        return ScoreWithFeedback(score=score, feedback="\n".join(parts))

    return metric


def make_resolve_metric(step_budget: int = STEP_BUDGET):
    """Score = did the task's verifier pass. Nothing else is charged.

    This is the CAPABILITY objective, as opposed to `make_metric`'s cost objective. There is no
    penalty term and no reference to LLM calls in the goal, because the question here is not "can
    the harness do this more cheaply" but "what is the harness missing that would let it solve more
    tasks". Those pull in opposite directions: the cost objective rewards *thinking less*, and a
    terminal agent that thinks less mostly gives up earlier.

    The feedback is therefore restructured around diagnosis rather than accounting. It still shows
    the transcript and the verifier output -- a harness can only be improved from evidence -- but the
    directive asks the optimizer to name the missing capability and build it into the harness as a
    reusable part, instead of asking it to spend fewer calls.

    What the optimizer can actually build (it rewrites the harness module, nothing else):
      * reusable Python helpers composed over the three container tools;
      * additional sub-predictors with their own signatures, for sub-problems worth isolating;
      * loop structure, retries, error handling, and a self-check before returning.
    It cannot add new host tools, and it can never see or touch the verifier.

    `step_budget` is unused and kept only so the two factories share a signature; the sweep driver
    selects between them.
    """

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None, program_trace=None) -> ScoreWithFeedback:
        exec_trace = program_trace if program_trace is not None else trace
        n_calls = len(exec_trace) if exec_trace else 0

        resolved = bool(getattr(pred, "resolved", False))
        score = 1.0 if resolved else 0.0

        n_commands = int(getattr(pred, "n_commands", 0) or 0)
        n_failed = int(getattr(pred, "n_failed_commands", 0) or 0)
        digest = str(getattr(pred, "transcript_digest", "") or "")
        verifier_tail = str(getattr(pred, "verifier_tail", "") or "")
        harness_error = getattr(pred, "harness_error", None)
        deadline_hit = bool(getattr(pred, "deadline_hit", False))

        if getattr(pred, "infra_error", False):
            return ScoreWithFeedback(
                score=0.0,
                feedback=("INFRASTRUCTURE FAILURE, not a harness failure: the container never "
                          f"started ({harness_error}). Ignore this example; it says nothing about "
                          "the harness code."),
            )

        # Reported as context. Not scored -- but not free either, which the goal text is careful
        # about: calling them "free" invites loops that burn the episode clock and then miss the
        # deadline, which lowers the resolve rate this metric exists to raise.
        budget = (f"{n_calls} model call(s), {n_commands} shell command(s) ({n_failed} nonzero exit)"
                  f" -- not counted against the score; the verifier alone decides it")

        goal = (
            "OBJECTIVE: solve more tasks. The verifier alone decides the score: model calls and "
            "shell commands are not counted against you, so do not optimise for using fewer. They "
            "are not unlimited either -- the episode has a wall-clock budget and a hard ceiling of "
            f"{MAX_PREDICTOR_CALLS} model calls -- so spend them where they buy information or "
            "progress, and not on repeating something that already failed.\n"
            "When an episode fails, the useful question is what CAPABILITY the harness lacked. "
            "Diagnose the gap and build the fix into the harness so "
            "it generalises to tasks you have not seen:\n"
            "  - turn anything you find yourself doing ad hoc into a reusable helper over the three "
            "tools (reconnaissance, locating files, running a command and checking its exit code, "
            "installing a missing dependency, applying an edit and confirming it took);\n"
            "  - isolate a hard sub-problem into its own sub-predictor with its own instructions "
            "when one round of reasoning is not enough;\n"
            "  - check your own work before returning -- run the thing you just built, read the "
            "error, and fix it, rather than reporting success unverified;\n"
            "  - make failures recoverable: read what a command actually printed, and try a "
            "different approach instead of repeating one that already failed.\n"
            "What to look at first, how to structure the loop, and when to stop is for you to work "
            "out from the transcripts."
        )

        head: str
        if harness_error and not resolved:
            head = (f"The harness RAISED before finishing ({budget}). Error: {harness_error}. This is "
                    f"a bug in the harness, and the most straightforward kind to remove: catch "
                    f"tool errors, check the strings the tools return, and always return "
                    f"dspy.Prediction(summary=<str>).")
        elif deadline_hit and not resolved:
            head = (f"FAILED by running out of wall clock ({budget}). Time, not model calls, is the "
                    f"binding resource: the episode was cut off mid-task. Reach a working state "
                    f"early and improve it, rather than exploring for a long time and committing "
                    f"late.")
        elif not resolved and n_commands == 0:
            head = (f"FAILED having run NO commands at all ({budget}) -- the worst outcome, because "
                    f"nothing was attempted in the container. The harness must actually call "
                    f"terminal_exec/write_file with session=<the session input>.")
        elif not resolved:
            head = (f"FAILED: the task's verifier returned 0 ({budget}). Work happened, but it did "
                    f"not satisfy the hidden tests. Compare what the verifier below checked against "
                    f"what the transcript actually did -- the gap between them is the capability to "
                    f"add.")
        else:
            head = (f"SOLVED ({budget}). Keep whatever produced this. If it relied on something ad "
                    f"hoc, make it a reusable part of the harness so it survives on other tasks.")

        parts = [head, "", "What the harness did, in order:", digest]
        if verifier_tail:
            parts += ["", "What the verifier said (the harness never sees this during the episode):",
                      _clip(verifier_tail, 1800)]
        parts += ["", goal]
        return ScoreWithFeedback(score=score, feedback="\n".join(parts))

    return metric


# ---------------------------------------------------------------------------
# Cost accounting
# ---------------------------------------------------------------------------


@contextmanager
def meter(*lms: dspy.LM):
    """Accumulate calls / tokens / litellm cost across `lms` for the duration of the block.

    Snapshots each LM's history index on entry and reads the tail on exit. `history` is append-only
    (see `make_lms`, which lifts the trim cap) and list.append is atomic under the GIL, so this is
    safe across the evaluation thread pool.
    """
    totals: dict[str, Any] = {
        "calls": 0, "cost_usd_litellm": 0.0,
        "prompt_tokens": 0, "completion_tokens": 0, "uncosted_calls": 0,
        # A truncated reflection emits half a Python class, which fails to bind and scores 0
        # everywhere -- indistinguishable from "the optimizer found nothing" unless you count it.
        # That cost the emails demo $3.99 and a whole run before it was noticed.
        "truncated_calls": 0, "max_completion_tokens_seen": 0,
    }
    starts = [len(lm.history) for lm in lms]
    yield totals
    for lm, start in zip(lms, starts, strict=True):
        for entry in lm.history[start:]:
            totals["calls"] += 1
            cost = entry.get("cost")
            if cost is None:
                totals["uncosted_calls"] += 1
            else:
                totals["cost_usd_litellm"] += cost
            usage = entry.get("usage") or {}
            totals["prompt_tokens"] += usage.get("prompt_tokens") or 0
            completion = usage.get("completion_tokens") or 0
            totals["completion_tokens"] += completion
            totals["max_completion_tokens_seen"] = max(totals["max_completion_tokens_seen"], completion)
            if _hit_length_cap(entry):
                totals["truncated_calls"] += 1


def _hit_length_cap(entry: dict) -> bool:
    """True if this call stopped because it ran out of max_tokens rather than finishing."""
    response = entry.get("response")
    if response is None:
        return False
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    for choice in choices or []:
        reason = getattr(choice, "finish_reason", None)
        if reason is None and isinstance(choice, dict):
            reason = choice.get("finish_reason")
        if reason in ("length", "max_tokens"):
            return True
    return False


def price(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rate_in, rate_out = PRICES.get(model, (0.0, 0.0))
    return (prompt_tokens * rate_in + completion_tokens * rate_out) / 1e6


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class Record(NamedTuple):
    """One episode's outcome -- enough to recompute any penalty's score offline."""

    task: str
    difficulty: str
    category: str
    resolved: bool
    n_calls: int
    n_commands: int
    n_failed_commands: int
    deadline_hit: bool
    latency_s: float        # agent wall clock: harness reasoning + every container command
    verify_s: float
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int
    summary: str
    verifier_tail: str
    error: str | None


class EpisodeCache:
    """Append-only JSONL of finished episodes, so a killed evaluation resumes where it stopped.

    Keyed by (penalty, harness source hash, task). The source hash is what makes this safe: change
    the harness by one character and every key changes, so a cached record can never be attributed
    to a program that did not produce it. Episodes are stochastic, so this is resumption, not
    memoization -- `--no-episode-cache` turns it off for a clean re-measurement.
    """

    def __init__(self, path: Path | None):
        self.path = path
        self._lock = threading.Lock()
        self._hits: dict[str, dict] = {}
        if path and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._hits[row["key"]] = row["record"]

    @staticmethod
    def key(penalty: float, program_src: str, task: str) -> str:
        digest = hashlib.sha256((program_src or "").encode("utf-8")).hexdigest()[:16]
        return f"{penalty:g}|{digest}|{task}"

    def get(self, key: str) -> Record | None:
        row = self._hits.get(key)
        return Record(**row) if row else None

    def put(self, key: str, record: Record) -> None:
        if not self.path:
            return
        with self._lock:
            self._hits[key] = record._asdict()
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "record": record._asdict()}, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._hits)


def run_program(program: dspy.Module, dataset: list, threads: int = 4,
                penalty: float = 0.0, cache: EpisodeCache | None = None,
                progress: bool = True) -> tuple[list[Record], dict]:
    """Run `program` over `dataset` one container at a time per thread, returning per-episode records."""
    src = getattr(getattr(program, "harness", None), "module_src", "") or ""

    def run_one(ex) -> Record:
        key = EpisodeCache.key(penalty, src, ex.task_name)
        if cache is not None:
            hit = cache.get(key)
            if hit is not None:
                return hit

        started = time.perf_counter()
        try:
            with dspy.context(trace=[]), dspy.track_usage() as usage:
                pred = program(**ex.inputs())
                trace = list(dspy.settings.trace or [])
            p_tok = c_tok = 0
            cost = 0.0
            for model, totals in usage.get_total_tokens().items():
                p = totals.get("prompt_tokens") or 0
                c = totals.get("completion_tokens") or 0
                p_tok += p
                c_tok += c
                cost += price(model, p, c)
            record = Record(
                task=ex.task_name, difficulty=ex.difficulty, category=ex.category,
                resolved=bool(pred.resolved), n_calls=len(trace),
                n_commands=int(pred.n_commands or 0),
                n_failed_commands=int(pred.n_failed_commands or 0),
                deadline_hit=bool(pred.deadline_hit),
                latency_s=float(pred.agent_wall_s or 0.0), verify_s=float(pred.verify_wall_s or 0.0),
                cost_usd=cost, prompt_tokens=p_tok, completion_tokens=c_tok,
                summary=_clip(str(pred.summary or ""), 1200),
                verifier_tail=_clip(str(pred.verifier_tail or ""), 2500),
                error=pred.harness_error,
            )
        except Exception as exc:
            # Counts as unresolved. `error` is surfaced so a run that silently fails every episode --
            # a stopped Docker daemon, a bad model id -- is visible in the JSON instead of looking
            # like a 0% resolve rate.
            record = Record(ex.task_name, ex.difficulty, ex.category, False, 0, 0, 0, False,
                            time.perf_counter() - started, 0.0, 0.0, 0, 0, "", "",
                            f"{type(exc).__name__}: {exc}"[:300])
        if cache is not None:
            cache.put(key, record)
        if progress:
            print(f"    [{'PASS' if record.resolved else 'fail'}] {record.task:<34} "
                  f"calls={record.n_calls:<3} cmds={record.n_commands:<3} "
                  f"{record.latency_s:6.0f}s ${record.cost_usd:.3f}"
                  + (f"  !! {record.error}" if record.error else ""), flush=True)
        return record

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        records = list(pool.map(run_one, dataset))
    return records, {"wall_s": time.perf_counter() - started, "threads": threads}


def summarize(records: list[Record], penalty: float, step_budget: int = STEP_BUDGET) -> dict:
    """Resolve rate + the three CAL axes, for one penalty, from per-episode records."""
    n = len(records) or 1
    lat = sorted(r.latency_s for r in records)
    resolved = sum(1 for r in records if r.resolved)
    score = sum(max(0.0, (1.0 if r.resolved else 0.0) - penalty * r.n_calls / step_budget)
                for r in records) / n

    def pct(p: float) -> float:
        return lat[min(len(lat) - 1, round(p * (len(lat) - 1)))] if lat else 0.0

    def group(field: str) -> dict:
        out = {}
        for value in sorted({getattr(r, field) for r in records}):
            sub = [r for r in records if getattr(r, field) == value]
            out[value] = {
                "n": len(sub),
                "resolve_rate": sum(1 for r in sub if r.resolved) / len(sub),
                "avg_calls": sum(r.n_calls for r in sub) / len(sub),
                "cost_usd_per_task": sum(r.cost_usd for r in sub) / len(sub),
            }
        return out

    return {
        "n": len(records),
        "penalty": penalty,
        "step_budget": step_budget,
        "score": score,
        "resolve_rate": resolved / n,
        "n_resolved": resolved,
        "avg_calls": sum(r.n_calls for r in records) / n,
        "calls_total": sum(r.n_calls for r in records),
        "avg_commands": sum(r.n_commands for r in records) / n,
        "avg_failed_commands": sum(r.n_failed_commands for r in records) / n,
        "frac_episodes_using_llm": sum(1 for r in records if r.n_calls > 0) / n,
        "cost_usd_total": sum(r.cost_usd for r in records),
        "cost_usd_per_task": sum(r.cost_usd for r in records) / n,
        "cost_usd_per_100_tasks": 100 * sum(r.cost_usd for r in records) / n,
        "latency_mean_s": sum(lat) / n,
        "latency_p50_s": pct(0.50),
        "latency_p95_s": pct(0.95),
        "verify_mean_s": sum(r.verify_s for r in records) / n,
        "prompt_tokens": sum(r.prompt_tokens for r in records),
        "completion_tokens": sum(r.completion_tokens for r in records),
        "deadline_hits": sum(1 for r in records if r.deadline_hit),
        "errors": sum(1 for r in records if r.error),
        "first_error": next((r.error for r in records if r.error), None),
        "by_difficulty": group("difficulty"),
        "by_category": group("category"),
    }


def fmt(row: dict) -> str:
    return (
        f"score={row['score']:.3f} resolved={row['n_resolved']}/{row['n']} "
        f"({row['resolve_rate']:.3f}) calls/task={row['avg_calls']:.1f} "
        f"cmds/task={row['avg_commands']:.1f} ${row['cost_usd_per_task']:.2f}/task "
        f"lat_mean={row['latency_mean_s']:.0f}s deadline_hits={row['deadline_hits']}"
    )


atexit.register(reap_orphans)
