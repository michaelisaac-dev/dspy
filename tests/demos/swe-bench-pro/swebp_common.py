"""Shared pieces for the SWE-bench Pro Flex+GEPA demo.

Fourth in the series after conflation (classification), political emails (extraction) and
terminal-bench-2 (agent harness over containerized terminal tasks). The object under optimization
is the same as TB2's: `dspy.Flex` is an **agent harness**, and GEPA rewrites the harness while the
tasks stay untouched. What changes is the task family and the verifier: each episode is a real
GitHub issue from SWE-bench Pro (public set), solved inside that instance's own prebuilt Docker
image, and graded by the benchmark's own per-instance test scripts.

The question this demo exists to answer (TB2's EXPERIMENT.md §11.4): does an evolved harness let a
*small* execution model resolve more issues than the same model under the un-optimized harness?

Verifier contract (mirrors scaleapi/SWE-bench_Pro-os `swe_bench_pro_eval.py` exactly):
  image  = jefzda/sweap-images:{dockerhub_tag}
  script = export ENVs from the instance's dockerfiles; cd /app; git reset --hard {base_commit};
           git checkout {base_commit}; git apply -v /workspace/patch.diff;
           {last line of before_repo_set_cmd};           # brings in the updated test files
           bash /workspace/run_script.sh {selected_test_files} > stdout.log 2> stderr.log;
           python /workspace/parser.py stdout.log stderr.log output.json
  resolved = (fail_to_pass ∪ pass_to_pass) ⊆ {t.name for t in output.json if t.status == "PASSED"}

The agent NEVER runs in the verification container. It works in its own container (same image,
repo reset to base_commit), its edits are extracted as a git patch when the episode ends, and the
patch is verified afterwards in a fresh container. The agent cannot see run_script.sh, parser.py,
the test lists, or the updated test files, and it cannot touch the thing that grades it.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import platform as py_platform
import re
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

from dotenv import load_dotenv

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

load_dotenv()

DEMO_DIR = Path(__file__).parent
DATA_DIR = DEMO_DIR / "swebp_data"
MANIFEST = DATA_DIR / "manifest.jsonl"
UPSTREAM = DEMO_DIR / "swebp_upstream"          # clone of scaleapi/SWE-bench_Pro-os
RUN_SCRIPTS = UPSTREAM / "run_scripts"
DOCKERFILES = UPSTREAM / "dockerfiles"

# ---------------------------------------------------------------------------
# Models and knobs
# ---------------------------------------------------------------------------

# The premise of this experiment is a SMALL execution model: the question is whether an evolved
# harness buys back capability that the model alone lacks. Haiku 4.5 is the weakest and cheapest
# Claude. Reflection stays on Opus 5 -- it writes the harness code, and reflection cost amortizes
# across all episodes while execution cost is paid per episode.
EXEC_MODEL = "anthropic/claude-haiku-4-5"
REFLECTION_MODEL = "anthropic/claude-opus-5"
EXEC_MAX_TOKENS = 8000
# 8k reflection tokens silently truncated GEPA's proposals in the emails demo (finish_reason ==
# "length" on ~7.6k-token candidates) and every truncated candidate failed to bind. Harness
# classes are long; 32k gives headroom, and `meter()` counts truncations so starvation is visible.
REFLECTION_MAX_TOKENS = 32000

# USD per 1M tokens (input, output). litellm's own per-call cost is recorded alongside as the
# authoritative aggregate; this table attributes cost to individual episodes under a thread pool.
PRICES: dict[str, tuple[float, float]] = {
    "anthropic/claude-haiku-4-5": (1.00, 5.00),
    "anthropic/claude-sonnet-5": (3.00, 15.00),
    "anthropic/claude-opus-5": (5.00, 25.00),
}

# `lambda * n_calls / STEP_BUDGET` is the charge; the unit of account is "fraction of the score a
# 30-call episode forfeits", same as TB2, so penalties read comparably across the demo series.
STEP_BUDGET = 30
# Hard ceiling on bridged predictor calls per forward, enforced by dspy.Flex itself.
MAX_PREDICTOR_CALLS = 60

EPISODE_TIMEOUT_S = 900.0        # agent wall clock per episode
MAX_COMMAND_TIMEOUT_S = 300      # per terminal_exec call
VERIFY_TIMEOUT_S = 3000.0        # verification container, end to end
MAX_OUTPUT_CHARS = 6000          # tool output clip
MAX_PATCH_CHARS = 400_000        # a bigger diff than this is a runaway, not a fix

DOCKER_LABEL = "dspy_swebp_demo=1"
DOCKERHUB_REPO = "jefzda/sweap-images"

# SWE-bench Pro images are linux/amd64 only; on Apple Silicon they run under emulation. The
# official local runner auto-detects exactly this way.
PLATFORM = "linux/amd64" if py_platform.machine().lower() in {"arm64", "aarch64"} else None

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class InstanceSpec(NamedTuple):
    instance_id: str
    repo: str                    # e.g. "qutebrowser/qutebrowser"
    language: str                # python / go / js / ts
    base_commit: str
    dockerhub_tag: str
    problem_statement: str
    requirements: str
    interface: str
    before_last_cmd: str         # LAST line of before_repo_set_cmd -- what the official eval uses
    selected_test_files: list[str]
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    gold_patch: str              # used ONLY by preflight --oracle; never shown to any model

    @property
    def image(self) -> str:
        return f"{DOCKERHUB_REPO}:{self.dockerhub_tag}"


def load_instances() -> dict[str, InstanceSpec]:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"{MANIFEST} not found -- run `python fetch_data.py` first")
    out: dict[str, InstanceSpec] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["instance_id"]] = InstanceSpec(**row)
    return out


# The pilot is deliberately Python-only: one language keeps the emulation/toolchain axis out of a
# 30-instance comparison (Go compiles and npm installs behave very differently under qemu), and
# Python is the language a small model is least handicapped on. The language axis belongs to a
# bigger run. Three repos so no single codebase carries the result.
PILOT_LANGUAGE = "python"
N_TRAIN_PER_REPO = {"ansible/ansible": 4, "internetarchive/openlibrary": 3, "qutebrowser/qutebrowser": 3}
N_VAL_PER_REPO = {"ansible/ansible": 3, "internetarchive/openlibrary": 3, "qutebrowser/qutebrowser": 2}
N_TEST_PER_REPO = {"ansible/ansible": 4, "internetarchive/openlibrary": 4, "qutebrowser/qutebrowser": 4}


def _to_example(spec: InstanceSpec) -> dspy.Example:
    return dspy.Example(instance_id=spec.instance_id, repo=spec.repo,
                        language=spec.language).with_inputs("instance_id")


def load_splits(seed: int = 0) -> tuple[list, list, list]:
    """Python-only pilot splits, stratified by repo, disjoint and deterministic under `seed`."""
    import random
    specs = [s for s in load_instances().values() if s.language == PILOT_LANGUAGE]
    by_repo: dict[str, list[InstanceSpec]] = {}
    for s in sorted(specs, key=lambda s: s.instance_id):
        by_repo.setdefault(s.repo, []).append(s)
    rng = random.Random(seed)
    train, val, test = [], [], []
    for repo in sorted(N_TRAIN_PER_REPO):
        pool = by_repo.get(repo, [])
        rng.shuffle(pool)
        a = N_TRAIN_PER_REPO[repo]
        b = a + N_VAL_PER_REPO[repo]
        c = b + N_TEST_PER_REPO[repo]
        train += [_to_example(s) for s in pool[:a]]
        val += [_to_example(s) for s in pool[a:b]]
        test += [_to_example(s) for s in pool[b:c]]
    for split in (train, val, test):
        rng.shuffle(split)
    return train, val, test


# ---------------------------------------------------------------------------
# Docker plumbing
# ---------------------------------------------------------------------------


class DockerError(RuntimeError):
    pass


def _docker(args: list[str], timeout: float, stdin: str | None = None) -> subprocess.CompletedProcess:
    """Run a docker command with byte-exact stdin/stdout.

    `subprocess.run(text=True, input=...)` rewrites newlines to `os.linesep` on Windows, which
    CRLF-corrupts every file written into a Linux container (found by TB2's oracle at $0).
    Encoding here and decoding by hand keeps the byte stream exact on every platform.
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
    try:
        proc = _docker(["version", "--format", "{{.Server.Version}}"], timeout=30)
    except FileNotFoundError:
        return False, "`docker` is not on PATH"
    except subprocess.TimeoutExpired:
        return False, "`docker version` timed out"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        return False, detail[-1] if detail else "unknown error"
    return True, f"server {proc.stdout.strip()}"


def ensure_image(image: str) -> None:
    if _docker(["image", "inspect", image], timeout=120).returncode == 0:
        return
    args = ["pull"] + ([f"--platform={PLATFORM}"] if PLATFORM else []) + [image]
    proc = _docker(args, timeout=3600)
    if proc.returncode != 0:
        raise DockerError(f"docker pull {image} failed: {proc.stderr.strip()[-300:]}")


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
    if len(text) <= limit:
        return text
    head, tail = limit * 2 // 3, limit // 3
    return f"{text[:head]}\n...[{len(text) - limit} characters omitted]...\n{text[-tail:]}"


def _sh_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


# ---------------------------------------------------------------------------
# Verifier (host-side, mirrors the official local-docker eval)
# ---------------------------------------------------------------------------


def strip_binary_hunks(patch: str) -> str:
    """Remove binary diff sections from a git patch (verbatim from the official eval)."""
    if not patch:
        return patch
    sections = re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE)
    kept: list[str] = []
    for section in sections:
        if not section.strip():
            continue
        if re.search(r"^Binary files .* differ$", section, re.MULTILINE):
            continue
        if re.search(r"^GIT binary patch$", section, re.MULTILINE):
            continue
        kept.append(section)
    return "".join(kept)


def _dockerfile_env_exports(instance_id: str) -> str:
    """ENV lines from the instance's two dockerfiles, converted to exports (official recipe)."""
    exports = []
    for sub in ("base_dockerfile", "instance_dockerfile"):
        path = DOCKERFILES / sub / instance_id / "Dockerfile"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("ENV"):
                exports.append(line.replace("ENV", "export", 1))
    return "\n".join(exports)


def build_entryscript(spec: InstanceSpec) -> str:
    """The official entryscript, byte-for-byte in structure (create_entryscript in the upstream)."""
    selected = ",".join(spec.selected_test_files)
    return f"""
{_dockerfile_env_exports(spec.instance_id)}
# apply patch
cd /app
git reset --hard {spec.base_commit}
git checkout {spec.base_commit}
git apply -v /workspace/patch.diff
{spec.before_last_cmd}
# run test and save stdout and stderr to separate files
bash /workspace/run_script.sh {selected} > /workspace/stdout.log 2> /workspace/stderr.log
# run parsing script
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
"""


class VerifyResult(NamedTuple):
    resolved: bool
    f2p_passed: int
    f2p_total: int
    p2p_failed: int
    p2p_total: int
    tail: str            # failing test names + log tails; feedback material
    wall_s: float
    infra_error: str | None


def run_verifier(spec: InstanceSpec, patch: str, timeout_s: float = VERIFY_TIMEOUT_S) -> VerifyResult:
    """Grade `patch` exactly as the official eval would: fresh container, run scripts, parse, compare."""
    started = time.perf_counter()
    run_script = RUN_SCRIPTS / spec.instance_id / "run_script.sh"
    parser = RUN_SCRIPTS / spec.instance_id / "parser.py"
    if not run_script.exists() or not parser.exists():
        return VerifyResult(False, 0, len(spec.fail_to_pass), 0, len(spec.pass_to_pass),
                            "", 0.0, f"run scripts missing for {spec.instance_id}")

    with tempfile.TemporaryDirectory(prefix="swebp_ws_") as ws:
        wsp = Path(ws)
        (wsp / "patch.diff").write_bytes(strip_binary_hunks(patch).encode("utf-8"))
        (wsp / "run_script.sh").write_bytes(run_script.read_bytes())
        (wsp / "parser.py").write_bytes(parser.read_bytes())
        (wsp / "entryscript.sh").write_bytes(build_entryscript(spec).encode("utf-8"))

        name = f"swebp-verify-{uuid.uuid4().hex[:12]}"
        args = ["run", "-d", "--name", name, "--label", DOCKER_LABEL,
                *([f"--platform={PLATFORM}"] if PLATFORM else []),
                "-v", f"{wsp}:/workspace",
                "--entrypoint", "/bin/bash", spec.image, "-c", "bash /workspace/entryscript.sh"]
        try:
            ensure_image(spec.image)
            proc = _docker(args, timeout=300)
            if proc.returncode != 0:
                return VerifyResult(False, 0, len(spec.fail_to_pass), 0, len(spec.pass_to_pass),
                                    "", time.perf_counter() - started,
                                    f"verify container failed to start: {proc.stderr.strip()[-300:]}")
            try:
                _docker(["wait", name], timeout=timeout_s)
            except subprocess.TimeoutExpired:
                _docker(["rm", "-f", name], timeout=120)
                return VerifyResult(False, 0, len(spec.fail_to_pass), 0, len(spec.pass_to_pass),
                                    f"verifier exceeded {timeout_s:.0f}s", time.perf_counter() - started, None)
            _docker(["rm", "-f", name], timeout=120)

            out_path = wsp / "output.json"
            stdout_tail = _tail_file(wsp / "stdout.log", 1500)
            stderr_tail = _tail_file(wsp / "stderr.log", 1500)
            if not out_path.exists():
                tail = ("the test runner produced no parseable output (output.json missing)\n"
                        f"--- stdout tail ---\n{stdout_tail}\n--- stderr tail ---\n{stderr_tail}")
                return VerifyResult(False, 0, len(spec.fail_to_pass), 0, len(spec.pass_to_pass),
                                    tail, time.perf_counter() - started, None)

            output = json.loads(out_path.read_text(encoding="utf-8", errors="replace"))
            passed = {t["name"] for t in output.get("tests", []) if t.get("status") == "PASSED"}
            f2p, p2p = set(spec.fail_to_pass), set(spec.pass_to_pass)
            resolved = (f2p | p2p) <= passed
            f2p_failing = sorted(f2p - passed)
            p2p_failing = sorted(p2p - passed)
            lines = []
            if f2p_failing:
                lines.append(f"{len(f2p_failing)}/{len(f2p)} issue tests still failing, e.g.:")
                lines += [f"  FAIL {t}" for t in f2p_failing[:8]]
            if p2p_failing:
                lines.append(f"{len(p2p_failing)}/{len(p2p)} previously-passing tests now broken (regression), e.g.:")
                lines += [f"  REGRESSION {t}" for t in p2p_failing[:8]]
            if not lines:
                lines.append("all required tests passed")
            lines.append(f"--- test runner stderr tail ---\n{stderr_tail}")
            return VerifyResult(resolved, len(f2p) - len(f2p_failing), len(f2p),
                                len(p2p_failing), len(p2p),
                                "\n".join(lines), time.perf_counter() - started, None)
        except DockerError as exc:
            return VerifyResult(False, 0, len(spec.fail_to_pass), 0, len(spec.pass_to_pass),
                                "", time.perf_counter() - started, str(exc)[:300])
        except Exception as exc:
            return VerifyResult(False, 0, len(spec.fail_to_pass), 0, len(spec.pass_to_pass),
                                "", time.perf_counter() - started, f"{type(exc).__name__}: {exc}"[:300])


def _tail_file(path: Path, limit: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "(missing)"
    return text[-limit:].strip() or "(empty)"


# ---------------------------------------------------------------------------
# Agent container session
# ---------------------------------------------------------------------------


class ContainerSession:
    """One instance's agent container, plus the transcript of everything the harness did to it.

    Same design as TB2's session layer: commands run through `docker exec bash -lc` (stateless),
    with the working directory carried across calls via a state file. The repo is at /app, reset
    hard to the instance's base commit at start. The updated test files, the run scripts and the
    test lists are never present in this container -- verification happens elsewhere, afterwards.
    """

    CWD_FILE = "/tmp/.dspy_swebp_cwd"

    def __init__(self, spec: InstanceSpec, episode_timeout_s: float = EPISODE_TIMEOUT_S):
        self.spec = spec
        self.sid = f"swebp-{uuid.uuid4().hex[:12]}"
        self.container_id: str | None = None
        self.workdir = "/app"
        self.transcript: list[dict[str, Any]] = []
        self.episode_timeout_s = episode_timeout_s
        self.deadline = time.perf_counter() + episode_timeout_s   # reset at the end of start()
        self.deadline_hit = False
        self._has_timeout_bin = True

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        ensure_image(self.spec.image)
        args = ["run", "-d", "--name", self.sid, "--label", DOCKER_LABEL,
                *([f"--platform={PLATFORM}"] if PLATFORM else []),
                "--entrypoint", "", self.spec.image, "sleep", "infinity"]
        proc = _docker(args, timeout=300)
        if proc.returncode != 0:
            raise DockerError(f"docker run failed for {self.spec.instance_id}: {proc.stderr.strip()[-300:]}")
        self.container_id = proc.stdout.strip()

        # A clean slate at the base commit: `reset --hard` puts tracked files back; untracked
        # build artifacts (dependency dirs, caches) are left alone on purpose -- they are part of
        # the prepared environment, and the official entryscript does not clean them either.
        boot = (f"git config --global --add safe.directory /app; cd /app && "
                f"git reset --hard {self.spec.base_commit} && "
                f"printf %s /app > {self.CWD_FILE}")
        proc = self._raw_exec(boot, 300, workdir="/")
        if proc.returncode != 0:
            raise DockerError(f"repo reset failed for {self.spec.instance_id}: {proc.stderr.strip()[-300:]}")
        self._has_timeout_bin = self._raw_exec("command -v timeout", 30).returncode == 0
        # The agent's clock starts only now, so a cold image pull is never charged to the harness.
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
        if not self.container_id:
            return "ERROR: container is not running."
        remaining = self.remaining_s()
        if remaining <= 0:
            self.deadline_hit = True
            return ("ERROR: the episode wall-clock budget is exhausted. Stop issuing commands and "
                    "return your summary now.")
        budget = int(max(1, min(int(timeout_s or 60), MAX_COMMAND_TIMEOUT_S, remaining)))

        # One shell restores the last working directory, runs the command, records the new one and
        # exits with the command's status -- a `cd` in a nested shell would never persist otherwise.
        script = (
            f'cd "$(cat {self.CWD_FILE} 2>/dev/null || echo {self.workdir})" 2>/dev/null || cd {self.workdir}\n'
            f"{command}\n"
            f"__dspy_rc=$?\n"
            f"pwd > {self.CWD_FILE} 2>/dev/null\n"
            f"exit $__dspy_rc\n"
        )
        # Kill container-side: a client-side timeout leaves the real process running inside.
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

    # -- patch extraction ----------------------------------------------------

    def extract_patch(self) -> tuple[str, str | None]:
        """(patch, error). Staged diff against the base commit, so new files are included and the
        result applies cleanly with `git apply` at that commit -- even if the harness committed."""
        if not self.container_id:
            return "", "container was not running at patch-extraction time"
        add = self._raw_exec("cd /app && git add -A .", 180, workdir="/")
        if add.returncode != 0:
            return "", f"git add failed: {add.stderr.strip()[:200]}"
        proc = self._raw_exec(f"cd /app && git diff --cached {self.spec.base_commit}", 180, workdir="/")
        if proc.returncode != 0:
            return "", f"git diff failed: {proc.stderr.strip()[:200]}"
        patch = proc.stdout
        if len(patch) > MAX_PATCH_CHARS:
            return "", f"patch too large ({len(patch)} chars > {MAX_PATCH_CHARS}) -- treated as a runaway"
        return patch, None

    # -- reporting ----------------------------------------------------------

    def digest(self, max_commands: int = 18) -> str:
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


# ---------------------------------------------------------------------------
# Host tools handed to dspy.Flex
# ---------------------------------------------------------------------------

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
    """Run one bash command in the repository's Linux container; returns exit code, stdout, stderr.

    `session` is the session id from the module's inputs. The working directory persists between
    calls; exported variables do not. The repository under work is at /app. Long output is
    clipped, so filter with grep/head when you expect a lot. Commands are killed at `timeout_s`
    (max 300).
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


class SolveSWEIssue(dspy.Signature):
    """Fix the described issue in the repository checked out at /app inside a Linux container by
    editing its source files, then report what you changed."""

    # Field descriptions are mechanical API facts, not strategy -- anything resembling strategy in
    # the seed is task-solving information the optimizer should have had to discover (and it would
    # inflate the baseline it is compared against).
    problem_statement: str = dspy.InputField(desc="The issue report, as filed.")
    requirements: str = dspy.InputField(desc="What an acceptable fix must do.")
    interface: str = dspy.InputField(desc="New or changed functions/methods the fix should expose, if any.")
    repo: str = dspy.InputField(desc="The GitHub repository this issue belongs to.")
    language: str = dspy.InputField(desc="The repository's primary language.")
    session: str = dspy.InputField(
        desc="Opaque session id. Pass it unchanged as the `session=` argument of every tool call."
    )
    summary: str = dspy.OutputField(desc="What was changed, and in which files.")


class SWEAgent(dspy.Module):
    """One episode end to end: start container -> run the Flex harness -> extract patch -> verify.

    Only `self.harness` is code-optimizable. The verifier runs *after* the harness has returned,
    in a separate fresh container the harness never had access to, using the benchmark's own
    per-instance scripts -- so no harness, however it is rewritten, can read or edit the thing
    that grades it.
    """

    def __init__(self, harness: dspy.Module | None = None,
                 episode_timeout_s: float = EPISODE_TIMEOUT_S,
                 verify_timeout_s: float = VERIFY_TIMEOUT_S):
        super().__init__()
        self.harness = harness if harness is not None else new_harness()
        self.episode_timeout_s = episode_timeout_s
        self.verify_timeout_s = verify_timeout_s

    def forward(self, instance_id: str) -> dspy.Prediction:
        spec = load_instances()[instance_id]
        session = ContainerSession(spec, episode_timeout_s=self.episode_timeout_s)
        summary, harness_error = "", None
        agent_wall_s = 0.0
        try:
            session.start()
            _register(session)
            started = time.perf_counter()
            try:
                out = self.harness(problem_statement=spec.problem_statement,
                                   requirements=spec.requirements,
                                   interface=spec.interface,
                                   repo=spec.repo, language=spec.language,
                                   session=session.sid)
                summary = str(getattr(out, "summary", "") or "")
            except Exception as exc:
                # A harness crash is a harness failure, not a lost episode: extract and verify
                # anyway, because a crash after the fix was written should still count.
                harness_error = f"{type(exc).__name__}: {exc}"[:400]
            agent_wall_s = time.perf_counter() - started
            patch, patch_error = session.extract_patch()
        except Exception as exc:
            return dspy.Prediction(
                resolved=False, summary=summary, n_commands=0, n_failed_commands=0,
                transcript_digest="(the container never started)", verifier_tail="",
                patch_chars=0, harness_error=f"{type(exc).__name__}: {exc}"[:400],
                deadline_hit=False, agent_wall_s=agent_wall_s, verify_wall_s=0.0,
                infra_error=True,
            )
        finally:
            _unregister(session)
            session.stop()

        if patch_error:
            harness_error = harness_error or patch_error
            patch = ""

        if patch.strip():
            v = run_verifier(spec, patch, timeout_s=self.verify_timeout_s)
            if v.infra_error:
                return dspy.Prediction(
                    resolved=False, summary=summary, n_commands=session.n_commands,
                    n_failed_commands=session.n_failed_commands,
                    transcript_digest=session.digest(), verifier_tail="",
                    patch_chars=len(patch), harness_error=v.infra_error,
                    deadline_hit=session.deadline_hit, agent_wall_s=agent_wall_s,
                    verify_wall_s=v.wall_s, infra_error=True,
                )
            resolved, verifier_tail, verify_wall_s = v.resolved, v.tail, v.wall_s
        else:
            # No edits at all: skip the container round-trip; the verdict is a foregone conclusion
            # (the fail_to_pass tests fail at the base commit by construction).
            resolved, verify_wall_s = False, 0.0
            verifier_tail = "no patch was produced -- the repository was left unmodified"

        return dspy.Prediction(
            resolved=bool(resolved),
            summary=summary,
            n_commands=session.n_commands,
            n_failed_commands=session.n_failed_commands,
            transcript_digest=session.digest(),
            verifier_tail=verifier_tail,
            patch_chars=len(patch),
            harness_error=harness_error,
            deadline_hit=session.deadline_hit,
            agent_wall_s=agent_wall_s,
            verify_wall_s=verify_wall_s,
            infra_error=False,
        )


def new_harness() -> dspy.Flex:
    """A fresh, un-optimized harness: `dspy.Flex` with tools seeds itself as a single `dspy.RLM`
    over the signature -- a code-REPL agent with container access. That IS the baseline; GEPA's
    job is to turn it into a software-engineering harness."""
    return dspy.Flex(SolveSWEIssue, tools=HARNESS_TOOLS, max_predictor_calls=MAX_PREDICTOR_CALLS)


def make_lms(exec_model: str = EXEC_MODEL,
             exec_max_tokens: int = EXEC_MAX_TOKENS,
             reflection_max_tokens: int = REFLECTION_MAX_TOKENS) -> tuple[dspy.LM, dspy.LM]:
    """Return (execution LM, reflection LM) and make history unbounded for cost accounting."""
    dspy.configure(max_history_size=10**9)
    return (dspy.LM(exec_model, max_tokens=exec_max_tokens),
            dspy.LM(REFLECTION_MODEL, max_tokens=reflection_max_tokens))


def disable_cache() -> None:
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


def make_resolve_metric(step_budget: int = STEP_BUDGET):
    """Score = did the benchmark's own tests pass. The CAPABILITY objective.

    This experiment's question is "does an evolved harness let a small model solve more issues",
    not "can it solve them more cheaply", so calls are reported as context but never charged. The
    feedback is a transcript plus the verifier's per-test verdicts, structured around diagnosis:
    name the capability the harness lacked and build it in as a reusable part.
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
        patch_chars = int(getattr(pred, "patch_chars", 0) or 0)
        harness_error = getattr(pred, "harness_error", None)
        deadline_hit = bool(getattr(pred, "deadline_hit", False))

        if getattr(pred, "infra_error", False):
            return ScoreWithFeedback(
                score=0.0,
                feedback=("INFRASTRUCTURE FAILURE, not a harness failure: the container or "
                          f"verifier could not run ({harness_error}). Ignore this example; it says "
                          "nothing about the harness code."),
            )

        budget = (f"{n_calls} model call(s), {n_commands} shell command(s) ({n_failed} nonzero exit), "
                  f"patch of {patch_chars} chars -- not counted against the score; the tests alone decide it")

        goal = (
            "OBJECTIVE: resolve more issues. The benchmark's own tests alone decide the score: an "
            "issue counts only if every issue test passes AND no previously-passing test breaks. "
            "Model calls and shell commands are not charged, but the episode has a wall-clock "
            f"budget and a hard ceiling of {MAX_PREDICTOR_CALLS} model calls -- spend them where "
            "they buy information or progress.\n"
            "When an episode fails, the useful question is what CAPABILITY the harness lacked. "
            "Diagnose the gap and build the fix into the harness so it generalises to issues you "
            "have not seen:\n"
            "  - turn ad-hoc moves into reusable helpers over the three tools (locating the code "
            "an issue is about, reading enough context before editing, applying an edit and "
            "re-reading the file to confirm it took, running the repo's own tests near the change);\n"
            "  - isolate a hard sub-problem into its own sub-predictor with its own instructions "
            "when one round of reasoning is not enough;\n"
            "  - check your own work before returning: run something that would fail if the fix "
            "were wrong, read the error, and fix it -- an unverified 'done' is how regressions ship;\n"
            "  - make failures recoverable: read what a command actually printed and try a "
            "different approach instead of repeating one that already failed.\n"
            "What to look at first, how to structure the loop, and when to stop is for you to "
            "work out from the transcripts."
        )

        head: str
        if harness_error and not resolved:
            head = (f"The harness RAISED before finishing ({budget}). Error: {harness_error}. This "
                    f"is a bug in the harness itself: catch tool errors, check the strings tools "
                    f"return, and always return dspy.Prediction(summary=<str>).")
        elif deadline_hit and not resolved:
            head = (f"FAILED by running out of wall clock ({budget}). Time, not model calls, was "
                    f"the binding resource: reach a candidate fix early and improve it, rather "
                    f"than exploring for a long time and committing late.")
        elif not resolved and n_commands == 0:
            head = (f"FAILED having run NO commands at all ({budget}) -- the worst outcome, because "
                    f"nothing was attempted in the repository. The harness must actually call "
                    f"terminal_exec/write_file/read_file with session=<the session input>.")
        elif not resolved and patch_chars == 0:
            head = (f"FAILED with NO patch ({budget}): commands ran but no file was changed, so "
                    f"there was nothing to grade. An issue is resolved by editing source files "
                    f"under /app, not by diagnosis alone.")
        elif not resolved:
            head = (f"FAILED: the patch did not satisfy the tests ({budget}). Compare what the "
                    f"verifier below checked against what the patch actually changed -- the gap "
                    f"between them is the capability to add.")
        else:
            head = (f"RESOLVED ({budget}). Keep whatever produced this. If it relied on something "
                    f"ad hoc, make it a reusable part of the harness so it survives on other issues.")

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
    """Accumulate calls / tokens / litellm cost across `lms` for the duration of the block."""
    totals: dict[str, Any] = {
        "calls": 0, "cost_usd_litellm": 0.0,
        "prompt_tokens": 0, "completion_tokens": 0, "uncosted_calls": 0,
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
    instance: str
    repo: str
    resolved: bool
    n_calls: int
    n_commands: int
    n_failed_commands: int
    patch_chars: int
    deadline_hit: bool
    latency_s: float
    verify_s: float
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int
    summary: str
    verifier_tail: str
    error: str | None


class EpisodeCache:
    """Append-only JSONL of finished episodes, keyed by (arm, harness source hash, instance) --
    a killed evaluation resumes where it stopped, and a cached record can never be attributed to
    a program that did not produce it."""

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
    def key(arm: str, program_src: str, instance: str) -> str:
        digest = hashlib.sha256((program_src or "").encode("utf-8")).hexdigest()[:16]
        return f"{arm}|{digest}|{instance}"

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


def run_program(program: dspy.Module, dataset: list, threads: int = 3,
                arm: str = "baseline", cache: EpisodeCache | None = None,
                progress: bool = True) -> tuple[list[Record], dict]:
    src = getattr(getattr(program, "harness", None), "module_src", "") or ""

    def run_one(ex) -> Record:
        key = EpisodeCache.key(arm, src, ex.instance_id)
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
                instance=ex.instance_id, repo=ex.repo,
                resolved=bool(pred.resolved), n_calls=len(trace),
                n_commands=int(pred.n_commands or 0),
                n_failed_commands=int(pred.n_failed_commands or 0),
                patch_chars=int(pred.patch_chars or 0),
                deadline_hit=bool(pred.deadline_hit),
                latency_s=float(pred.agent_wall_s or 0.0), verify_s=float(pred.verify_wall_s or 0.0),
                cost_usd=cost, prompt_tokens=p_tok, completion_tokens=c_tok,
                summary=_clip(str(pred.summary or ""), 1200),
                verifier_tail=_clip(str(pred.verifier_tail or ""), 2500),
                error=pred.harness_error,
            )
        except Exception as exc:
            record = Record(ex.instance_id, ex.repo, False, 0, 0, 0, 0, False,
                            time.perf_counter() - started, 0.0, 0.0, 0, 0, "", "",
                            f"{type(exc).__name__}: {exc}"[:300])
        if cache is not None:
            cache.put(key, record)
        if progress:
            short = record.instance.replace("instance_", "")[:44]
            print(f"    [{'PASS' if record.resolved else 'fail'}] {short:<46} "
                  f"calls={record.n_calls:<3} cmds={record.n_commands:<3} patch={record.patch_chars:<6} "
                  f"{record.latency_s:5.0f}s+{record.verify_s:4.0f}s ${record.cost_usd:.3f}"
                  + (f"  !! {record.error}" if record.error else ""), flush=True)
        return record

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        records = list(pool.map(run_one, dataset))
    return records, {"wall_s": time.perf_counter() - started, "threads": threads}


def summarize(records: list[Record]) -> dict:
    n = len(records) or 1
    lat = sorted(r.latency_s for r in records)
    resolved = sum(1 for r in records if r.resolved)

    def pct(p: float) -> float:
        return lat[min(len(lat) - 1, round(p * (len(lat) - 1)))] if lat else 0.0

    by_repo = {}
    for repo in sorted({r.repo for r in records}):
        sub = [r for r in records if r.repo == repo]
        by_repo[repo] = {"n": len(sub), "resolved": sum(1 for r in sub if r.resolved)}

    return {
        "n": len(records),
        "resolve_rate": resolved / n,
        "n_resolved": resolved,
        "avg_calls": sum(r.n_calls for r in records) / n,
        "avg_commands": sum(r.n_commands for r in records) / n,
        "avg_failed_commands": sum(r.n_failed_commands for r in records) / n,
        "avg_patch_chars": sum(r.patch_chars for r in records) / n,
        "n_empty_patches": sum(1 for r in records if r.patch_chars == 0),
        "cost_usd_total": sum(r.cost_usd for r in records),
        "cost_usd_per_task": sum(r.cost_usd for r in records) / n,
        "latency_mean_s": sum(lat) / n,
        "latency_p50_s": pct(0.50),
        "latency_p95_s": pct(0.95),
        "verify_mean_s": sum(r.verify_s for r in records) / n,
        "prompt_tokens": sum(r.prompt_tokens for r in records),
        "completion_tokens": sum(r.completion_tokens for r in records),
        "deadline_hits": sum(1 for r in records if r.deadline_hit),
        "errors": sum(1 for r in records if r.error),
        "first_error": next((r.error for r in records if r.error), None),
        "by_repo": by_repo,
    }


def fmt(row: dict) -> str:
    return (
        f"resolved={row['n_resolved']}/{row['n']} ({row['resolve_rate']:.3f}) "
        f"calls/task={row['avg_calls']:.1f} cmds/task={row['avg_commands']:.1f} "
        f"${row['cost_usd_per_task']:.2f}/task lat_mean={row['latency_mean_s']:.0f}s "
        f"deadline_hits={row['deadline_hits']} empty_patches={row['n_empty_patches']}"
    )


atexit.register(reap_orphans)
