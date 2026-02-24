# RFC: Production-Readiness for RunSkillScriptTool

**Authors:** haiyuancao
**Date:** 2026-02-24
**Status:** Proposed
**Audience:** ADK TL / UTL
**Effort:** 8-11 engineering days (P0 scope only; see
  `code_executor_enhancements.md` for full 15-22 day estimate
  covering P0-P2)
**Tracking:** Follow-up to PR #4575 (RunSkillScriptTool)

---

## 1. Executive Summary

`RunSkillScriptTool` lets agents execute Python and shell scripts
bundled with skills. It shipped in PR #4575 and is functional, but two
gaps block production use:

1. **Python scripts can hang forever.** Shell scripts have a
   `subprocess.run(timeout=300)` guard; Python scripts have none. A
   single stuck `runpy.run_path()` call holds a process-global lock,
   blocking every subsequent execution across all agents in the process.

2. **All local execution is unprotected.** `UnsafeLocalCodeExecutor` —
   the only zero-dependency executor and the default for development —
   runs `exec()` in the host process. A malicious or buggy script can
   read secrets, write to the filesystem, exfiltrate data over the
   network, or crash the process.

This RFC proposes two changes, scoped to what is required before skills
can be used in any environment beyond local prototyping:

- **P0-A: Uniform timeout** — Add `timeout_seconds` to the executor
  contract so every `execute_code()` call has a bounded lifetime.
- **P0-B: `LocalSandboxCodeExecutor`** — A new stdlib-only executor
  that runs code in a subprocess with resource limits, replacing
  `UnsafeLocalCodeExecutor` as the recommended local default.

Both changes are backward-compatible, additive, and independently
shippable.

---

## 2. Problem Statement

### 2.1 The Timeout Gap

The executor landscape today:

| Executor | Timeout | How |
|----------|---------|-----|
| `UnsafeLocalCodeExecutor` | **None** | `exec()` blocks forever |
| `ContainerCodeExecutor` | **None** | `exec_run()` blocks forever |
| `GkeCodeExecutor` | 300 s | K8s watch API |
| `VertexAiCodeExecutor` | Opaque | Vertex AI internal |
| `AgentEngineSandboxCodeExecutor` | Opaque | Vertex AI Sandbox internal |

`RunSkillScriptTool` works around this for shell scripts by embedding
`subprocess.run(timeout=N)` in generated wrapper code. The default is
300 seconds, configurable via `SkillToolset(script_timeout=N)`.

**Python scripts have zero timeout at any layer.** The tool generates:

```python
import sys, runpy
sys.argv = ['scripts/run.py', '--verbose', 'True']
runpy.run_path('scripts/run.py', run_name='__main__')
```

This runs inline inside `exec()`. There is no subprocess boundary, no
watchdog thread, and no way for the caller to interrupt it. Worse,
`UnsafeLocalCodeExecutor` holds a process-global `threading.Lock()`
for the entire execution (required because `redirect_stdout` and
`os.chdir` mutate process-global state). A hung Python script
deadlocks the lock, blocking **all** code execution across the entire
process — every agent and every `UnsafeLocalCodeExecutor` instance
shares the same module-level `_execution_lock`.

**Impact:** A single infinite-loop in a skill script takes down the
entire ADK process. This is a denial-of-service risk for any
deployment — not just production, but also development servers and
demos.

### 2.2 The Security Gap

`UnsafeLocalCodeExecutor` runs code with the full privileges of the
host Python process:

| Threat | Impact | Current Mitigation |
|--------|--------|--------------------|
| Read env vars / secrets | Data exfiltration | None |
| Write to host filesystem | Data loss / corruption | Partial (temp-dir sandbox when `input_files` or `working_dir` is set) |
| Outbound network calls | Data leak | None |
| `sys.exit()` | Process crash | `SystemExit` caught in tool |
| Infinite loop / fork bomb | DoS | Shell: `subprocess.run(timeout)`; Python: **none** |

This executor is the only one that requires zero external dependencies.
While `LlmAgent.code_executor` does not globally default to it (it
defaults to `None`, and CFC paths may force `BuiltInCodeExecutor`),
`UnsafeLocalCodeExecutor` is the common choice in practice for setups
that need local code execution:
- Samples and tutorials that demonstrate `code_executor=` configuration
- `adk web` / `adk run` development workflows
- CI test suites without Docker

Any code the LLM generates or any third-party skill script runs with
full host access. This is acceptable for trusted, single-developer
prototyping. It is not acceptable for:
- Multi-user development servers
- CI/CD pipelines running untrusted skill scripts
- Any path toward production deployment

---

## 3. Proposed Solution

### 3.1 P0-A: Uniform Timeout Support

**Goal:** Every `execute_code()` call has a bounded lifetime, regardless
of executor backend or script type.

#### 3.1.1 Contract Changes (Backward-Compatible)

Add one field to each layer:

```python
# Per-invocation timeout (caller sets this)
@dataclasses.dataclass
class CodeExecutionInput:
    code: str
    input_files: list[File] = field(default_factory=list)
    execution_id: Optional[str] = None
    working_dir: Optional[str] = None
    timeout_seconds: Optional[int] = None  # NEW

# Executor-level default (fallback when caller doesn't set one)
class BaseCodeExecutor(BaseModel):
    default_timeout_seconds: Optional[int] = None  # NEW
    ...
```

Resolution logic in every executor:
```python
timeout = (
    code_execution_input.timeout_seconds
    if code_execution_input.timeout_seconds is not None
    else self.default_timeout_seconds
)
```

**Why two fields, not one?**
- A single executor instance may be shared across agents/tools with
  different timeout needs (quick validation vs. long data analysis).
- Per-invocation is the source of truth; executor default is the
  safety net for callers that don't set one.
- Existing code that sets neither field gets `None` → no timeout →
  identical to current behavior. Zero breaking changes.

#### 3.1.2 `UnsafeLocalCodeExecutor` — Thread + Join + Unhealthy Guard

`exec()` cannot be interrupted from the same thread. Run it in a
daemon thread with `join(timeout)`:

```
                        ┌─ execute_code() ──────────────────────┐
                        │                                        │
  check _healthy ───────│   if not self._healthy: raise error    │
                        │                                        │
  acquire               │   with _execution_lock:                │
  _execution_lock ──────│     spawn daemon thread ──► exec()     │
                        │     thread.join(timeout)               │
                        │     if thread.is_alive():              │
                        │       self._healthy = False ◄── mark!  │
                        │       return stderr="timed out"        │
                        │                                        │
  release lock ─────────│   (lock released)                      │
                        └────────────────────────────────────────┘
```

**Critical invariant — unhealthy after timeout:**

Simply releasing `_execution_lock` while a timed-out daemon thread is
still running would let subsequent executions proceed while the
lingering thread continues to mutate process-global `sys.stdout` and
the working directory. This causes cross-execution stdout
contamination and cwd corruption — a data-integrity bug, not just a
cosmetic issue.

The solution is to **mark the executor unhealthy on timeout**:

1. On timeout, set `self._healthy = False` before returning the
   timeout error. The lock is released normally when the `with` block
   exits.
2. Subsequent `execute_code()` calls check `self._healthy` at the top
   and **fail fast** with a clear error: `"Executor is unhealthy after
   a timed-out execution. Call reinitialize() to recover."`
3. `reinitialize()` waits for the lingering daemon thread to finish
   (with a generous join timeout), resets `_healthy = True`, and
   allows execution to resume.

This ensures that no new execution can run while a zombie thread is
still alive and mutating shared state. The pattern mirrors the
`ContainerCodeExecutor` unhealthy-state design (§3.1.3).

**Trade-off:** After a timeout, the executor is unavailable until
`reinitialize()` is called. This is acceptable for a development
executor — the alternative (silent stdout/cwd corruption) is worse.
Production deployments should use `LocalSandboxCodeExecutor` (§3.2)
or `ContainerCodeExecutor`, where timeout kill is clean.

**Why not skip thread-timeout on `UnsafeLocalCodeExecutor` entirely?**
Without any timeout, a hung `exec()` holds `_execution_lock` forever,
which is strictly worse — the executor is permanently blocked with no
error and no recovery path. The unhealthy-guard approach at least
unblocks the lock, reports the error, and offers a recovery mechanism.

#### 3.1.3 `ContainerCodeExecutor` — Docker Exec Kill

Run `exec_start` in a thread. On timeout, kill the process from the
host side:

1. `exec_inspect(exec_id)` → get host-namespace PID
2. `os.kill(host_pid, SIGKILL)`
3. If `PermissionError` (common: container runs as root, ADK does not)
   → `container.restart(timeout=1)` + readiness check
4. If both fail → set `self._healthy = False`, return error; caller
   must call `reinitialize()` to recover

This is the only design that guarantees the caller is never blocked
beyond `timeout` seconds, regardless of whether the kill succeeds.

#### 3.1.4 Wire Timeout in `RunSkillScriptTool`

Once `CodeExecutionInput.timeout_seconds` exists, the tool sets it:

```python
CodeExecutionInput(
    code=code,
    input_files=input_files,
    working_dir=".",
    timeout_seconds=self._toolset._script_timeout,  # NEW
)
```

This gives Python scripts the same 300-second default that shell
scripts already have, and makes it configurable via
`SkillToolset(script_timeout=N)`. The shell `subprocess.run(timeout)`
is kept as defense-in-depth.

#### 3.1.5 Migration

| Step | Change | Risk |
|------|--------|------|
| 1 | Add `timeout_seconds` to `CodeExecutionInput` | None |
| 2 | Add `default_timeout_seconds` to `BaseCodeExecutor` | None |
| 3 | Implement in `UnsafeLocalCodeExecutor` (thread + join) | Low |
| 4 | Implement in `ContainerCodeExecutor` (exec kill) | Low |
| 5 | Migrate `GkeCodeExecutor.timeout_seconds` to new field | None |
| 6 | Wire in `RunSkillScriptTool` | None |

**Estimated effort:** 3-4 days (including tests).

---

### 3.2 P0-B: `LocalSandboxCodeExecutor`

**Goal:** A zero-dependency executor that provides meaningful isolation
without Docker or cloud services.

#### 3.2.1 Why Not Just Harden `UnsafeLocalCodeExecutor`?

We considered two alternatives before proposing a new executor:

| Option | Approach | Why rejected |
|--------|----------|-------------|
| **Restricted builtins** | Block `open`, `__import__`, `exec`, `eval` in `exec()` globals | Trivially bypassed via `object.__subclasses__()`, `importlib` through `sys.modules`, `__builtins__.__dict__`. Not a security boundary — at best a speed bump. |
| **SecurityWarning only** | Emit `warnings.warn(SecurityWarning)` on first use | Does not reduce attack surface. Users ignore warnings. |

Both are worth adding as **supplementary** friction (low cost, some
value), but neither solves the problem. True isolation requires a
process boundary.

#### 3.2.2 Design: Subprocess with Resource Limits

```python
class LocalSandboxCodeExecutor(BaseCodeExecutor):
    """Executes Python in a sandboxed subprocess.

    Isolation:
    - Separate process (no shared memory with host)
    - Resource limits (CPU time, memory) via resource.setrlimit
    - Restricted environment variables
    - Temporary working directory
    - subprocess.run(timeout=N) for wall-clock timeout
    """

    default_timeout_seconds: int = 30
    max_memory_mb: int = 256
    max_cpu_seconds: int = 30
    allowed_env_vars: list[str] = []
```

Execution flow:

```
  1. Write code to a NamedTemporaryFile (.py)
  2. Write input_files to a TemporaryDirectory
  3. Build minimal env: only allowed_env_vars + PATH
  4. Popen(
         ['python3', '-c', <limit_code> + exec(open(file).read())],
         env=env,
         cwd=temp_dir,
         process_group=0,       # Python 3.11+; preexec_fn fallback
         stdout=PIPE, stderr=PIPE,
     )
  5. communicate(timeout=timeout)
  6. On TimeoutExpired:
       os.killpg(proc.pid, signal.SIGKILL)   # kill entire group
       proc.communicate(timeout=5)             # reap zombies
  7. Return CodeExecutionResult(stdout, stderr)
```

**Why `Popen` + `os.killpg` instead of `subprocess.run(timeout)`:**

`subprocess.run(timeout=N)` only kills the direct child process. If
the script spawns subprocesses (e.g., `os.system()`, `Popen`,
`multiprocessing`), those descendants survive the timeout and become
orphans. With `process_group=0`, the child is placed in its own
process group. On timeout, `os.killpg(proc.pid, SIGKILL)` kills the
entire group — the child and all its descendants. The follow-up
`proc.communicate(timeout=5)` reaps any zombies.

On Python 3.10 (where `process_group` is unavailable), the fallback
uses `preexec_fn=_setup_child` where `_setup_child` calls
`os.setpgrp()` (process-group isolation) **and** sets
`resource.setrlimit` for CPU/memory. This achieves the same
kill-group semantics, with a documented caveat about fork-safety
in multi-threaded programs (see §3.2.3).

The inline `limit_code` wrapper sets `resource.setrlimit` for CPU and
memory inside the child process (guarded with `try/except ImportError`
for platforms where `resource` is unavailable).

**What this protects against vs. what it doesn't:**

| Threat | Protected? | How |
|--------|-----------|-----|
| Infinite loop (direct) | Yes | Wall-clock timeout + `RLIMIT_CPU` |
| Infinite loop (child procs) | Yes | `os.killpg` kills entire process group |
| Fork bomb | **Partial** | `RLIMIT_CPU` + timeout bound total wall-clock; does not cap `RLIMIT_NPROC` (could be added) |
| Memory bomb | Yes | `RLIMIT_AS` |
| Env var / secret reading | Yes | Restricted `env` dict |
| `sys.exit()` crash | Yes | Separate process |
| Filesystem read/write | **Partial** | `cwd` is temp dir, but host fs still accessible via absolute paths |
| Network exfiltration | **No** | Requires OS-level firewall (out of scope) |

This is strictly stronger than `UnsafeLocalCodeExecutor` across every
dimension, with zero additional dependencies. The remaining gaps
(full filesystem isolation, network restriction, `RLIMIT_NPROC`)
require containers or OS-level policy.

#### 3.2.3 Platform Considerations

- **Python 3.11+:** Use `process_group=0` (fork-safe, replaces
  `preexec_fn`). Enables clean `os.killpg()` on timeout.
- **Python 3.10:** Fall back to `preexec_fn=_setup_child` where
  `_setup_child` calls `os.setpgrp()` (new process group, required
  for `os.killpg` on timeout) **and** sets `resource.setrlimit` for
  CPU/memory. Documented caveat: `preexec_fn` is not fork-safe in
  multi-threaded programs. ADK minimum is `>=3.10` per
  `pyproject.toml`.
- **Windows:** Not supported. Raise `NotImplementedError` directing
  users to `ContainerCodeExecutor`. (`resource.setrlimit` and
  `process_group` are Unix-only.)

#### 3.2.4 Migration

| Step | Change | Risk |
|------|--------|------|
| 1 | Implement `LocalSandboxCodeExecutor` | Low |
| 2 | Add `SecurityWarning` to `UnsafeLocalCodeExecutor` | None |
| 3 | Add `restrict_builtins` option (opt-in, supplementary) | Low |
| 4 | Update samples and `adk web` defaults to recommend sandbox | Low |
| 5 | Update documentation | None |

**Estimated effort:** 5-7 days (including tests and docs).

---

## 4. Alternatives Considered

### 4.1 "Do Nothing — Document the Risks"

Add warnings to docs and let users choose their executor.

**Rejected.** Samples, tutorials, and common development workflows
use `UnsafeLocalCodeExecutor` for local code execution. New users
follow sample code without reading security docs. The executor most
commonly reached by new developers must be safe enough for its
intended use (local development with untrusted LLM-generated code).

### 4.2 "Require Docker for All Local Execution"

Make `ContainerCodeExecutor` the default. Remove
`UnsafeLocalCodeExecutor`.

**Rejected.** Docker is a significant dependency. Many developers
(especially on macOS) don't have it installed. CI environments
may not have Docker available. The zero-dependency story is a key
ADK differentiator for onboarding. We need a middle ground.

### 4.3 "Timeout Only, No New Executor"

Ship P0-A (timeout) without P0-B (sandbox). Defer security to a
later phase.

**Viable but insufficient.** Timeout prevents DoS but does not
prevent data exfiltration, secret reading, or filesystem manipulation.
These are the higher-impact threats for skill scripts, which may come
from third-party authors. We recommend shipping both P0-A and P0-B,
but they are independently valuable and can be phased if needed.

---

## 5. Rollout Plan

### 5.1 Phase 1: Timeout Foundation (Week 1)

| Day | Deliverable |
|-----|-------------|
| 1 | Add `timeout_seconds` to `CodeExecutionInput` and `default_timeout_seconds` to `BaseCodeExecutor`. Unit tests for field defaults and resolution logic. |
| 2 | Implement thread-based timeout in `UnsafeLocalCodeExecutor`. Tests with `time.sleep()` scripts. |
| 3 | Implement Docker exec kill in `ContainerCodeExecutor` with `_healthy` guard and `reinitialize()`. Mock-based tests for kill path, permission error path, and total failure path. |
| 4 | Migrate `GkeCodeExecutor`, wire `RunSkillScriptTool`, end-to-end integration tests. |

**Exit criteria:** `RunSkillScriptTool` Python scripts respect
`script_timeout`. Existing tests pass. No behavioral change for
callers that don't set timeout.

### 5.2 Phase 2: Security Hardening (Week 2)

| Day | Deliverable |
|-----|-------------|
| 1-2 | Implement `LocalSandboxCodeExecutor` with resource limits. |
| 3 | Add `SecurityWarning` to `UnsafeLocalCodeExecutor`. Add `restrict_builtins` opt-in. |
| 4 | Tests: resource limit enforcement, env var isolation, timeout kill via `os.killpg`, platform fallback for 3.10. |
| 5 | Update samples, docs, and recommendation matrix. |

**Exit criteria:** `LocalSandboxCodeExecutor()` works as a drop-in
replacement for `UnsafeLocalCodeExecutor()` for the supported script
profile: scripts that use stdout/stderr for output, do not depend on
host environment variables beyond an explicit allowlist, and access
files only within the sandbox working directory. Scripts that rely on
broad host-filesystem access or specific env vars will need to
configure `allowed_env_vars` or use `UnsafeLocalCodeExecutor`.
Documentation recommends `LocalSandboxCodeExecutor` for all new local
development and clearly states the compatibility envelope.

### 5.3 Recommendation Matrix (Post-Rollout)

| Use Case | Executor | Why |
|----------|----------|-----|
| Local development | `LocalSandboxCodeExecutor` | Zero deps, subprocess isolation |
| Quick prototyping (trusted code) | `UnsafeLocalCodeExecutor` | Fastest, no isolation |
| CI/CD | `ContainerCodeExecutor` | Docker available in CI |
| Production (single tenant) | `ContainerCodeExecutor` | Full container isolation |
| Production (multi-tenant) | `GkeCodeExecutor` | gVisor, per-execution isolation |
| Google Cloud | `AgentEngineSandboxCodeExecutor` | Managed, scalable |

### 5.4 Success Metrics

- **Timeout coverage:** 100% of executors support `timeout_seconds`
  (currently 1 of 5).
- **Default safety:** `LocalSandboxCodeExecutor` passes all existing
  `RunSkillScriptTool` integration tests that fit the supported script
  profile (stdout/stderr output, sandbox-local file access, no
  dependency on host env vars beyond allowlist).
- **No regressions:** All existing unit and integration tests pass
  with zero behavioral changes for callers that don't opt in.
- **Adoption signal:** Samples and `adk web` default documentation
  reference `LocalSandboxCodeExecutor`.

---

## Appendix A: Current RunSkillScriptTool Execution Flow

```
LLM calls run_skill_script(skill_name, script_path, args)
  │
  ▼
RunSkillScriptTool.run_async()
  │
  ├─ Validate params (skill_name, script_path, args type)
  ├─ Resolve skill → resolve script from resources
  ├─ Resolve executor: toolset._code_executor → agent.code_executor
  ├─ Package ALL skill files as input_files (refs, assets, scripts)
  ├─ _prepare_code():
  │    .py  → runpy.run_path() wrapper (NO timeout)
  │    .sh  → subprocess.run(timeout=N) wrapper with JSON envelope
  │
  ├─ await asyncio.to_thread(executor.execute_code, ...)
  │    CodeExecutionInput(code=..., input_files=..., working_dir=".")
  │    *** timeout_seconds NOT SET — this is the gap ***
  │
  ├─ Parse result:
  │    Shell: unpack JSON envelope {stdout, stderr, returncode}
  │    Python: use stdout/stderr directly
  │
  └─ Return {skill_name, script_path, stdout, stderr, status}
```

## Appendix B: Key Source Files

| File | Role |
|------|------|
| `src/google/adk/tools/skill_toolset.py` | `RunSkillScriptTool`, `SkillToolset` |
| `src/google/adk/code_executors/base_code_executor.py` | `BaseCodeExecutor` abstract class |
| `src/google/adk/code_executors/code_execution_utils.py` | `CodeExecutionInput`, `CodeExecutionResult`, `File` |
| `src/google/adk/code_executors/unsafe_local_code_executor.py` | Current local executor |
| `src/google/adk/code_executors/container_code_executor.py` | Docker-based executor |
| `src/google/adk/code_executors/gke_code_executor.py` | GKE-based executor (has timeout) |
| `tests/unittests/tools/test_skill_toolset.py` | ~1170-line test suite for skill tools |
| `docs/design/code_executor_enhancements.md` | Detailed design doc (companion to this RFC) |
