# ADK Code Executor Enhancements — Design Document

**Authors:** haiyuancao, Claude Code
**Date:** 2026-02-24
**Status:** Draft
**Tracking:** Related to PR #4575 (RunSkillScriptTool)

---

## 1. Motivation

The ADK code executor infrastructure (`src/google/adk/code_executors/`) is
the backbone for both LLM-driven code execution and skill script execution.
A review of the current implementations reveals three critical gaps that
limit production readiness:

1. **No uniform timeout enforcement** — Only `GkeCodeExecutor` has a
   `timeout_seconds` field. All other executors can hang indefinitely on
   malicious, buggy, or slow code. The `RunSkillScriptTool` works
   around this for shell scripts by embedding `subprocess.run(timeout=N)`
   in generated code, but this is a workaround, not a systemic solution.

2. **No stateful Python execution** — `ContainerCodeExecutor` maintains a
   persistent Docker container but explicitly freezes `stateful=False`.
   Agents cannot preserve variables, imports, or working directory across
   code execution calls. `VertexAiCodeExecutor` and
   `AgentEngineSandboxCodeExecutor` allow `stateful=True` at the field
   level but only support it for their specific backend APIs.

3. **No safe local executor** — `UnsafeLocalCodeExecutor` runs `exec()` in
   the host process with zero isolation. It is the only executor that
   requires no external dependencies (no Docker, no GKE, no Vertex AI),
   making it the default choice for development and demos. A compromised
   script can read secrets, modify the filesystem, or crash the process.

This document proposes solutions for all three gaps with minimal disruption
to the existing API.

---

## 1.1 Prioritized Next Steps for `RunSkillScriptTool`

The three major proposals in this doc remain valid, but for improving
`RunSkillScriptTool` specifically, we should prioritize them alongside
additional high-impact tool-contract work.

| Rank | Priority | Area | Why it matters now |
|------|----------|------|--------------------|
| 1 | P0 | Uniform timeout support (Proposal 1) | Python script execution can still hang indefinitely without executor-level timeout controls. |
| 2 | P0 | Security hardening (Proposal 3) | `UnsafeLocalCodeExecutor` remains unsafe for untrusted scripts and is a major deployment risk. |
| 3 | P1 | Structured `RunSkillScriptTool` result contract | Agents need explicit machine-readable execution metadata (`return_code`, timeout flag), not just inferred status from `stdout/stderr`. |
| 4 | P1 | Propagate `output_files` / artifacts | Script-generated outputs are currently dropped by tool responses, limiting practical utility. |
| 5 | P1 | Strengthen script argument contract | Argument normalization rules are underspecified, which leads to fragile calls and inconsistent behavior. |
| 6 | P1 | Wire `execution_id` in `RunSkillScriptTool` | Needed for predictable namespace isolation and future stateful execution compatibility. |
| 7 | P2 | Stateful `ContainerCodeExecutor` (Proposal 2) | Valuable, but more complex and less urgent than reliability/safety/tool-contract gaps above. |

**Interpretation for implementation planning:**
- P0 items are required for production reliability/safety.
- P1 items directly improve agent correctness and tool usability.
- P2 items are strategic enhancements once P0/P1 are complete.

---

## 2. Non-Goals & Invariants

The following are explicitly **out of scope** for this design:

1. **Full sandboxing of `UnsafeLocalCodeExecutor`** — The restricted
   builtins mechanism (Tier 1, §6.3.1-B) is a best-effort friction layer,
   not a security boundary. Any determined code can bypass it via
   `object.__subclasses__()`, `importlib` through `__builtins__`, etc.
   True isolation requires a process or container boundary.

2. **Automatic state recovery after crash** — The stateful
   `ContainerCodeExecutor` (Proposal 2) uses a persistent REPL
   (Option A). If the REPL or container crashes, in-process state
   is lost. The executor reports an error; it does **not** attempt
   automatic replay of prior code blocks, because prior blocks may
   have had non-idempotent side effects (file writes, network calls,
   database mutations) that should not be re-executed.

3. **Multi-tenant per-execution isolation** — Per-execution isolation
   (fresh sandbox per call) is the domain of `GkeCodeExecutor` and
   cloud-hosted executors. Container and local executors share a
   single execution environment within a session.

4. **Windows support for `LocalSandboxCodeExecutor`** —
   `resource.setrlimit` and `process_group` are Unix-only. Windows
   support is deferred to a future iteration.

**Key invariants:**

- `timeout_seconds` is a **per-invocation** parameter, not executor-global
  state. When a single executor instance is shared across agents/tools,
  each `execute_code()` call may specify its own timeout via
  `CodeExecutionInput.timeout_seconds`.
- Code is appended to stateful history **only after** successful execution.
  A failing code block is never replayed.
- Executor instances are not thread-safe unless documented otherwise.
  Concurrent `execute_code()` calls on the same instance may require
  external synchronization.
- `UnsafeLocalCodeExecutor` is a special case: it currently serializes
  `execute_code()` with an internal lock because `redirect_stdout` and
  `os.chdir()` mutate process-global state.

---

## 3. Current State

### 3.1 Executor Landscape

| Executor | Stateful | Timeout | Isolation | Dependencies |
|----------|----------|---------|-----------|-------------|
| `UnsafeLocalCodeExecutor` | No (frozen) | None | Partial (temp-dir sandbox when `input_files` or `working_dir` set; no isolation otherwise) | None |
| `ContainerCodeExecutor` | No (frozen) | None | Docker container | `docker` |
| `GkeCodeExecutor` | No (ephemeral) | `timeout_seconds=300` | gVisor sandbox | `kubernetes` |
| `VertexAiCodeExecutor` | Allowed | None | Vertex AI Extension | `vertexai` |
| `AgentEngineSandboxCodeExecutor` | Allowed | None | Vertex AI Sandbox | `vertexai` |
| `BuiltInCodeExecutor` | N/A (delegates to Gemini model's built-in code execution) | N/A (Gemini-internal) | Gemini model | `google-genai` |

**Note on `UnsafeLocalCodeExecutor` partial isolation:** When `input_files`
or `working_dir` is set in `CodeExecutionInput`, the executor creates a
`tempfile.TemporaryDirectory`, writes all input files with preserved
relative paths (e.g., `references/data.csv`), and `os.chdir()`s into it
before calling `exec()`. This provides filesystem-level isolation for the
script's view of its working directory, but does **not** restrict access
to the rest of the host filesystem, environment variables, or network.
The temp directory is cleaned up after execution, and the original working
directory is restored in a `finally` block. Both the sandbox and plain
paths hold a process-global `_execution_lock` because `redirect_stdout`
mutates `sys.stdout`.

### 3.2 Base Class Contract

```python
class BaseCodeExecutor(BaseModel):
    optimize_data_file: bool = False
    stateful: bool = False
    error_retry_attempts: int = 2
    code_block_delimiters: List[tuple[str, str]] = [
        ('```tool_code\n', '\n```'),
        ('```python\n', '\n```'),
    ]
    execution_result_delimiters: tuple[str, str] = (
        '```tool_output\n', '\n```'
    )

    @abc.abstractmethod
    def execute_code(
        self,
        invocation_context: InvocationContext,
        code_execution_input: CodeExecutionInput,
    ) -> CodeExecutionResult: ...
```

### 3.3 Data Model

```python
@dataclasses.dataclass(frozen=True)
class File:
    name: str
    content: str | bytes  # Text or binary; executor writes 'w'/'wb'
    mime_type: str = 'text/plain'
    path: Optional[str] = None  # e.g. "scripts/run.py"

@dataclasses.dataclass
class CodeExecutionInput:
    code: str
    input_files: list[File] = field(default_factory=list)
    execution_id: Optional[str] = None  # For stateful execution
    working_dir: Optional[str] = None   # e.g. "."

@dataclasses.dataclass
class CodeExecutionResult:
    stdout: str = ''
    stderr: str = ''
    output_files: list[File] = field(default_factory=list)
```

**Binary content handling:** `File.content` accepts both `str` and
`bytes`. When `UnsafeLocalCodeExecutor` writes input files to the
temp-dir sandbox, it selects file mode based on content type:
`'wb'` for `bytes`, `'w'` for `str`. This is relevant for
`RunSkillScriptTool`, which packages skill resources as `File` objects
— references and assets may contain binary content (e.g., images,
serialized data).

**`_prepare_globals` helper in `UnsafeLocalCodeExecutor`:** The executor
has a `_prepare_globals()` function that scans the code for
`if __name__ == '__main__'` patterns and injects `__name__ = '__main__'`
into the execution globals. This interacts with `RunSkillScriptTool`'s
Python wrapper, which uses `runpy.run_path(script_path,
run_name='__main__')` — `runpy` sets `__name__` independently, so the
executor's `_prepare_globals` applies to the wrapper code's outer scope
while `runpy` sets it for the script's scope.

### 3.4 How Executors Are Used

The primary consumer is `_code_execution.py` in the LLM flows layer:

1. **Pre-processor**: Extracts data files, runs preprocessing code
2. **Post-processor**: Extracts code blocks from LLM responses, executes
   them, feeds results back to the LLM
3. **Stateful support**: Uses `execution_id` (from `CodeExecutorContext`)
   to maintain state across calls when `stateful=True`

`RunSkillScriptTool` is a secondary consumer that wraps
`execute_code()` in `asyncio.to_thread()` to avoid blocking the async
event loop:

```python
result = await asyncio.to_thread(
    code_executor.execute_code,
    tool_context._invocation_context,
    CodeExecutionInput(
        code=code,
        input_files=input_files,
        working_dir=".",
    ),
)
```

This is architecturally significant: the tool is always called from an
async context (`run_async`), but all `BaseCodeExecutor.execute_code()`
implementations are synchronous. The `to_thread()` bridge ensures the
executor's blocking call (which may involve `exec()`, Docker API calls,
or HTTP requests) does not starve the event loop.

#### 3.4.1 `RunSkillScriptTool` Current Implementation Details

Key behaviors of the current implementation (`skill_toolset.py`) that
inform the proposals in this document:

- **Executor resolution chain:** Toolset-level `_code_executor` (highest
  priority) → `agent.code_executor` attribute → `NO_CODE_EXECUTOR` error.
- **`script_timeout` parameter:** `SkillToolset.__init__` accepts
  `script_timeout: int` (default 300s). This timeout is embedded in
  generated shell wrapper code via `subprocess.run(timeout=N)`. It does
  **not** apply to Python scripts executed via `runpy.run_path()` — those
  run inline in `exec()` with no timeout at any layer.
- **`scripts/` prefix normalization:** `_prepare_code()` auto-prepends
  `"scripts/"` if the `script_path` does not already start with it. This
  allows the LLM to call with either `"setup.py"` or `"scripts/setup.py"`.
- **Resource packaging:** ALL skill files (references, assets, scripts)
  are packaged as `input_files` with preserved relative paths (e.g.,
  `"references/data.csv"`, `"assets/template.txt"`). Empty resources
  (content `""`) are still included — they are not silently dropped.
  Both text (`str`) and binary (`bytes`) content are supported; the
  executor writes them with the appropriate file mode (`'w'` vs `'wb'`).
- **SystemExit handling:** `SystemExit(0)` or `SystemExit(None)` →
  treated as success (empty stdout/stderr, status `"success"`).
  `SystemExit(non-zero)` → `EXECUTION_ERROR` with the exit code in the
  error message. This prevents scripts from terminating the host process.
- **Error message truncation:** Exception messages longer than 200
  characters are truncated to `message[:200] + "..."` to conserve LLM
  context tokens.
- **Shell JSON envelope:** Shell scripts are wrapped in a
  `subprocess.run` call that serializes output as JSON:
  `{"__shell_result__": true, "stdout": "...", "stderr": "...",
  "returncode": N}`. On parse:
  - Non-zero `returncode` with empty `stderr` → synthesized
    `"Exit code {rc}"` as stderr
  - Non-JSON stdout (e.g., if the wrapper itself fails) → raw stdout
    is passed through without parsing
- **Status derivation:** Purely based on stream presence:
  `stderr` only → `"error"`, both streams → `"warning"`, no
  `stderr` → `"success"`. For shell scripts, non-zero return codes
  influence status indirectly (via synthesized stderr). For Python
  scripts, there is no return code extraction — status is determined
  solely by stdout/stderr content.
- **Duplicate skill names:** `SkillToolset.__init__` validates that all
  skill names are unique and raises `ValueError` on duplicates.
- **System instruction injection:** `SkillToolset.process_llm_request()`
  appends `DEFAULT_SKILL_SYSTEM_INSTRUCTION` plus an XML-formatted skill
  list to every outgoing LLM request, informing the model about available
  skills and the `run_skill_script` tool.

---

## 4. Proposal 1: Uniform Timeout Support

### 4.1 Problem

A code execution call can hang indefinitely. This is a denial-of-service
risk for any production deployment, whether the code comes from an LLM, a
skill script, or user input.

| Executor | Current timeout behavior |
|----------|------------------------|
| `UnsafeLocalCodeExecutor` | `exec()` blocks forever |
| `ContainerCodeExecutor` | `exec_run()` blocks forever |
| `GkeCodeExecutor` | K8s watch timeout (works) |
| `VertexAiCodeExecutor` | Vertex AI internal timeout (opaque) |
| `AgentEngineSandboxCodeExecutor` | Vertex AI internal timeout (opaque) |

### 4.2 Design

#### 4.2.1 Add `timeout_seconds` to `CodeExecutionInput`

Timeout is a **per-invocation** concern, not executor-global state. A single
executor instance may be shared across agents, tools, and concurrent calls
with different timeout requirements (e.g., a quick validation script vs. a
long-running data analysis). Placing timeout on the executor would create
race conditions when multiple callers set different values.

```python
@dataclasses.dataclass
class CodeExecutionInput:
    code: str
    input_files: list[File] = field(default_factory=list)
    execution_id: Optional[str] = None
    timeout_seconds: Optional[int] = None  # NEW
    """Maximum execution time in seconds. None means no timeout
    (executor default behavior). Each execute_code() call reads
    this from its input, not from executor-level state."""
```

Additionally, a **default** timeout on `BaseCodeExecutor` serves as
a fallback when callers don't specify one:

```python
class BaseCodeExecutor(BaseModel):
    default_timeout_seconds: Optional[int] = None
    """Default timeout applied when CodeExecutionInput.timeout_seconds
    is None. Subclasses may override (e.g., GkeCodeExecutor defaults
    to 300). None means no timeout."""
```

The effective timeout is resolved as:
```python
input_t = code_execution_input.timeout_seconds
timeout = (
    input_t if input_t is not None
    else self.default_timeout_seconds
)
```

**Why per-invocation + executor default:**
- Backward compatible — existing code that doesn't set it works unchanged
- Safe for shared executors — no global mutable state
- Callers can override per-call (e.g., `RunSkillScriptTool` sets
  `script_timeout`, LLM flows use a different default)
- Executor subclasses can define their own defaults

#### 4.2.2 `UnsafeLocalCodeExecutor` — Thread-Based Timeout

`exec()` cannot be interrupted from the same thread. The solution is to
run it in a separate thread with a join timeout:

```python
import threading

def execute_code(self, invocation_context, code_execution_input):
    input_t = code_execution_input.timeout_seconds
    timeout = (
        input_t if input_t is not None
        else self.default_timeout_seconds
    )
    if timeout is None:
        # No timeout: current behavior (blocking exec)
        return self._execute_inline(code_execution_input)

    # Run in a daemon thread with timeout
    result_holder = {}
    def _run():
        result_holder['result'] = self._execute_inline(
            code_execution_input
        )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        # Thread is still running — timeout exceeded.
        # Daemon thread will be killed when the process exits.
        return CodeExecutionResult(
            stderr=f'Execution timed out after {timeout}s'
        )
    return result_holder.get('result', CodeExecutionResult(
        stderr='Execution produced no result'
    ))
```

**Trade-offs:**
- Daemon threads cannot be forcefully killed in CPython. If the code enters
  a long-running C extension call (e.g., `time.sleep(9999)`), the thread
  lingers until process exit. This is acceptable for a local development
  executor — production deployments should use container-based executors.
- An alternative is `multiprocessing`, but that adds complexity around
  serialization and shared state.

**Interaction with existing `_execution_lock`:**

The current `UnsafeLocalCodeExecutor` holds a process-global
`_execution_lock` (`threading.Lock()`) for the entire duration of
`execute_code()`, covering both the sandbox path (temp-dir + chdir) and
the plain path (redirect_stdout only). The timeout thread proposal must
account for this:

- **Lock must be acquired outside the timeout thread.** The worker thread
  must hold the lock while executing, and the calling thread must release
  it after the join (whether the worker finishes or times out). If the
  lock were acquired inside the worker thread, a timed-out worker would
  hold the lock indefinitely, deadlocking all subsequent calls.
- **Recommended pattern:** Acquire the lock in `execute_code()` before
  spawning the worker thread, pass the lock-holding context to the worker,
  and release in a `finally` block after `thread.join(timeout)`:
  ```python
  with _execution_lock:
      thread = threading.Thread(target=_run, daemon=True)
      thread.start()
      thread.join(timeout=timeout)
      if thread.is_alive():
          # Lock is released when `with` exits, even though
          # the daemon thread may still be running.
          # This is acceptable: the lingering thread's exec()
          # is no longer protected by the lock, but it is a
          # daemon thread that will be killed on process exit.
          return CodeExecutionResult(
              stderr=f'Execution timed out after {timeout}s'
          )
  ```
- **Risk:** A timed-out daemon thread may still be mutating
  `sys.stdout` or the working directory after the lock is released.
  This is a best-effort trade-off for a development executor — the
  alternative (never releasing the lock) would deadlock the process.

**Recommendation:** Thread-based timeout for `UnsafeLocalCodeExecutor` is
sufficient. Document that it provides best-effort timeout only.

#### 4.2.3 `ContainerCodeExecutor` — Docker Exec Kill on Timeout

**Problem with thread+join in containers:** Unlike `UnsafeLocalCodeExecutor`
where a lingering daemon thread is merely wasteful, a runaway process inside
a shared container consumes CPU/memory and can interfere with subsequent
executions. The thread+join pattern would leave the container process
running indefinitely after the join timeout expires.

**Primary approach: Docker exec kill via `exec_inspect` + PID kill.**

The key constraint is that `exec_start` is a **blocking** call — a
timer thread cannot unblock it if the in-container kill fails. The
correct design runs `exec_start` in a worker thread so the caller
can enforce the timeout via `thread.join(timeout)`, then uses the
**host-side Docker API** to kill the exec'd process by its host PID.

```python
import os
import signal
import threading

def execute_code(self, invocation_context, code_execution_input):
    # Fail fast if a prior timeout left the executor unhealthy.
    if not self._healthy:
        raise RuntimeError(
            'ContainerCodeExecutor is unhealthy after a failed '
            'timeout cleanup. Call reinitialize() to recover.'
        )

    input_t = code_execution_input.timeout_seconds
    timeout = (
        input_t if input_t is not None
        else self.default_timeout_seconds
    )

    # Create the exec instance
    exec_id = self._client.api.exec_create(
        self._container.id,
        ['python3', '-c', code_execution_input.code],
    )['Id']

    # Run exec_start in a thread so we can enforce timeout
    # from the calling thread.
    result_holder = {}

    def _run_exec():
        result_holder['output'] = (
            self._client.api.exec_start(exec_id, demux=True)
        )

    thread = threading.Thread(target=_run_exec, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        # Timeout exceeded — kill from host side.
        # exec_inspect returns the host-namespace PID, which
        # is the correct PID for os.kill() on the host.
        # (Killing from inside the container would require
        # the container-namespace PID, which is different.)
        cleanup_failed = False
        try:
            info = self._client.api.exec_inspect(exec_id)
            host_pid = info.get('Pid', 0)
            if host_pid > 0:
                os.kill(host_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # Process already exited — no action needed
        except (PermissionError, Exception):
            # os.kill failed (most commonly PermissionError
            # when container runs as root and ADK does not).
            # Restart the container to ensure the runaway
            # process is terminated.
            try:
                self._container.restart(timeout=1)
                # Re-validate runtime readiness after restart,
                # mirroring the init-time check (see
                # container_code_executor.py:169).
                check = self._container.exec_run(
                    ['python3', '--version']
                )
                if check.exit_code != 0:
                    raise RuntimeError(
                        f'Post-restart readiness check failed '
                        f'(exit_code={check.exit_code})'
                    )
            except Exception as restart_err:
                cleanup_failed = True
                logger.error(
                    'Timeout cleanup failed: could not kill '
                    'process or restart container: %s',
                    restart_err,
                )
                self._healthy = False

        # Give the worker thread a short window to finish
        # after kill/restart, so it doesn't leak indefinitely.
        thread.join(timeout=2)
        if thread.is_alive():
            logger.warning(
                'Worker thread still alive after timeout '
                'cleanup; daemon thread will linger until '
                'process exit.'
            )

        if cleanup_failed:
            return CodeExecutionResult(
                stderr=(
                    f'Execution timed out after {timeout}s '
                    f'and cleanup failed — executor is '
                    f'unhealthy. Reinitialize the executor '
                    f'before further use.'
                )
            )
        return CodeExecutionResult(
            stderr=f'Execution timed out after {timeout}s'
        )

    output = result_holder.get('output')
    # ... parse output as before
```

**Why this design:**

1. **`exec_start` in a thread + `join(timeout)`** — The caller is
   never blocked longer than `timeout` seconds, regardless of whether
   the kill succeeds. This resolves the primary DoS risk.

2. **Host-side `os.kill(host_pid, SIGKILL)`** — `exec_inspect()`
   returns the **host-namespace** PID. Using `os.kill()` from the
   host process operates in the host PID namespace, so the PID
   matches correctly. This avoids:
   - The PID namespace mismatch of killing from inside the container
   - The overbroad pattern matching of `pkill -f` (which can hit
     concurrent execs or unrelated Python processes)
   - Dependency on `procps`/`pkill` being installed in the image

3. **Post-kill thread join** — After kill/restart, a short
   `thread.join(timeout=2)` gives the worker thread time to exit
   cleanly. If it's still alive, a warning is logged. The thread is
   a daemon, so it will not prevent process exit, but repeated
   timeout failures without this join could accumulate leaked threads.

4. **Unhealthy state on total cleanup failure** — If both `os.kill`
   and `container.restart()` fail, the executor sets `self._healthy
   = False` and returns a distinct error message. Subsequent calls
   check `self._healthy` and raise early rather than queueing work
   against a broken container. The `_healthy` lifecycle:
   - Initialized to `True` in `__init__` (alongside container start)
   - Set to `False` on total cleanup failure (kill + restart both fail)
   - Set back to `True` after successful `reinitialize()` (stops
     current container, creates new one, passes readiness check)

5. **Container restart as last resort** — If `os.kill` fails (e.g.,
   insufficient permissions when Docker runs rootless), restart the
   container. This is the most reliable fallback but destroys
   in-container state.

**Permissions and user mismatch risk:** `os.kill()` on the host PID
requires the ADK process to run as the same user that owns the
container's exec'd process on the host. The current
`ContainerCodeExecutor` does not set `user=` on `containers.run()`
(`container_code_executor.py:182`), so the container process runs as
root. If the ADK process runs as a non-root user (common in
development), `os.kill(host_pid, SIGKILL)` will raise
`PermissionError` and the container-restart fallback activates.

This user mismatch is the expected case for default Docker usage,
so the `PermissionError → container.restart()` path is the **primary
timeout mechanism in practice**, not an edge case. The `os.kill` path
becomes primary only when ADK runs as root or the container is
configured with `user=` matching the host user.

**Recovery after timeout:**
- **Stateless mode:** No recovery needed. The killed process is gone;
  the next `exec_run` starts a fresh process in the same container.
- **Stateful mode (Option A / persistent process):** The timed-out
  block is NOT appended to history (append-after-success invariant).
  If the persistent REPL was killed or the container was restarted,
  the executor returns an error with `stderr` indicating state was
  lost. The caller (LLM flow or skill tool) must handle this —
  typically by starting a new session. Automatic replay is not
  attempted because it may re-execute side effects.

#### 4.2.4 `GkeCodeExecutor` — Already Implemented

`GkeCodeExecutor` already has `timeout_seconds: int = 300` applied to the
K8s watch API. Migrate it to use the base class default field:

```python
class GkeCodeExecutor(BaseCodeExecutor):
    default_timeout_seconds: int = 300  # Override base default
```

In `execute_code()`, resolve timeout from per-invocation input first:
```python
input_t = code_execution_input.timeout_seconds
timeout = (
    input_t if input_t is not None
    else self.default_timeout_seconds
)
```

No behavioral change for existing callers (they don't set per-invocation
timeout, so the 300s default applies as before).

#### 4.2.5 Remote Executors (Vertex AI, Agent Engine)

These executors delegate to Google Cloud APIs that have their own internal
timeouts. Adding client-side timeout is still valuable as a safety net.

Note: The current `execute_code()` implementations in
`VertexAiCodeExecutor` and `AgentEngineSandboxCodeExecutor` are
**synchronous** (they call blocking SDK methods), so `asyncio.wait_for()`
is not applicable. Use the same thread+join pattern as
`UnsafeLocalCodeExecutor`:

- Run the blocking API call in a daemon thread with `join(timeout)`
- Return `CodeExecutionResult(stderr='...')` on timeout
- Log a warning that the server-side execution may still be running
- If these executors are migrated to async in the future, switch to
  `asyncio.wait_for()` at that point

### 4.3 Migration Plan

| Phase | Action | Risk |
|-------|--------|------|
| 1 | Add `timeout_seconds: Optional[int] = None` to `CodeExecutionInput` | None (backward compatible) |
| 2 | Add `default_timeout_seconds: Optional[int] = None` to `BaseCodeExecutor` | None (backward compatible) |
| 3 | Implement in `UnsafeLocalCodeExecutor` (thread + join) | Low |
| 4 | Implement in `ContainerCodeExecutor` (Docker exec kill) | Low |
| 5 | Migrate `GkeCodeExecutor.timeout_seconds` to `default_timeout_seconds` | None |
| 6 | Add client-side timeout to remote executors | Low |

### 4.4 Impact on `RunSkillScriptTool`

Once `BaseCodeExecutor` has native timeout support, the
`RunSkillScriptTool` can optionally delegate timeout enforcement to
the executor rather than embedding it in generated shell wrapper code.
However, the shell wrapper timeout (`subprocess.run(timeout=N)`) should
be kept as defense-in-depth — it catches the subprocess even if the
executor timeout fails.

**Critical gap — Python scripts have zero timeout protection:**

Shell scripts benefit from two layers of timeout: the `subprocess.run(
timeout=N)` embedded in generated wrapper code, and (once implemented)
the executor-level timeout. Python scripts have **neither**:

- `_prepare_code()` generates `runpy.run_path()` inside a plain `exec()`
  call — there is no subprocess boundary to kill.
- `SkillToolset.script_timeout` only applies to the shell path (it is
  passed to `subprocess.run(timeout=N)`). The docstring explicitly notes:
  "Does not apply to Python scripts executed via exec()."
- Until executor-level timeout is implemented, a Python script that hangs
  (infinite loop, blocking I/O, deadlock) will block the executor thread
  indefinitely. With `UnsafeLocalCodeExecutor`, this also holds the
  `_execution_lock`, blocking all other executions.

**Recommended actions for Phase 1:**
1. Executor-level timeout (§4.2) is the primary fix — it covers both
   Python and shell scripts uniformly.
2. As defense-in-depth, `RunSkillScriptTool` should also set
   `CodeExecutionInput.timeout_seconds = self._toolset._script_timeout`
   once the field is available, ensuring per-invocation timeout even if
   the executor has no default.
3. Optionally, the Python wrapper code could be enhanced with a
   watchdog thread pattern (similar to the shell `subprocess.run`
   timeout), though this is less clean than executor-level enforcement.

---

## 5. Proposal 2: Stateful `ContainerCodeExecutor`

### 5.1 Problem

Agents often need multi-step code execution where later steps depend on
earlier results. For example:

```
Step 1: import pandas as pd; df = pd.read_csv('data.csv')
Step 2: filtered = df[df['status'] == 'active']
Step 3: print(filtered.describe())
```

Currently, each `execute_code()` call in `ContainerCodeExecutor` runs
`python3 -c <code>` — a fresh Python process with no memory of prior
calls. Step 2 would fail with `NameError: name 'df' is not defined`.

### 5.2 Design

#### 5.2.1 Architecture

```
┌─ ContainerCodeExecutor ─────────────────────┐
│                                              │
│  stateful=False (default):                   │
│    exec_run(['python3', '-c', code])         │
│    └─ Fresh process per call                 │
│                                              │
│  stateful=True:                              │
│    exec_run(['python3', '-c',                │
│      'exec(open("/app/session.py").read())'  │
│      + '\n' + code                           │
│      + '\n_save_state()'])                   │
│    └─ Loads prior state, executes,           │
│       saves state back                       │
│                                              │
└──────────────────────────────────────────────┘
```

**Option A: Persistent Python Process (Complex)**

Start a long-running Python REPL in the container and pipe code to its
stdin:

```python
# On init (when stateful=True):
self._exec_id = self._client.api.exec_create(
    self._container.id,
    ['python3', '-i'],
    stdin=True, stdout=True, stderr=True,
)
self._socket = self._client.api.exec_start(
    self._exec_id, socket=True
)
```

Pros: True statefulness — all Python state (variables, imports, objects)
persists naturally.

Cons: Complex I/O management. Need to detect when output is "done" for a
given code block (no clean delimiter). Prone to deadlocks. Hard to detect
crashes.

**Option B: State Serialization via `dill` (Moderate)**

After each execution, serialize the global namespace to a file in the
container. Before next execution, deserialize it:

```python
STATEFUL_WRAPPER = '''
import dill as _dill, os as _os
_state_file = '/tmp/.adk_state.pkl'
if _os.path.exists(_state_file):
    _globals = _dill.load(open(_state_file, 'rb'))
    globals().update(_globals)

{user_code}

# Save state after execution
_save_vars = {k: v for k, v in globals().items()
              if not k.startswith('_')}
_dill.dump(_save_vars, open(_state_file, 'wb'))
'''
```

Pros: Simpler than persistent REPL. Each call is a clean `exec_run()`.

Cons: Not all Python objects are serializable (e.g., open file handles,
database connections, generators). Adds `dill` dependency to the container
image. Serialization/deserialization overhead grows with state size.

**Option C: Shared Globals File (Simple)**

Write executed code to a cumulative Python file. Each call appends the new
code block and re-executes the entire history:

```python
HISTORY_FILE = '/tmp/.adk_history.py'

def execute_code(self, invocation_context, code_execution_input):
    if self.stateful:
        # Append new code to history
        self._container.exec_run(
            ['sh', '-c',
             f'cat >> {HISTORY_FILE} << "ADKEOF"\n'
             f'{code_execution_input.code}\n'
             f'ADKEOF'],
        )
        # Execute full history
        exec_result = self._container.exec_run(
            ['python3', HISTORY_FILE],
            demux=True,
        )
    else:
        exec_result = self._container.exec_run(
            ['python3', '-c', code_execution_input.code],
            demux=True,
        )
```

Pros: Simplest approach. No serialization issues. All Python features work.
No new dependencies.

Cons: Re-executes entire history on each call — side effects run again.
Grows linearly with history length.

**Mitigation for stdout leakage:** Wrap prior code in a guard:

```python
# Only new code produces output; prior blocks set up state silently
import sys, io
_old_stdout = sys.stdout
sys.stdout = io.StringIO()
# ... prior code blocks ...
sys.stdout = _old_stdout
# ... new code block (produces output) ...
```

**WARNING — Side-effect replay is NOT mitigated by stdout suppression.**
Prior blocks that perform file writes, network calls, database mutations,
or other I/O will re-execute those side effects on every subsequent call.
Stdout suppression only hides `print()` output — it does not prevent or
guard against non-idempotent operations. This is a fundamental limitation
of the cumulative replay approach (Option C). Users must keep
side-effecting code in the final block or use Option A (persistent
process) when side effects are unavoidable.

#### 5.2.2 Recommended Approach: Option A (Persistent Process)

**Option A is the recommended approach.** It is the most robust for
true statefulness and is the standard approach used by Jupyter kernels
and similar systems:

- Variables, imports, and objects persist naturally
- No re-execution of side effects
- No serialization issues
- O(1) cost per call (not O(n) like Option C)

Option C (cumulative replay) has a fundamental side-effect replay
problem that cannot be fully mitigated. We recommend **going directly
to Option A** rather than shipping Option C as an interim MVP that
would accumulate technical debt and user-facing bugs.

If a simpler interim is needed before the persistent-process protocol
is ready, Option C may be used with the following restrictions:
- Documented as limited to **pure computation only** (variable setup,
  data transforms, aggregations)
- Side-effecting code (file writes, network calls, DB mutations) is
  explicitly unsupported and will produce incorrect results
- Clearly labeled as experimental / unstable

#### 5.2.3 Implementation Plan (Phase 1, if pursued)

1. **Unfreeze `stateful` in `ContainerCodeExecutor`:**

```python
# Remove frozen=True
stateful: bool = False
```

2. **Add a code history list:**

```python
_code_history: list[str] = []
```

3. **Modify `execute_code()`:**

**Critical invariant:** Code is appended to history **only after**
successful execution. A failing code block must never be replayed.

```python
def execute_code(self, invocation_context, code_execution_input):
    code = code_execution_input.code

    if self.stateful:
        # Build cumulative script from prior SUCCESSFUL blocks
        setup_code = '\n'.join(
            f'# --- Block {i} ---\n{block}'
            for i, block in enumerate(self._code_history)
        )
        # Suppress stdout for prior blocks
        full_code = (
            'import sys as _sys, io as _io\n'
            '_sys.stdout = _io.StringIO()\n'
            f'{setup_code}\n'
            '_sys.stdout = _sys.__stdout__\n'
            f'{code}\n'
        )
    else:
        full_code = code

    exec_result = self._container.exec_run(
        ['python3', '-c', full_code],
        demux=True,
    )

    # Parse output
    stdout, stderr = self._parse_exec_output(exec_result)
    success = (exec_result.exit_code == 0)

    # ONLY append to history after confirmed success
    if self.stateful and success:
        self._code_history.append(code)

    return CodeExecutionResult(stdout=stdout, stderr=stderr)
```

4. **Add `reset_state()` method:**

```python
def reset_state(self):
    """Clears the execution history for stateful mode."""
    self._code_history.clear()
```

5. **Update the `__init__` validation:**

```python
# Remove the ValueError for stateful=True
# Keep optimize_data_file frozen
```

#### 5.2.4 Interaction with `execution_id`

The LLM flow layer uses `execution_id` (from `CodeExecutorContext`) to
identify stateful sessions. For `ContainerCodeExecutor`:

- Each `execution_id` maps to a separate code history
- Use a dict: `_code_histories: dict[str, list[str]] = {}`
- When `execution_id` is provided in `CodeExecutionInput`, use the
  corresponding history
- When `execution_id` is `None`, use default (empty) history

This aligns with how `VertexAiCodeExecutor` uses `session_id`.

**Gap: `RunSkillScriptTool` does not wire `execution_id`.**

Currently, `RunSkillScriptTool.run_async()` creates
`CodeExecutionInput(code=..., input_files=..., working_dir='.')` without
setting `execution_id`.
This means all skill script executions share the same (default) namespace
in a stateful executor, with no isolation between different skills or
invocations.

**Action items:**
1. `RunSkillScriptTool` should generate a **session-stable**
   `execution_id` scoped to skill + agent. The key must persist
   across turns so that stateful code history is preserved:
   ```python
   execution_id = f"skill:{skill_name}:{session.id}:{agent_name}"
   ```
   Using `invocation_id` would be incorrect here — it changes every
   turn, defeating statefulness. `session.id` is stable for the
   lifetime of the conversation.
2. Pass `execution_id` to `CodeExecutionInput`
3. This enables future stateful skill scripts where a skill can
   maintain state across multiple calls within the same session

This is tracked as part of the Phase 2 implementation plan.

### 5.3 Testing Plan

| Test | Description |
|------|-------------|
| `test_stateful_variable_persistence` | Define variable in call 1, access in call 2 |
| `test_stateful_import_persistence` | Import in call 1, use in call 2 |
| `test_stateful_no_stdout_leakage` | Prior blocks' print() should not appear in later output |
| `test_stateful_error_in_later_block` | Error in call 2 should not corrupt state |
| `test_stateful_reset` | `reset_state()` clears history |
| `test_stateless_unchanged` | Default `stateful=False` behavior unchanged |
| `test_execution_id_isolation` | Different `execution_id` values use separate histories |

---

## 6. Proposal 3: Security Hardening

### 6.1 Problem

`UnsafeLocalCodeExecutor` is the default executor for local development
because it requires no external dependencies. But it runs `exec()` in the
host Python process with full access to:

- The filesystem (read/write any file)
- Environment variables (including API keys, secrets)
- Network (outbound HTTP, DNS)
- The ADK process itself (`os.kill`, `sys.exit`)

This is a critical security concern when executing:
- LLM-generated code (prompt injection → arbitrary code execution)
- Third-party skill scripts (supply chain risk)
- User-provided code in multi-tenant deployments

### 6.2 Threat Model

| Threat | Impact | Current mitigation |
|--------|--------|--------------------|
| LLM generates malicious code | Full host compromise | None |
| Skill script reads secrets | Data exfiltration | None (documented warning only) |
| Infinite loop / fork bomb | DoS / resource exhaustion | Shell: `subprocess.run(timeout=N)` via `script_timeout`; Python: None (no timeout at any layer) |
| `sys.exit()` in script | Process termination | `RunSkillScriptTool` catches `SystemExit`: code 0 or `None` → success; non-zero → `EXECUTION_ERROR` with exit code in message |
| Long error messages | LLM context waste | Exception messages >200 chars truncated to `msg[:200] + "..."` |
| Network exfiltration | Data leak | None |
| File system manipulation | Data loss / corruption | Partial (temp-dir sandbox when `input_files`/`working_dir` set) |

### 6.3 Design

We propose a layered approach with three tiers of security:

#### 6.3.1 Tier 1: `UnsafeLocalCodeExecutor` Hardening (Quick Wins)

These changes improve safety without changing the fundamental architecture:

**A. Timeout support** (covered in Proposal 1)

**B. Restricted builtins (best-effort friction, NOT a security boundary):**

**Important caveat:** Builtin/module blocking in `exec()` is trivially
bypassed. Determined code can reach blocked functionality via:
- `object.__subclasses__()` → find `os._wrap_close` → access `os.system`
- `__builtins__.__dict__['__import__']('os')` (if `__builtins__` is a
  module, not a dict)
- Encoding tricks, `importlib` via `sys.modules`, etc.

This is explicitly **not a security control** — it is a speed bump that
catches accidental misuse and makes intentional abuse more visible. True
isolation requires `LocalSandboxCodeExecutor` (Tier 2) or containers
(Tier 3).

```python
_BLOCKED_BUILTINS = {
    'exec', 'eval', 'compile',  # Prevent meta-execution
    '__import__',               # Prevent arbitrary imports
    'open',                     # Prevent file access
    'breakpoint',               # Prevent debugger attach
}

_BLOCKED_MODULES = {
    'os', 'subprocess', 'shutil',   # System access
    'socket', 'http', 'urllib',     # Network access
    'ctypes', 'cffi',               # Native code
    'importlib',                    # Dynamic imports
}
```

**Trade-off:** This breaks legitimate use cases (e.g., data analysis scripts
that need `open()` for file I/O). It should be opt-in:

```python
class UnsafeLocalCodeExecutor(BaseCodeExecutor):
    restrict_builtins: bool = False
    """When True, block dangerous builtins (exec, eval, open,
    __import__). This is a best-effort friction layer, NOT a
    security boundary — determined code can bypass it. Use
    LocalSandboxCodeExecutor or ContainerCodeExecutor for
    actual isolation. Default False for backward compatibility."""
```

**C. Warning on first use:**

```python
import warnings

def execute_code(self, ...):
    if not self._warned:
        warnings.warn(
            'UnsafeLocalCodeExecutor runs code in the host '
            'process with NO isolation. Use '
            'ContainerCodeExecutor or GkeCodeExecutor for '
            'production deployments.',
            SecurityWarning,
            stacklevel=2,
        )
        self._warned = True
    ...
```

#### 6.3.2 Tier 2: `LocalSandboxCodeExecutor` (New, Recommended)

A new executor that provides meaningful isolation without requiring Docker
or cloud services:

**Approach: `subprocess` with resource limits**

```python
class LocalSandboxCodeExecutor(BaseCodeExecutor):
    """Executes Python code in a sandboxed subprocess.

    Provides isolation via:
    - Separate process (no shared memory with host)
    - Resource limits via ulimit (CPU time, memory)
    - Restricted environment variables
    - Optional chroot or tmpdir working directory
    """

    default_timeout_seconds: int = 30
    max_memory_mb: int = 256
    max_cpu_seconds: int = 30
    allowed_env_vars: list[str] = []

    def execute_code(self, invocation_context, code_execution_input):
        import platform
        import subprocess
        import sys
        import tempfile

        # Windows is out of scope (§2 Non-Goals).
        if platform.system() == 'Windows':
            raise NotImplementedError(
                'LocalSandboxCodeExecutor is not supported on '
                'Windows. Use ContainerCodeExecutor instead.'
            )

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=True
        ) as f:
            f.write(code_execution_input.code)
            f.flush()

            # Build restricted environment
            env = {k: os.environ[k] for k in self.allowed_env_vars
                   if k in os.environ}
            env['PATH'] = '/usr/bin:/usr/local/bin'

            input_t = code_execution_input.timeout_seconds
            timeout = (
                input_t if input_t is not None
                else self.default_timeout_seconds
            )
            if timeout is None:
                timeout = self.max_cpu_seconds

            # Prefer process_group (3.11+) over preexec_fn
            # (not fork-safe with threads).
            spawn_kwargs = {}
            if sys.version_info >= (3, 11):
                spawn_kwargs['process_group'] = 0
            else:
                # Fallback for 3.10; caveat: not fork-safe.
                # Guard resource import for platforms where
                # the module is unavailable.
                def _set_limits():
                    try:
                        import resource
                        resource.setrlimit(
                            resource.RLIMIT_CPU,
                            (self.max_cpu_seconds,) * 2,
                        )
                        mem = self.max_memory_mb * 1024 * 1024
                        resource.setrlimit(
                            resource.RLIMIT_AS, (mem, mem),
                        )
                    except (ImportError, OSError):
                        pass  # timeout-only enforcement
                spawn_kwargs['preexec_fn'] = _set_limits

            # Inline wrapper sets resource limits in the child
            # process. Guarded for missing resource module.
            limit_code = (
                f'try:\n'
                f'  import resource\n'
                f'  resource.setrlimit(resource.RLIMIT_CPU, '
                f'({self.max_cpu_seconds}, '
                f'{self.max_cpu_seconds}))\n'
                f'  resource.setrlimit(resource.RLIMIT_AS, '
                f'({self.max_memory_mb * 1024 * 1024}, '
                f'{self.max_memory_mb * 1024 * 1024}))\n'
                f'except (ImportError, OSError):\n'
                f'  pass\n'
            )
            cmd = [
                'python3', '-c',
                limit_code
                + f'exec(open({f.name!r}).read())',
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=tempfile.gettempdir(),
                **spawn_kwargs,
            )

        return CodeExecutionResult(
            stdout=result.stdout,
            stderr=result.stderr if result.returncode != 0
                   else '',
        )
```

**Platform considerations:**
- `resource.setrlimit` is Unix-only (Linux, macOS). On platforms
  where `resource` is unavailable, the inline `-c` wrapper must
  skip the `setrlimit` calls and rely on timeout-only enforcement.
  The code should guard with `try: import resource; ... except
  ImportError: pass` in the wrapper script.
- `process_group=0` requires Python 3.11+ (ADK supports >=3.10, so
  this must be gated with a version check or use `preexec_fn` as
  fallback on 3.10)
- Windows is out of scope (see §2 Non-Goals). The executor should
  raise `NotImplementedError` on Windows with a message directing
  users to `ContainerCodeExecutor`.

**Why `process_group` instead of `preexec_fn`:**
- `preexec_fn` is not fork-safe with threads — the Python docs warn
  that it can deadlock in multi-threaded programs because it runs
  between `fork()` and `exec()` while all parent thread locks are
  held. ADK executors may be called from async/threaded contexts.
- `process_group=0` (Python 3.11+) is fork-safe and places the child
  in its own process group, enabling clean `os.killpg()` on timeout.
- Resource limits are set via an inline `-c` wrapper script instead
  of `preexec_fn`, avoiding the fork-safety issue entirely.
- On Python 3.10 (ADK minimum is `>=3.10`), fall back to
  `preexec_fn=set_limits` with a documented caveat about thread safety.

**Dependencies:** None (stdlib only). This is the key advantage over
`ContainerCodeExecutor`.

**Limitations:**
- Less isolation than containers (shared filesystem, kernel)
- Cannot restrict network access without OS-level firewall rules
- `process_group` requires Python 3.11+; falls back to `preexec_fn`
  on 3.10 (ADK minimum is `>=3.10` per `pyproject.toml`)

#### 6.3.3 Tier 3: Promote `ContainerCodeExecutor` as Default

For production deployments, container-based isolation should be the
standard recommendation:

**A. Simplify setup with digest-pinned default image:**

```python
# Current: requires explicit image or docker_path
executor = ContainerCodeExecutor(image='python:3.11-slim')

# Proposed: auto-pull default image (digest-pinned)
executor = ContainerCodeExecutor()
```

**Default image should use a digest-pinned or versioned tag**, not a
mutable tag like `python:3.11-slim`. Mutable tags can change content
silently (e.g., security patches, Python micro-version bumps), leading
to non-reproducible behavior across environments and over time.

```python
# In ContainerCodeExecutor defaults:
_DEFAULT_IMAGE = (
    'python:3.11.11-slim@sha256:<pinned-digest>'
)
# Updated in ADK releases with tested digests
```

When an official `adk-code-executor` image is published, the default
should reference a versioned tag matching the ADK release:
```python
_DEFAULT_IMAGE = 'gcr.io/adk/code-executor:0.5.0'
```

**B. Pre-built ADK executor image:**

Create an official `adk-code-executor` Docker image with:
- Python 3.11+ slim base (digest-pinned in Dockerfile)
- Common data science libraries (pandas, numpy, matplotlib)
- Non-root user
- Read-only filesystem (writable `/tmp` only)
- No network access by default (`--network=none`)
- Versioned tags matching ADK releases

```dockerfile
FROM python:3.11.11-slim@sha256:<pinned-digest>
RUN pip install --no-cache-dir pandas numpy matplotlib
RUN useradd -m -s /bin/bash executor
USER executor
WORKDIR /home/executor
```

**C. Network isolation by default:**

```python
def __init_container(self):
    self._container = self._client.containers.run(
        image=self.image,
        detach=True,
        tty=True,
        network_mode='none',  # No network access
        read_only=True,       # Read-only filesystem
        tmpfs={'/tmp': 'size=100M'},  # Writable tmp
        mem_limit='512m',     # Memory limit
        cpu_period=100000,
        cpu_quota=50000,      # 50% of one CPU
    )
```

### 6.4 Recommendation Matrix

| Use Case | Recommended Executor | Why |
|----------|---------------------|-----|
| Local development | `LocalSandboxCodeExecutor` | No deps, basic isolation |
| Quick prototyping | `UnsafeLocalCodeExecutor` | Fastest setup, no isolation |
| CI/CD testing | `ContainerCodeExecutor` | Docker available in CI |
| Production (single tenant) | `ContainerCodeExecutor` | Good isolation, local |
| Production (multi-tenant) | `GkeCodeExecutor` | gVisor, per-execution isolation |
| Google Cloud | `AgentEngineSandboxCodeExecutor` | Managed, scalable |

### 6.5 Implementation Plan

| Phase | Action | Effort | Risk |
|-------|--------|--------|------|
| 1 | Add `SecurityWarning` to `UnsafeLocalCodeExecutor` | Small | None |
| 2 | Add `restrict_builtins` option | Small | Low |
| 3 | Implement `LocalSandboxCodeExecutor` | Medium | Low |
| 4 | Add default image + network isolation to `ContainerCodeExecutor` | Medium | Low |
| 5 | Create official `adk-code-executor` Docker image | Medium | Low |
| 6 | Update documentation and samples | Small | None |

---

## 7. Cross-Cutting Concerns

### 7.1 `BaseCodeExecutor` API Changes

All three proposals touch `BaseCodeExecutor`. The combined changes:

```python
class BaseCodeExecutor(BaseModel):
    # Existing fields (unchanged)
    optimize_data_file: bool = False
    stateful: bool = False
    error_retry_attempts: int = 2
    code_block_delimiters: List[tuple[str, str]] = [
        ('```tool_code\n', '\n```'),
        ('```python\n', '\n```'),
    ]
    execution_result_delimiters: tuple[str, str] = (
        '```tool_output\n', '\n```'
    )

    # NEW: Proposal 1
    default_timeout_seconds: Optional[int] = None
    """Default timeout applied when CodeExecutionInput.timeout_seconds
    is None. Subclasses may override. None = no timeout."""

    @abc.abstractmethod
    def execute_code(
        self,
        invocation_context: InvocationContext,
        code_execution_input: CodeExecutionInput,
    ) -> CodeExecutionResult: ...


@dataclasses.dataclass
class CodeExecutionInput:
    code: str
    input_files: list[File] = field(default_factory=list)
    execution_id: Optional[str] = None
    working_dir: Optional[str] = None
    timeout_seconds: Optional[int] = None  # NEW: per-invocation
    """Per-invocation timeout. Overrides executor default when set."""
```

### 7.2 Backward Compatibility

| Change | Backward compatible? | Migration needed? |
|--------|---------------------|------------------|
| `default_timeout_seconds` on base class | Yes (default `None`) | No |
| `timeout_seconds` on `CodeExecutionInput` | Yes (default `None`) | No |
| Unfreeze `stateful` on `ContainerCodeExecutor` | Yes (default `False`) | No |
| `SecurityWarning` on `UnsafeLocalCodeExecutor` | Yes (warning only) | No |
| New `LocalSandboxCodeExecutor` | Yes (additive) | No |
| `restrict_builtins` on `UnsafeLocalCodeExecutor` | Yes (default `False`) | No |
| Default image for `ContainerCodeExecutor` | Breaking (currently requires image/docker_path) | Minor |

### 7.3 Impact on `RunSkillScriptTool`

| Feature | Current workaround | After enhancements |
|---------|-------------------|-------------------|
| Shell timeout | Embedded `subprocess.run(timeout=N)` via `SkillToolset.script_timeout` (default 300s) | Keep as defense-in-depth + executor-level timeout |
| Python timeout | **None** — `runpy.run_path()` runs inline in `exec()` with no timeout at any layer | Executor-level timeout handles it; tool should also set `CodeExecutionInput.timeout_seconds` |
| Isolation | Partial temp-dir sandbox (input_files/working_dir) + `_execution_lock` for stdout/cwd; no restriction on filesystem/network/env access | `LocalSandboxCodeExecutor` or container |
| Stateful scripts | Not supported (`execution_id` not wired) | Available via `ContainerCodeExecutor(stateful=True)` with `execution_id` |
| Output files | `CodeExecutionResult.output_files` silently dropped | Surfaced in tool response (§7.5.2) |
| System instructions | `SkillToolset.process_llm_request()` injects `DEFAULT_SKILL_SYSTEM_INSTRUCTION` + XML skill list | No change needed |
| Error truncation | Exception messages >200 chars truncated | Consider making threshold configurable |

### 7.4 Testing Strategy

**Current test coverage gaps:** Unit tests exist for
`UnsafeLocalCodeExecutor`, `GkeCodeExecutor`,
`AgentEngineSandboxCodeExecutor`, `BuiltInCodeExecutor`, and
`CodeExecutorContext`, but **no unit test file exists for
`ContainerCodeExecutor`** (`tests/unittests/code_executors/` has no
`test_container_code_executor.py`). Likewise, `LocalSandboxCodeExecutor`
is new and has no tests yet.

| Category | Approach | New tests needed |
|----------|----------|-----------------|
| Unit tests | Mock-based tests per executor | **Add `test_container_code_executor.py`**, add `test_local_sandbox_code_executor.py` |
| Integration tests | Real executor tests (like `RunSkillScriptTool` integration tests) | Add Docker-based container tests (CI-gated) |
| Timeout tests | Scripts with `time.sleep()` to verify enforcement | Per-executor timeout tests |
| Timeout kill fallback | Verify `PermissionError` from `os.kill` triggers container restart | Mock `os.kill` to raise `PermissionError`, assert `container.restart()` called and `CodeExecutionResult.stderr` contains timeout message |
| Timeout kill success | Verify `os.kill(host_pid)` path when permitted | Mock `exec_inspect` to return PID, assert `os.kill` called with correct signal |
| Timeout total failure | Verify both `os.kill` and `container.restart()` fail → unhealthy | Mock both to raise, assert `_healthy` is `False` and `stderr` contains "cleanup failed" |
| Timeout thread leak | Verify post-kill `join(2)` is called and warning logged if thread lingers | Mock thread to stay alive after kill, assert warning logged |
| Security tests | Scripts attempting blocked operations | `restrict_builtins` bypass attempts, env var leakage |
| Stateful tests | Multi-call sequences verifying variable persistence | Append-after-success, failure-does-not-poison, `execution_id` isolation |
| Stateful crash recovery | Verify error returned on REPL/container crash | Kill REPL mid-execution, assert error indicates state loss |
| Tool contract tests | Validate structured `RunSkillScriptTool` response schema | Assert `return_code`, `timed_out`, `status`, and error envelope consistency |
| Output propagation tests | Verify executor `output_files` are surfaced by tool | Assert tool response includes generated files/metadata |
| Args normalization tests | Verify deterministic mapping from JSON args to argv | Cover booleans, lists, positional args, and invalid types |
| `execution_id` wiring tests | Verify stable and scoped `execution_id` generation | Assert per-session/per-skill isolation and persistence semantics |

### 7.5 High-Priority `RunSkillScriptTool` Contract Enhancements

These improvements are additive and can be shipped before stateful container
support.

#### 7.5.1 Structured Execution Result Schema

Current output is largely free-form (`stdout`, `stderr`, derived `status`).
Add explicit structured fields:

```python
{
  "skill_name": "...",
  "script_path": "...",
  "status": "success|warning|error",
  "return_code": int | None,
  "timed_out": bool,
  "stdout": str,
  "stderr": str,
  "output_files": [...],  # see §7.5.2
}
```

Guidance:
- `status` should be derived from structured fields (`return_code`,
  `timed_out`, and stream presence), not treated as the source of truth.
- Tool-level validation/configuration errors should continue using explicit
  `error_code` values.

**Current status derivation and its asymmetry:**

The current implementation determines status purely from stream presence:
```python
if stderr and not stdout:
    status = "error"
elif stderr:
    status = "warning"
else:
    status = "success"
```

This creates an asymmetry between script types:

- **Shell scripts:** Non-zero `returncode` from the JSON envelope causes
  synthesized stderr (`"Exit code {rc}"`), so return codes **indirectly**
  influence status. A shell script that fails silently (non-zero exit but
  no stderr) still gets `"error"` status.
- **Python scripts:** There is **no return code extraction**. A Python
  script that exits cleanly but writes warnings to stderr (common in
  data science libraries) would be classified as `"error"` or `"warning"`
  even if it succeeded. Conversely, a Python script that silently produces
  incorrect output would get `"success"` status.

The proposed `return_code` field resolves this by providing a uniform
source of truth. For Python scripts, this would require either:
(a) wrapping the `runpy.run_path()` call to capture the exit code, or
(b) treating any non-exception completion as `return_code = 0`.

#### 7.5.2 Propagate `output_files` and Artifact Metadata

`CodeExecutionResult.output_files` should be surfaced in the tool response.
This is critical for scripts that generate reports, transformed datasets, or
intermediate artifacts.

Minimum expected shape:

```python
output_files = [
  {
    "name": str,
    "mime_type": str | None,
    # optional future fields:
    # "artifact_id": str,
    # "path": str,
  }
]
```

#### 7.5.3 `execution_id` Wiring in `RunSkillScriptTool`

Even before full stateful container support, wire a deterministic
`execution_id` to avoid ambiguous namespaces in stateful-capable executors.

Recommended key shape:

```python
execution_id = (
    f"skill:{skill_name}:session:{session_id}:agent:{agent_name}"
)
```

Rules:
- Stable across turns within the same session.
- Scoped by skill and agent.
- Never derived from `invocation_id` (too short-lived).

#### 7.5.4 Script Args Normalization Contract

**Current behavior:** The implementation uses a simple `str(v)` conversion
for all argument values, with no type-aware normalization:

```python
# Both Python and shell paths use the same logic:
for k, v in script_args.items():
    argv_list.extend([f"--{k}", str(v)])
```

This means:
- `{"verbose": true}` → `["--verbose", "True"]` (string, not a flag)
- `{"flag": false}` → `["--flag", "False"]` (not omitted)
- `{"items": [1, 2, 3]}` → `["--items", "[1, 2, 3]"]` (repr of list)
- `{"count": 42}` → `["--count", "42"]` (correct)
- Nested objects → `["--config", "{'a': 1}"]` (repr, not useful)

Additionally, `args` type is validated to be a `dict` — non-dict values
(strings, lists, integers, booleans) return `INVALID_ARGS_TYPE` error.

**Proposed rules** (define deterministic mapping from JSON args to argv):
- `str|int|float` -> `--key value`
- `true` -> `--key` (flag only, no value)
- `false|None` -> omit entirely
- `list[...]` -> repeated `--key value` entries
- Optional reserved key for positional args (for example, `_positional`)
- Reject nested objects with explicit validation error

This reduces LLM-side ambiguity and improves replay/debug stability.

**Migration note:** Changing boolean handling from `"--key True"` to
`"--key"` (flag) is a behavioral change. Existing skill scripts that
parse `--verbose True` as a string value would break. The migration
should be opt-in or gated behind a version flag.

---

## 8. Implementation Roadmap

### 8.1 Priority-Ordered Plan

#### Phase 1 (P0): Timeout Foundation (3-4 days)

1. Add `timeout_seconds: Optional[int] = None` to `CodeExecutionInput`
2. Add `default_timeout_seconds: Optional[int] = None` to
   `BaseCodeExecutor`
3. Implement thread-based timeout in `UnsafeLocalCodeExecutor`
4. Implement Docker exec kill timeout in `ContainerCodeExecutor`
   (including `_healthy` guard, post-restart readiness validation,
   and post-kill thread join)
5. Add public `reinitialize()` method to `ContainerCodeExecutor`:
   stops the current container (if any), creates a new one, runs
   readiness check, and sets `_healthy = True`. This is the
   documented recovery path when `_healthy` is `False`. Callable
   by users or by higher-level retry logic.
6. Migrate `GkeCodeExecutor.timeout_seconds` to `default_timeout_seconds`
7. Add timeout tests for each executor
8. Update `RunSkillScriptTool` to set per-invocation timeout via
   `CodeExecutionInput.timeout_seconds`

#### Phase 2 (P1): `RunSkillScriptTool` Contract Hardening (2-3 days)

1. Add structured response fields: `return_code`, `timed_out`,
   schema-stable status semantics
2. Surface executor `output_files` in tool output
3. Define and implement args normalization contract
4. Add deterministic `execution_id` wiring for tool calls
5. Add tool-level contract tests (schema, args, output propagation,
   `execution_id` isolation behavior)

#### Phase 3 (P0): Security Hardening (5-7 days)

1. Add `SecurityWarning` to `UnsafeLocalCodeExecutor`
2. Add `restrict_builtins` option (documented as best-effort friction)
3. Implement `LocalSandboxCodeExecutor` (using `process_group`, not
   `preexec_fn`)
4. Add digest-pinned default image to `ContainerCodeExecutor`
5. Add network isolation defaults to `ContainerCodeExecutor`
6. Create official `adk-code-executor` Docker image (versioned tags)
7. Update all samples to recommend secure executors
8. Add security-focused tests

#### Phase 4 (P2): Stateful Container (5-8 days)

Implement Option A (persistent process) directly, as recommended in
§5.2.2. This avoids the side-effect replay problems of Option C.

1. Unfreeze `stateful` on `ContainerCodeExecutor`
2. Design persistent-process protocol: sentinel-delimited I/O for
   output boundaries, error detection, and process health checks
3. Implement persistent Python REPL management (start, send code,
   read output, detect crash/restart)
4. Add `execution_id`-based session isolation (one REPL per
   `execution_id`)
5. Wire `execution_id` in `RunSkillScriptTool`
6. Add `reset_state()` method (kills and restarts the REPL)
7. Add stateful execution tests (variable persistence, crash recovery,
   `execution_id` isolation)
8. Update samples and documentation

### Total estimated effort: 15-22 days

---

## 9. Open Questions

1. **What I/O boundary protocol should the persistent REPL use?**
   The roadmap targets Option A (persistent process) directly. The
   key design question is how to delimit output for each code block:
   sentinel strings in stdout, JSON-envelope protocol, or a side
   channel (e.g., file-based result). Sentinel strings are simplest
   but can collide with user output. Decision: spike during Phase 2.

2. **Should `LocalSandboxCodeExecutor` support stateful execution?**
   Subprocess-based execution is inherently stateless. Stateful support
   would require the same workarounds as `ContainerCodeExecutor` (Option
   C). We recommend keeping it stateless in the initial implementation.

3. **Should we deprecate `UnsafeLocalCodeExecutor`?**
   Not immediately. It's useful for zero-dependency development. But the
   `SecurityWarning` and documentation should steer users toward safer
   alternatives for anything beyond local prototyping.

4. **How should `ContainerCodeExecutor` handle container/REPL crashes
   in stateful mode?**
   If the container crashes (OOM, segfault) or the persistent REPL
   dies, in-process state is lost. The executor returns an error
   indicating state loss and lets the caller handle recovery (e.g.,
   start a new session). Automatic replay is not attempted because
   prior code blocks may have had side effects that should not be
   re-executed (consistent with §4.2.3 recovery policy).

---

## 10. References

- [BaseCodeExecutor](../../src/google/adk/code_executors/base_code_executor.py)
- [ContainerCodeExecutor](../../src/google/adk/code_executors/container_code_executor.py)
- [UnsafeLocalCodeExecutor](../../src/google/adk/code_executors/unsafe_local_code_executor.py)
- [GkeCodeExecutor](../../src/google/adk/code_executors/gke_code_executor.py)
- [VertexAiCodeExecutor](../../src/google/adk/code_executors/vertex_ai_code_executor.py)
- [AgentEngineSandboxCodeExecutor](../../src/google/adk/code_executors/agent_engine_sandbox_code_executor.py)
- [RunSkillScriptTool](../../src/google/adk/tools/skill_toolset.py)
- [Code Execution Flow](../../src/google/adk/flows/llm_flows/_code_execution.py)
- [PR #4575 — RunSkillScriptTool](https://github.com/google/adk-python/pull/4575)
