# ADK Code Executor Enhancements — Design Document

**Authors:** haiyuancao, Claude Code
**Date:** 2026-02-21
**Status:** Draft
**Tracking:** Related to PR #4575 (ExecuteSkillScriptTool)

---

## 1. Motivation

The ADK code executor infrastructure (`src/google/adk/code_executors/`) is
the backbone for both LLM-driven code execution and skill script execution.
A review of the current implementations reveals three critical gaps that
limit production readiness:

1. **No uniform timeout enforcement** — Only `GkeCodeExecutor` has a
   `timeout_seconds` field. All other executors can hang indefinitely on
   malicious, buggy, or slow code. The `ExecuteSkillScriptTool` works
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

## 2. Non-Goals & Invariants

The following are explicitly **out of scope** for this design:

1. **Full sandboxing of `UnsafeLocalCodeExecutor`** — The restricted
   builtins mechanism (Tier 1, §6.3.1-B) is a best-effort friction layer,
   not a security boundary. Any determined code can bypass it via
   `object.__subclasses__()`, `importlib` through `__builtins__`, etc.
   True isolation requires a process or container boundary.

2. **Idempotent replay of side-effecting code** — The stateful
   `ContainerCodeExecutor` (Proposal 2) replays prior code blocks.
   Code with non-idempotent side effects (file writes, network calls,
   database mutations) is **not supported** in stateful replay mode.
   The design suppresses stdout but cannot suppress arbitrary I/O.
   Users must keep side-effecting code in the final block or use the
   persistent-process approach (Phase 2 / Option A).

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
  Concurrent `execute_code()` calls on the same instance require external
  synchronization.

---

## 3. Current State

### 3.1 Executor Landscape

| Executor | Stateful | Timeout | Isolation | Dependencies |
|----------|----------|---------|-----------|-------------|
| `UnsafeLocalCodeExecutor` | No (frozen) | None | None | None |
| `ContainerCodeExecutor` | No (frozen) | None | Docker container | `docker` |
| `GkeCodeExecutor` | No (ephemeral) | `timeout_seconds=300` | gVisor sandbox | `kubernetes` |
| `VertexAiCodeExecutor` | Allowed | None | Vertex AI Extension | `vertexai` |
| `AgentEngineSandboxCodeExecutor` | Allowed | None | Vertex AI Sandbox | `vertexai` |
| `BuiltInCodeExecutor` | N/A | N/A | Gemini model | `google-genai` |

### 3.2 Base Class Contract

```python
class BaseCodeExecutor(BaseModel):
    stateful: bool = False
    error_retry_attempts: int = 2
    code_block_delimiters: List[tuple[str, str]]
    execution_result_delimiters: tuple[str, str]

    @abc.abstractmethod
    def execute_code(
        self,
        invocation_context: InvocationContext,
        code_execution_input: CodeExecutionInput,
    ) -> CodeExecutionResult: ...
```

### 3.3 Data Model

```python
@dataclasses.dataclass
class CodeExecutionInput:
    code: str
    input_files: list[File] = field(default_factory=list)
    execution_id: Optional[str] = None  # For stateful execution

@dataclasses.dataclass
class CodeExecutionResult:
    stdout: str = ''
    stderr: str = ''
    output_files: list[File] = field(default_factory=list)
```

### 3.4 How Executors Are Used

The primary consumer is `_code_execution.py` in the LLM flows layer:

1. **Pre-processor**: Extracts data files, runs preprocessing code
2. **Post-processor**: Extracts code blocks from LLM responses, executes
   them, feeds results back to the LLM
3. **Stateful support**: Uses `execution_id` (from `CodeExecutorContext`)
   to maintain state across calls when `stateful=True`

`ExecuteSkillScriptTool` is a secondary consumer that calls
`execute_code()` directly with generated Python code wrapping skill scripts.

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
- Callers can override per-call (e.g., `ExecuteSkillScriptTool` sets
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
        try:
            info = self._client.api.exec_inspect(exec_id)
            host_pid = info.get('Pid', 0)
            if host_pid > 0:
                os.kill(host_pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass  # Process already exited
        except Exception:
            # Last resort: restart the container
            try:
                self._container.restart(timeout=1)
            except Exception:
                pass

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

3. **Container restart as last resort** — If `os.kill` fails (e.g.,
   insufficient permissions when Docker runs rootless), restart the
   container. This is the most reliable fallback but destroys
   in-container state.

**Permissions:** `os.kill()` on the host PID requires the ADK process
to run as the same user that started the container (or as root). This
is the normal case for local Docker usage. For rootless Docker or
restricted environments, the container-restart fallback applies.

**Recovery after timeout:**
- **Stateless mode:** No recovery needed. The killed process is gone;
  the next `exec_run` starts a fresh process in the same container.
- **Stateful mode:** The timed-out block is NOT appended to history
  (append-after-success invariant). If the container was restarted
  as fallback, the executor must detect this and replay the
  accumulated history on the next call.

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

### 4.4 Impact on `ExecuteSkillScriptTool`

Once `BaseCodeExecutor` has native timeout support, the
`ExecuteSkillScriptTool` can optionally delegate timeout enforcement to
the executor rather than embedding it in generated shell wrapper code.
However, the shell wrapper timeout (`subprocess.run(timeout=N)`) should
be kept as defense-in-depth — it catches the subprocess even if the
executor timeout fails.

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

**Gap: `ExecuteSkillScriptTool` does not wire `execution_id`.**

Currently, `ExecuteSkillScriptTool.run_async()` creates
`CodeExecutionInput(code=prepared_code)` without setting `execution_id`.
This means all skill script executions share the same (default) namespace
in a stateful executor, with no isolation between different skills or
invocations.

**Action items:**
1. `ExecuteSkillScriptTool` should generate a **session-stable**
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
| Infinite loop / fork bomb | DoS / resource exhaustion | None (no timeout) |
| `sys.exit()` in script | Process termination | Partial (`SystemExit` catch in `ExecuteSkillScriptTool`) |
| Network exfiltration | Data leak | None |
| File system manipulation | Data loss / corruption | None |

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
        import subprocess
        import tempfile

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

            import sys
            # Prefer process_group (3.11+) over preexec_fn
            # (not fork-safe with threads).
            spawn_kwargs = {}
            if sys.version_info >= (3, 11):
                spawn_kwargs['process_group'] = 0
            else:
                # Fallback for 3.10; caveat: not fork-safe
                def _set_limits():
                    import resource
                    resource.setrlimit(
                        resource.RLIMIT_CPU,
                        (self.max_cpu_seconds,) * 2,
                    )
                    mem = self.max_memory_mb * 1024 * 1024
                    resource.setrlimit(
                        resource.RLIMIT_AS, (mem, mem),
                    )
                spawn_kwargs['preexec_fn'] = _set_limits

            # Guard resource import for platforms where
            # it is unavailable (falls back to timeout-only).
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
    code_block_delimiters: List[tuple[str, str]] = [...]
    execution_result_delimiters: tuple[str, str] = (...)

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

### 7.3 Impact on `ExecuteSkillScriptTool`

| Feature | Current workaround | After enhancements |
|---------|-------------------|-------------------|
| Shell timeout | Embedded `subprocess.run(timeout=N)` | Keep as defense-in-depth |
| Python timeout | None | Executor-level timeout handles it |
| Isolation | Documentation warning only | `LocalSandboxCodeExecutor` or container |
| Stateful scripts | Not supported | Available via `ContainerCodeExecutor(stateful=True)` |

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
| Integration tests | Real executor tests (like `ExecuteSkillScriptTool` integration tests) | Add Docker-based container tests (CI-gated) |
| Timeout tests | Scripts with `time.sleep()` to verify enforcement | Per-executor timeout tests |
| Security tests | Scripts attempting blocked operations | `restrict_builtins` bypass attempts, env var leakage |
| Stateful tests | Multi-call sequences verifying variable persistence | Append-after-success, failure-does-not-poison, `execution_id` isolation |

---

## 8. Implementation Roadmap

### Phase 1: Timeout (3-4 days)

1. Add `timeout_seconds: Optional[int] = None` to `CodeExecutionInput`
2. Add `default_timeout_seconds: Optional[int] = None` to
   `BaseCodeExecutor`
3. Implement thread-based timeout in `UnsafeLocalCodeExecutor`
4. Implement Docker exec kill timeout in `ContainerCodeExecutor`
5. Migrate `GkeCodeExecutor.timeout_seconds` to `default_timeout_seconds`
6. Add timeout tests for each executor
7. Update `ExecuteSkillScriptTool` to set per-invocation timeout via
   `CodeExecutionInput.timeout_seconds`

### Phase 2: Stateful Container (3-5 days)

1. Unfreeze `stateful` on `ContainerCodeExecutor`
2. Implement cumulative code history with stdout suppression
   (append-after-success invariant)
3. Add `execution_id`-based history isolation
4. Wire `execution_id` in `ExecuteSkillScriptTool`
5. Add `reset_state()` method
6. Add stateful execution tests (including failure-does-not-poison test)
7. Update samples and documentation
8. Evaluate persistent-process approach (Option A) for Phase 2b

### Phase 3: Security Hardening (5-7 days)

1. Add `SecurityWarning` to `UnsafeLocalCodeExecutor`
2. Add `restrict_builtins` option (documented as best-effort friction)
3. Implement `LocalSandboxCodeExecutor` (using `process_group`, not
   `preexec_fn`)
4. Add digest-pinned default image to `ContainerCodeExecutor`
5. Add network isolation defaults to `ContainerCodeExecutor`
6. Create official `adk-code-executor` Docker image (versioned tags)
7. Update all samples to recommend secure executors
8. Add security-focused tests

### Total estimated effort: 11-16 days

---

## 9. Open Questions

1. **Should we skip Phase 1 (cumulative replay) and go straight to
   Phase 2 (persistent process) for stateful execution?**
   The side-effect replay problem is fundamental to Option C. If the
   persistent-process I/O boundary protocol can be solved with
   reasonable complexity (e.g., sentinel-delimited output), the MVP
   phase may not be worth the tech debt. Decision: Evaluate during
   Phase 2 planning.

2. **Should `LocalSandboxCodeExecutor` support stateful execution?**
   Subprocess-based execution is inherently stateless. Stateful support
   would require the same workarounds as `ContainerCodeExecutor` (Option
   C). We recommend keeping it stateless in the initial implementation.

3. **Should we deprecate `UnsafeLocalCodeExecutor`?**
   Not immediately. It's useful for zero-dependency development. But the
   `SecurityWarning` and documentation should steer users toward safer
   alternatives for anything beyond local prototyping.

4. **How should `ContainerCodeExecutor` handle container crashes in
   stateful mode?**
   If the container crashes (OOM, segfault), the code history is lost.
   Options: (a) re-create container and replay history, (b) return error
   and let user restart, (c) persist history to host volume. Recommend
   (b) for simplicity.

---

## 10. References

- [BaseCodeExecutor](../../src/google/adk/code_executors/base_code_executor.py)
- [ContainerCodeExecutor](../../src/google/adk/code_executors/container_code_executor.py)
- [UnsafeLocalCodeExecutor](../../src/google/adk/code_executors/unsafe_local_code_executor.py)
- [GkeCodeExecutor](../../src/google/adk/code_executors/gke_code_executor.py)
- [VertexAiCodeExecutor](../../src/google/adk/code_executors/vertex_ai_code_executor.py)
- [AgentEngineSandboxCodeExecutor](../../src/google/adk/code_executors/agent_engine_sandbox_code_executor.py)
- [ExecuteSkillScriptTool](../../src/google/adk/tools/skill_toolset.py)
- [Code Execution Flow](../../src/google/adk/flows/llm_flows/_code_execution.py)
- [PR #4575 — ExecuteSkillScriptTool](https://github.com/google/adk-python/pull/4575)
