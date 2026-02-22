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

## 2. Current State

### 2.1 Executor Landscape

| Executor | Stateful | Timeout | Isolation | Dependencies |
|----------|----------|---------|-----------|-------------|
| `UnsafeLocalCodeExecutor` | No (frozen) | None | None | None |
| `ContainerCodeExecutor` | No (frozen) | None | Docker container | `docker` |
| `GkeCodeExecutor` | No (ephemeral) | `timeout_seconds=300` | gVisor sandbox | `kubernetes` |
| `VertexAiCodeExecutor` | Allowed | None | Vertex AI Extension | `vertexai` |
| `AgentEngineSandboxCodeExecutor` | Allowed | None | Vertex AI Sandbox | `vertexai` |
| `BuiltInCodeExecutor` | N/A | N/A | Gemini model | `google-genai` |

### 2.2 Base Class Contract

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

### 2.3 Data Model

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

### 2.4 How Executors Are Used

The primary consumer is `_code_execution.py` in the LLM flows layer:

1. **Pre-processor**: Extracts data files, runs preprocessing code
2. **Post-processor**: Extracts code blocks from LLM responses, executes
   them, feeds results back to the LLM
3. **Stateful support**: Uses `execution_id` (from `CodeExecutorContext`)
   to maintain state across calls when `stateful=True`

`ExecuteSkillScriptTool` is a secondary consumer that calls
`execute_code()` directly with generated Python code wrapping skill scripts.

---

## 3. Proposal 1: Uniform Timeout Support

### 3.1 Problem

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

### 3.2 Design

#### 3.2.1 Add `timeout_seconds` to `BaseCodeExecutor`

```python
class BaseCodeExecutor(BaseModel):
    timeout_seconds: Optional[int] = None
    """Maximum execution time in seconds. None means no timeout
    (executor default behavior). Subclasses should enforce this
    in their execute_code() implementation."""
```

**Why `Optional[int]` with `None` default:**
- Backward compatible — existing code that doesn't set it works unchanged
- Allows subclasses to define their own defaults
- `None` means "use executor-specific default or no timeout"

#### 3.2.2 `UnsafeLocalCodeExecutor` — Thread-Based Timeout

`exec()` cannot be interrupted from the same thread. The solution is to
run it in a separate thread with a join timeout:

```python
import threading

def execute_code(self, invocation_context, code_execution_input):
    timeout = self.timeout_seconds
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

#### 3.2.3 `ContainerCodeExecutor` — Docker `exec_run` Timeout

Docker's `exec_run` does not natively support a timeout, but we can use
the Docker API's exec endpoint with a socket timeout:

```python
def execute_code(self, invocation_context, code_execution_input):
    timeout = self.timeout_seconds

    exec_result = self._container.exec_run(
        ['python3', '-c', code_execution_input.code],
        demux=True,
        # Docker SDK does not support exec_run timeout directly.
        # Use socket_timeout on the client instead.
    )
    ...
```

**Better approach:** Use `threading.Timer` to kill the exec if it exceeds
the timeout:

```python
import threading

def execute_code(self, invocation_context, code_execution_input):
    timeout = self.timeout_seconds

    # Create the exec instance
    exec_id = self._client.api.exec_create(
        self._container.id,
        ['python3', '-c', code_execution_input.code],
    )['Id']

    # Start a timer to kill the exec if it exceeds timeout
    timer = None
    timed_out = threading.Event()
    if timeout is not None:
        def _kill():
            timed_out.set()
            # Kill the exec'd process inside the container
            self._container.exec_run(
                ['kill', '-9', '-1'],  # kill all procs
                detach=True,
            )
        timer = threading.Timer(timeout, _kill)
        timer.start()

    try:
        output = self._client.api.exec_start(exec_id, demux=True)
    finally:
        if timer is not None:
            timer.cancel()

    if timed_out.is_set():
        return CodeExecutionResult(
            stderr=f'Execution timed out after {timeout}s'
        )
    # ... parse output as before
```

**Alternative — simpler approach:** Wrap `exec_run` in a thread with
join timeout (same pattern as `UnsafeLocalCodeExecutor`). Simpler, and
the container process can be cleaned up on next execution.

**Recommendation:** Use the thread + join approach for consistency across
executors. Add a follow-up to use Docker's native exec kill for more
robust cleanup.

#### 3.2.4 `GkeCodeExecutor` — Already Implemented

`GkeCodeExecutor` already has `timeout_seconds: int = 300` applied to the
K8s watch API. Migrate it to use the base class field:

```python
class GkeCodeExecutor(BaseCodeExecutor):
    timeout_seconds: int = 300  # Override base default
```

No behavioral change needed.

#### 3.2.5 Remote Executors (Vertex AI, Agent Engine)

These executors delegate to Google Cloud APIs that have their own internal
timeouts. Adding client-side timeout is still valuable as a safety net:

- Wrap the API call in `asyncio.wait_for()` or `threading.Timer`
- Return `CodeExecutionResult(stderr='...')` on timeout
- Log a warning that the server-side execution may still be running

### 3.3 Migration Plan

| Phase | Action | Risk |
|-------|--------|------|
| 1 | Add `timeout_seconds: Optional[int] = None` to `BaseCodeExecutor` | None (backward compatible) |
| 2 | Implement in `UnsafeLocalCodeExecutor` (thread + join) | Low |
| 3 | Implement in `ContainerCodeExecutor` (thread + join) | Low |
| 4 | Migrate `GkeCodeExecutor.timeout_seconds` to use base field | None |
| 5 | Add client-side timeout to remote executors | Low |

### 3.4 Impact on `ExecuteSkillScriptTool`

Once `BaseCodeExecutor` has native timeout support, the
`ExecuteSkillScriptTool` can optionally delegate timeout enforcement to
the executor rather than embedding it in generated shell wrapper code.
However, the shell wrapper timeout (`subprocess.run(timeout=N)`) should
be kept as defense-in-depth — it catches the subprocess even if the
executor timeout fails.

---

## 4. Proposal 2: Stateful `ContainerCodeExecutor`

### 4.1 Problem

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

### 4.2 Design

#### 4.2.1 Architecture

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

**Option C: Shared Globals File (Simple) — RECOMMENDED**

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

**Mitigation for side effects:** Wrap prior code in a guard:

```python
# Only new code produces output; prior blocks set up state silently
import sys, io
_old_stdout = sys.stdout
sys.stdout = io.StringIO()
# ... prior code blocks ...
sys.stdout = _old_stdout
# ... new code block (produces output) ...
```

#### 4.2.2 Recommended Approach

**Option A (Persistent Process)** is the most robust for true statefulness
and is the standard approach used by Jupyter kernels and similar systems.
Despite its complexity, it provides the best user experience:

- Variables, imports, and objects persist naturally
- No re-execution of side effects
- No serialization issues
- O(1) cost per call (not O(n) like Option C)

However, implementing a full REPL protocol is a significant engineering
effort. We recommend a **phased approach**:

**Phase 1 (MVP):** Option C (cumulative file) with stdout suppression for
prior blocks. Simple, works for the common case (data analysis, variable
setup).

**Phase 2 (Full):** Option A (persistent process) with a proper
execution protocol using sentinel markers for output boundaries.

#### 4.2.3 Implementation Plan (Phase 1)

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

```python
def execute_code(self, invocation_context, code_execution_input):
    code = code_execution_input.code

    if self.stateful:
        # Build cumulative script
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
        self._code_history.append(code)
    else:
        full_code = code

    exec_result = self._container.exec_run(
        ['python3', '-c', full_code],
        demux=True,
    )
    # ... parse output as before
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

#### 4.2.4 Interaction with `execution_id`

The LLM flow layer uses `execution_id` (from `CodeExecutorContext`) to
identify stateful sessions. For `ContainerCodeExecutor`:

- Each `execution_id` maps to a separate code history
- Use a dict: `_code_histories: dict[str, list[str]] = {}`
- When `execution_id` is provided in `CodeExecutionInput`, use the
  corresponding history
- When `execution_id` is `None`, use default (empty) history

This aligns with how `VertexAiCodeExecutor` uses `session_id`.

### 4.3 Testing Plan

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

## 5. Proposal 3: Security Hardening

### 5.1 Problem

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

### 5.2 Threat Model

| Threat | Impact | Current mitigation |
|--------|--------|--------------------|
| LLM generates malicious code | Full host compromise | None |
| Skill script reads secrets | Data exfiltration | None (documented warning only) |
| Infinite loop / fork bomb | DoS / resource exhaustion | None (no timeout) |
| `sys.exit()` in script | Process termination | Partial (`SystemExit` catch in `ExecuteSkillScriptTool`) |
| Network exfiltration | Data leak | None |
| File system manipulation | Data loss / corruption | None |

### 5.3 Design

We propose a layered approach with three tiers of security:

#### 5.3.1 Tier 1: `UnsafeLocalCodeExecutor` Hardening (Quick Wins)

These changes improve safety without changing the fundamental architecture:

**A. Timeout support** (covered in Proposal 1)

**B. Restricted builtins:**

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
    __import__). Default False for backward compatibility."""
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

#### 5.3.2 Tier 2: `LocalSandboxCodeExecutor` (New, Recommended)

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

    timeout_seconds: int = 30
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

            # Set resource limits via preexec_fn
            def set_limits():
                import resource
                # CPU time limit
                resource.setrlimit(
                    resource.RLIMIT_CPU,
                    (self.max_cpu_seconds, self.max_cpu_seconds)
                )
                # Memory limit
                mem_bytes = self.max_memory_mb * 1024 * 1024
                resource.setrlimit(
                    resource.RLIMIT_AS,
                    (mem_bytes, mem_bytes)
                )

            result = subprocess.run(
                ['python3', f.name],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env=env,
                preexec_fn=set_limits,  # Unix only
                cwd=tempfile.gettempdir(),
            )

        return CodeExecutionResult(
            stdout=result.stdout,
            stderr=result.stderr if result.returncode != 0
                   else '',
        )
```

**Platform considerations:**
- `resource.setrlimit` is Unix-only (Linux, macOS)
- On Windows, use `subprocess.CREATE_NO_WINDOW` and
  `subprocess.Popen` with `creationflags` for job object limits
- Fallback to timeout-only on platforms without `resource` module

**Dependencies:** None (stdlib only). This is the key advantage over
`ContainerCodeExecutor`.

**Limitations:**
- Less isolation than containers (shared filesystem, kernel)
- `preexec_fn` is not fork-safe with threads (use `process_group` on
  Python 3.11+)
- Cannot restrict network access without OS-level firewall rules

#### 5.3.3 Tier 3: Promote `ContainerCodeExecutor` as Default

For production deployments, container-based isolation should be the
standard recommendation:

**A. Simplify setup:**

```python
# Current: requires explicit image or docker_path
executor = ContainerCodeExecutor(image='python:3.11-slim')

# Proposed: auto-pull default image
executor = ContainerCodeExecutor()  # Uses python:3.11-slim
```

**B. Pre-built ADK executor image:**

Create an official `adk-code-executor` Docker image with:
- Python 3.11+ slim base
- Common data science libraries (pandas, numpy, matplotlib)
- Non-root user
- Read-only filesystem (writable `/tmp` only)
- No network access by default (`--network=none`)

```dockerfile
FROM python:3.11-slim
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

### 5.4 Recommendation Matrix

| Use Case | Recommended Executor | Why |
|----------|---------------------|-----|
| Local development | `LocalSandboxCodeExecutor` | No deps, basic isolation |
| Quick prototyping | `UnsafeLocalCodeExecutor` | Fastest setup, no isolation |
| CI/CD testing | `ContainerCodeExecutor` | Docker available in CI |
| Production (single tenant) | `ContainerCodeExecutor` | Good isolation, local |
| Production (multi-tenant) | `GkeCodeExecutor` | gVisor, per-execution isolation |
| Google Cloud | `AgentEngineSandboxCodeExecutor` | Managed, scalable |

### 5.5 Implementation Plan

| Phase | Action | Effort | Risk |
|-------|--------|--------|------|
| 1 | Add `SecurityWarning` to `UnsafeLocalCodeExecutor` | Small | None |
| 2 | Add `restrict_builtins` option | Small | Low |
| 3 | Implement `LocalSandboxCodeExecutor` | Medium | Low |
| 4 | Add default image + network isolation to `ContainerCodeExecutor` | Medium | Low |
| 5 | Create official `adk-code-executor` Docker image | Medium | Low |
| 6 | Update documentation and samples | Small | None |

---

## 6. Cross-Cutting Concerns

### 6.1 `BaseCodeExecutor` API Changes

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
    timeout_seconds: Optional[int] = None
    """Maximum execution time in seconds. None = no timeout."""

    @abc.abstractmethod
    def execute_code(
        self,
        invocation_context: InvocationContext,
        code_execution_input: CodeExecutionInput,
    ) -> CodeExecutionResult: ...
```

### 6.2 Backward Compatibility

| Change | Backward compatible? | Migration needed? |
|--------|---------------------|------------------|
| `timeout_seconds` on base class | Yes (default `None`) | No |
| Unfreeze `stateful` on `ContainerCodeExecutor` | Yes (default `False`) | No |
| `SecurityWarning` on `UnsafeLocalCodeExecutor` | Yes (warning only) | No |
| New `LocalSandboxCodeExecutor` | Yes (additive) | No |
| `restrict_builtins` on `UnsafeLocalCodeExecutor` | Yes (default `False`) | No |
| Default image for `ContainerCodeExecutor` | Breaking (currently requires image/docker_path) | Minor |

### 6.3 Impact on `ExecuteSkillScriptTool`

| Feature | Current workaround | After enhancements |
|---------|-------------------|-------------------|
| Shell timeout | Embedded `subprocess.run(timeout=N)` | Keep as defense-in-depth |
| Python timeout | None | Executor-level timeout handles it |
| Isolation | Documentation warning only | `LocalSandboxCodeExecutor` or container |
| Stateful scripts | Not supported | Available via `ContainerCodeExecutor(stateful=True)` |

### 6.4 Testing Strategy

| Category | Approach |
|----------|----------|
| Unit tests | Mock-based tests for each executor (existing pattern) |
| Integration tests | Real executor tests (like the ones added for `ExecuteSkillScriptTool`) |
| Timeout tests | Scripts with `time.sleep()` to verify timeout enforcement |
| Security tests | Scripts attempting blocked operations to verify restrictions |
| Stateful tests | Multi-call sequences verifying variable persistence |

---

## 7. Implementation Roadmap

### Phase 1: Timeout (2-3 days)

1. Add `timeout_seconds: Optional[int] = None` to `BaseCodeExecutor`
2. Implement thread-based timeout in `UnsafeLocalCodeExecutor`
3. Implement thread-based timeout in `ContainerCodeExecutor`
4. Migrate `GkeCodeExecutor.timeout_seconds` to base class field
5. Add timeout tests for each executor
6. Update `ExecuteSkillScriptTool` to set executor timeout when available

### Phase 2: Stateful Container (3-5 days)

1. Unfreeze `stateful` on `ContainerCodeExecutor`
2. Implement cumulative code history with stdout suppression
3. Add `execution_id`-based history isolation
4. Add `reset_state()` method
5. Add stateful execution tests
6. Update samples and documentation

### Phase 3: Security Hardening (5-7 days)

1. Add `SecurityWarning` to `UnsafeLocalCodeExecutor`
2. Add `restrict_builtins` option
3. Implement `LocalSandboxCodeExecutor`
4. Add default image support to `ContainerCodeExecutor`
5. Add network isolation defaults to `ContainerCodeExecutor`
6. Create official `adk-code-executor` Docker image
7. Update all samples to recommend secure executors
8. Add security-focused tests

### Total estimated effort: 10-15 days

---

## 8. Open Questions

1. **Should `timeout_seconds` be enforced at the base class level?**
   We could add a wrapper in `BaseCodeExecutor.execute_code()` that
   enforces the timeout generically, rather than requiring each subclass
   to implement it. However, this would require the base class to manage
   threading, which may not be appropriate for remote executors.

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

## 9. References

- [BaseCodeExecutor](../../src/google/adk/code_executors/base_code_executor.py)
- [ContainerCodeExecutor](../../src/google/adk/code_executors/container_code_executor.py)
- [UnsafeLocalCodeExecutor](../../src/google/adk/code_executors/unsafe_local_code_executor.py)
- [GkeCodeExecutor](../../src/google/adk/code_executors/gke_code_executor.py)
- [VertexAiCodeExecutor](../../src/google/adk/code_executors/vertex_ai_code_executor.py)
- [AgentEngineSandboxCodeExecutor](../../src/google/adk/code_executors/agent_engine_sandbox_code_executor.py)
- [ExecuteSkillScriptTool](../../src/google/adk/tools/skill_toolset.py)
- [Code Execution Flow](../../src/google/adk/flows/llm_flows/_code_execution.py)
- [PR #4575 — ExecuteSkillScriptTool](https://github.com/google/adk-python/pull/4575)
