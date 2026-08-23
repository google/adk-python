# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import signal
import subprocess
import textwrap
import time
from unittest.mock import MagicMock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.code_executors import unsafe_local_code_executor
from google.adk.code_executors.code_execution_utils import CodeExecutionInput
from google.adk.code_executors.code_execution_utils import CodeExecutionResult
from google.adk.code_executors.unsafe_local_code_executor import UnsafeLocalCodeExecutor
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.session import Session
import pytest


def _written_pid(pid_file) -> int | None:
  """Returns the pid the executed code recorded, or None if it has not yet."""
  try:
    recorded = pid_file.read_text().strip()
  except OSError:
    return None
  return int(recorded) if recorded else None


def _is_alive(pid: int) -> bool:
  """Returns whether `pid` is a live (non-zombie) process."""
  try:
    with open(f"/proc/{pid}/stat", encoding="utf-8") as stat_file:
      state = stat_file.read().rsplit(")", 1)[1].split()[0]
  except OSError:
    return False
  return state != "Z"


def _execute_within(
    executor: UnsafeLocalCodeExecutor,
    invocation_context: InvocationContext,
    code: str,
    seconds: float,
) -> CodeExecutionResult:
  """Executes `code` under a wall-clock bound.

  Several callers below cover code that dies without reporting a result, which
  is precisely the case the old executor could wait on forever. The bound is
  the executor's own timeout rather than a watchdog around it: a child that
  dies closes its pipes, so the wait ends on its own, and passing the timeout
  through also exercises the path that enforces it.
  """
  started = time.monotonic()
  result = executor.execute_code(
      invocation_context, CodeExecutionInput(code=code)
  )
  elapsed = time.monotonic() - started
  assert (
      elapsed < seconds * 4
  ), f"the executor took {elapsed:.1f}s against a {seconds}s timeout"
  return result


@pytest.fixture
def mock_invocation_context() -> InvocationContext:
  """Provides a mock InvocationContext."""
  mock_agent = MagicMock(spec=BaseAgent)
  mock_session = MagicMock(spec=Session)
  mock_session_service = MagicMock(spec=BaseSessionService)
  return InvocationContext(
      invocation_id="test_invocation",
      agent=mock_agent,
      session=mock_session,
      session_service=mock_session_service,
  )


class TestUnsafeLocalCodeExecutor:

  def test_init_default(self):
    executor = UnsafeLocalCodeExecutor()
    assert not executor.stateful
    assert not executor.optimize_data_file

  def test_init_stateful_raises_error(self):
    with pytest.raises(
        ValueError,
        match="Cannot set `stateful=True` in UnsafeLocalCodeExecutor.",
    ):
      UnsafeLocalCodeExecutor(stateful=True)

  def test_init_optimize_data_file_raises_error(self):
    with pytest.raises(
        ValueError,
        match=(
            "Cannot set `optimize_data_file=True` in UnsafeLocalCodeExecutor."
        ),
    ):
      UnsafeLocalCodeExecutor(optimize_data_file=True)

  def test_execute_code_simple_print(
      self, mock_invocation_context: InvocationContext
  ):
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code='print("hello world")')
    result = executor.execute_code(mock_invocation_context, code_input)

    assert isinstance(result, CodeExecutionResult)
    assert result.stdout == "hello world\n"
    assert result.stderr == ""
    assert result.output_files == []

  def test_execute_code_with_error(
      self, mock_invocation_context: InvocationContext
  ):
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code='raise ValueError("Test error")')
    result = executor.execute_code(mock_invocation_context, code_input)

    assert isinstance(result, CodeExecutionResult)
    assert result.stdout == ""
    assert "Test error" in result.stderr
    assert result.output_files == []

  def test_execute_code_variable_assignment(
      self, mock_invocation_context: InvocationContext
  ):
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code="x = 10\nprint(x * 2)")
    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stdout == "20\n"
    assert result.stderr == ""

  def test_execute_code_empty(self, mock_invocation_context: InvocationContext):
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code="")
    result = executor.execute_code(mock_invocation_context, code_input)
    assert result.stdout == ""
    assert result.stderr == ""

  def test_execute_code_nested_function_call(
      self, mock_invocation_context: InvocationContext
  ):
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code=(textwrap.dedent("""
                def helper(name):
                  return f'hi {name}'

                def run():
                  print(helper('ada'))

                run()
                """)))

    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stderr == ""
    assert result.stdout == "hi ada\n"

  def test_execute_code_timeout(
      self, mock_invocation_context: InvocationContext
  ):
    executor = UnsafeLocalCodeExecutor(timeout_seconds=1)
    code_input = CodeExecutionInput(code="import time\ntime.sleep(2)")
    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stdout == ""
    assert "Code execution timed out after 1 seconds." in result.stderr

  def test_execute_code_main_guard_runs(
      self, mock_invocation_context: InvocationContext
  ):
    """Code guarded on `__main__` runs, as it did in the previous child."""
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code=textwrap.dedent("""
        if __name__ == '__main__':
          print('guarded')
        """))

    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stdout == "guarded\n"
    assert result.stderr == ""

  def test_execute_code_without_main_guard_is_not_main(
      self, mock_invocation_context: InvocationContext
  ):
    """Code that does not ask to be `__main__` is not given the name."""
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code="print(globals().get('__name__'))")

    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stdout == "None\n"

  def test_execute_code_separates_stdout_from_stderr(
      self, mock_invocation_context: InvocationContext
  ):
    """Each stream lands in its own field rather than being interleaved."""
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code=textwrap.dedent("""
        import sys
        sys.stdout.write('to out')
        sys.stderr.write('to err')
        sys.exit(0)
        """))

    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stdout == "to out"
    # The program wrote to stderr but succeeded. A non-empty stderr is what
    # marks a result failed and drives the retry counter, so a warning must not
    # be reported as a failure.
    assert result.stderr == ""

  def test_execute_code_nonzero_exit_is_reported_as_a_failure(
      self, mock_invocation_context: InvocationContext
  ):
    """A program that exits non-zero fails even when it says nothing."""
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code="import sys\nsys.exit(3)")

    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stdout == ""
    assert result.stderr == "Code execution exited with status 3."

  def test_execute_code_reports_a_crash_rather_than_hanging(
      self, mock_invocation_context: InvocationContext
  ):
    """Code that exits without reporting anything still returns a result."""
    executor = UnsafeLocalCodeExecutor()

    result = _execute_within(
        executor,
        mock_invocation_context,
        "import os\nprint('before', flush=True)\nos._exit(3)",
        seconds=30,
    )

    assert result.stdout == "before\n"
    assert result.stderr == "Code execution exited with status 3."

  @pytest.mark.skipif(
      not hasattr(signal, "SIGKILL"),
      reason="Death by signal is checked on POSIX only.",
  )
  def test_execute_code_reports_death_by_signal(
      self, mock_invocation_context: InvocationContext
  ):
    """Code killed outright -- by a segfault or the OOM killer -- returns."""
    executor = UnsafeLocalCodeExecutor()

    result = _execute_within(
        executor,
        mock_invocation_context,
        "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)",
        seconds=30,
    )

    assert result.stdout == ""
    assert result.stderr == (
        f"Code execution exited with status {-signal.SIGKILL}."
    )

  def test_execute_code_traceback_omits_this_module(
      self, mock_invocation_context: InvocationContext
  ):
    """A failure shows the model its own code, not a frame from this package."""
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(
        code="def divide():\n  return 1 / 0\n\ndivide()"
    )

    result = executor.execute_code(mock_invocation_context, code_input)

    assert "ZeroDivisionError" in result.stderr
    assert "unsafe_local_code_executor" not in result.stderr
    # Both frames of the executed code survive; only the wrapper's is dropped.
    assert result.stderr.count('File "<code>"') == 2

  def test_execute_code_preserves_unicode(
      self, mock_invocation_context: InvocationContext
  ):
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code="print('你好, café')")

    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stdout == "你好, café\n"
    assert result.stderr == ""

  def test_execute_code_output_encoding_does_not_follow_the_host_locale(
      self, mock_invocation_context: InvocationContext
  ):
    """Otherwise a program printing non-ASCII dies encoding its own output."""
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(
        code="import sys\nprint(sys.stdout.encoding)"
    )

    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stdout.strip().lower() == "utf-8"

  def test_execute_code_large_output(
      self, mock_invocation_context: InvocationContext
  ):
    """Output far larger than a pipe buffer is read out rather than deadlocked."""
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code="print('x' * 1000000)")

    result = _execute_within(
        executor, mock_invocation_context, code_input.code, seconds=60
    )

    assert result.stdout == "x" * 1000000 + "\n"
    assert result.stderr == ""

  def test_execute_code_large_program(
      self, mock_invocation_context: InvocationContext
  ):
    """A program carrying its own data is not capped by an argument limit."""
    executor = UnsafeLocalCodeExecutor()
    payload = "a" * 300000
    code_input = CodeExecutionInput(
        code=f"data = {payload!r}\nprint(len(data))"
    )

    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stdout == f"{len(payload)}\n"
    assert result.stderr == ""

  def test_execute_code_imports_resolve_from_the_agent_path(
      self, mock_invocation_context: InvocationContext
  ):
    """Code importing what the application can import still resolves."""
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code="import google.adk\nprint('imported')")

    result = _execute_within(
        executor, mock_invocation_context, code_input.code, seconds=120
    )

    assert result.stdout == "imported\n"
    assert result.stderr == ""

  def test_kill_execution_signals_group_before_killing_it(self, monkeypatch):
    """The group gets SIGTERM and its grace period before SIGKILL."""
    signalled = []
    monkeypatch.setattr(
        unsafe_local_code_executor.os,
        "killpg",
        lambda group, sig: signalled.append((group, sig)),
        raising=False,
    )
    process = MagicMock()
    process.pid = 4321
    process.terminate.side_effect = lambda: signalled.append(("child", "term"))
    process.kill.side_effect = lambda: signalled.append(("child", "kill"))
    process.communicate.return_value = ("out", "err")

    assert unsafe_local_code_executor._kill_execution(process) == ("out", "err")

    assert signalled == [
        (4321, signal.SIGTERM),
        ("child", "term"),
        (4321, signal.SIGKILL),
        ("child", "kill"),
    ]
    process.wait.assert_called_once_with(
        unsafe_local_code_executor._TERMINATE_GRACE_SECONDS
    )

  def test_kill_execution_gives_up_on_pipes_that_never_close(self, monkeypatch):
    """A pipe held open by something unkillable does not block the agent."""
    monkeypatch.setattr(
        unsafe_local_code_executor.os,
        "killpg",
        lambda group, sig: None,
        raising=False,
    )
    process = MagicMock()
    process.pid = 4321
    process.wait.side_effect = subprocess.TimeoutExpired("cmd", 5)
    process.communicate.side_effect = subprocess.TimeoutExpired("cmd", 5)

    assert unsafe_local_code_executor._kill_execution(process) == ("", "")

  @pytest.mark.skipif(
      not hasattr(os, "killpg")
      or not hasattr(os, "fork")
      or not os.path.isdir("/proc"),
      reason="Process-group teardown is checked on POSIX with /proc only.",
  )
  def test_timeout_kills_what_the_code_spawned(
      self, mock_invocation_context: InvocationContext, tmp_path
  ):
    """A timed-out execution takes the processes it spawned with it."""
    pid_file = tmp_path / "spawned.pid"
    # Forked rather than spawned through `sys.executable`, so the descendant
    # exists within milliseconds and the test never waits on interpreter
    # start-up.
    code = textwrap.dedent(f"""
        import os
        import time

        spawned = os.fork()
        if spawned == 0:
          time.sleep(60)
          os._exit(0)
        with open({str(pid_file)!r}, 'w') as f:
          f.write(str(spawned))
        time.sleep(60)
        """)
    executor = UnsafeLocalCodeExecutor(timeout_seconds=5)

    spawned_pid = None
    try:
      result = _execute_within(
          executor, mock_invocation_context, code, seconds=60
      )
      assert "Code execution timed out after 5 seconds." in result.stderr

      spawned_pid = _written_pid(pid_file)
      if spawned_pid is None:
        pytest.skip("this environment could not start the execution process")
      deadline = time.time() + 10
      while time.time() < deadline and _is_alive(spawned_pid):
        time.sleep(0.05)
      assert not _is_alive(spawned_pid)
    finally:
      if spawned_pid is not None:
        try:
          os.kill(spawned_pid, signal.SIGKILL)
        except OSError:
          pass
