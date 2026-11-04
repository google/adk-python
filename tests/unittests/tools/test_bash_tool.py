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

import asyncio
import gc
import logging
import os
import pathlib
import signal
import sys
import time
from unittest import mock

import pytest

if sys.platform == "win32":
  pytest.skip(
      "bash tool tests require Unix resource module", allow_module_level=True
  )

import resource

from google.adk.tools import bash_tool
from google.adk.tools import tool_context
from google.adk.tools.tool_confirmation import ToolConfirmation


@pytest.fixture
def workspace(tmp_path):
  """Creates a workspace mirroring the anthropics/skills PDF skill layout."""
  # Skill: pdf/
  skill_dir = tmp_path / "pdf"
  skill_dir.mkdir()
  (skill_dir / "SKILL.md").write_text(
      "---\nname: pdf\n"
      "description: Use this skill whenever the user wants to do"
      " anything with PDF files.\n"
      "---\n# PDF Processing Guide\n\n## Overview\n"
      "This guide covers PDF processing operations."
  )
  scripts = skill_dir / "scripts"
  scripts.mkdir()
  (scripts / "extract_form_structure.py").write_text(
      "import sys; print(f'extracting from {sys.argv[1]}')"
  )
  (scripts / "fill_pdf_form_with_annotations.py").write_text(
      "print('filling form')"
  )
  references = skill_dir / "references"
  references.mkdir()
  (references / "REFERENCE.md").write_text("# Reference\nDetailed docs.")
  # A loose file at workspace root (not inside a skill).
  (tmp_path / "sample.pdf").write_bytes(b"%PDF-1.4 fake")
  return tmp_path


@pytest.fixture
def tool_context_no_confirmation():
  """ToolContext with no confirmation (initial call)."""
  ctx = mock.create_autospec(tool_context.ToolContext, instance=True)
  ctx.tool_confirmation = None
  ctx.actions = mock.MagicMock()
  return ctx


@pytest.fixture
def tool_context_confirmed():
  """ToolContext with confirmation approved."""
  ctx = mock.create_autospec(tool_context.ToolContext, instance=True)
  confirmation = mock.create_autospec(ToolConfirmation, instance=True)
  confirmation.confirmed = True
  ctx.tool_confirmation = confirmation
  ctx.actions = mock.MagicMock()
  return ctx


@pytest.fixture
def tool_context_rejected():
  """ToolContext with confirmation rejected."""
  ctx = mock.create_autospec(tool_context.ToolContext, instance=True)
  confirmation = mock.create_autospec(ToolConfirmation, instance=True)
  confirmation.confirmed = False
  ctx.tool_confirmation = confirmation
  ctx.actions = mock.MagicMock()
  return ctx


def _pid_is_alive(pid: int) -> bool:
  """Returns whether pid still exists."""
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return False
  return True


async def _wait_for_pid_exit(pid: int, timeout: float = 2.0) -> bool:
  """Returns True once pid is gone, or False if it outlives timeout."""
  deadline = time.monotonic() + timeout
  while _pid_is_alive(pid):
    if time.monotonic() >= deadline:
      return False
    await asyncio.sleep(0.05)
  return True


def _kill_pidfile(path: pathlib.Path) -> None:
  """SIGKILLs the pid recorded in path, if that process is still running."""
  if not path.is_file():
    return
  text = path.read_text().strip()
  if not text:
    return
  try:
    pid = int(text)
  except ValueError:
    return
  try:
    os.kill(pid, signal.SIGKILL)
  except ProcessLookupError:
    return


# A setsid grandchild keeps stdout open after its parent leaves the group.
_SETSID_PIPE_HOLDER = (
    "import os\n"
    "import sys\n"
    "import time\n"
    "if os.fork() == 0:\n"
    "    os.setsid()\n"
    "    with open('writer.pid', 'w') as handle:\n"
    "        handle.write(str(os.getpid()))\n"
    "    while True:\n"
    '        sys.stdout.buffer.write(b"z" * 1024)\n'
    "        sys.stdout.buffer.flush()\n"
    "        time.sleep(0.05)\n"
    "else:\n"
    "    with open('parent.pid', 'w') as handle:\n"
    "        handle.write(str(os.getpid()))\n"
    "    time.sleep(60)\n"
)


# --- _validate_command tests ---


class TestValidateCommand:

  def test_empty_command(self):
    policy = bash_tool.BashToolPolicy()
    assert bash_tool._validate_command("", policy) is not None
    assert bash_tool._validate_command("   ", policy) is not None

  def test_default_policy_allows_everything(self):
    policy = bash_tool.BashToolPolicy()
    assert bash_tool._validate_command("rm -rf /", policy) is None
    assert bash_tool._validate_command("cat /etc/passwd", policy) is None
    assert bash_tool._validate_command("sudo curl", policy) is None
    assert bash_tool._validate_command("echo hello | grep h", policy) is None
    assert bash_tool._validate_command("ls ; rm -rf /", policy) is None

  def test_empty_allowlist_blocks_everything(self, workspace):
    policy = bash_tool.BashToolPolicy(allowed_command_prefixes=())
    assert bash_tool._validate_command("ls", policy) is not None
    assert "<none>" in bash_tool._validate_command("ls", policy)
    tool = bash_tool.ExecuteBashTool(workspace=workspace, policy=policy)
    assert "Allowed: commands matching prefixes: <none>." in tool.description

  def test_restricted_policy_allows_prefixes(self):
    policy = bash_tool.BashToolPolicy(allowed_command_prefixes=("ls", "cat"))
    assert bash_tool._validate_command("ls -la", policy) is None
    assert bash_tool._validate_command("cat file.txt", policy) is None

  def test_restricted_policy_blocks_prefix_bypasses(self):
    policy = bash_tool.BashToolPolicy(allowed_command_prefixes=("ls", "cat"))
    assert bash_tool._validate_command("lsof -i", policy) is not None
    assert bash_tool._validate_command("lscpu", policy) is not None
    assert bash_tool._validate_command("catfish", policy) is not None

  def test_restricted_policy_blocks_planted_executables(self):
    policy = bash_tool.BashToolPolicy(allowed_command_prefixes=("ls",))
    assert bash_tool._validate_command("./ls", policy) is not None
    assert bash_tool._validate_command("/tmp/payload/ls", policy) is not None
    assert bash_tool._validate_command("/bin/ls", policy) is not None

  def test_restricted_policy_allows_path_entries(self):
    policy = bash_tool.BashToolPolicy(
        allowed_command_prefixes=("/bin/ls", "/usr/bin/unbuffer /bin/cat")
    )
    assert bash_tool._validate_command("/bin/ls -la", policy) is None
    assert (
        bash_tool._validate_command("/usr/bin/unbuffer /bin/cat f", policy)
        is None
    )
    assert bash_tool._validate_command("/bin/lsof", policy) is not None
    assert bash_tool._validate_command("ls", policy) is not None
    assert (
        bash_tool._validate_command("/usr/bin/unbuffer sh", policy) is not None
    )

  def test_restricted_policy_ignores_entry_padding(self):
    policy = bash_tool.BashToolPolicy(allowed_command_prefixes=("ls ", "cat "))
    assert bash_tool._validate_command("ls", policy) is None
    assert bash_tool._validate_command("cat file.txt", policy) is None
    assert bash_tool._validate_command("lsof", policy) is not None

  def test_restricted_policy_supports_multi_token_entries(self):
    policy = bash_tool.BashToolPolicy(allowed_command_prefixes=("git status",))
    assert bash_tool._validate_command("git status --short", policy) is None
    assert bash_tool._validate_command("git statuses", policy) is not None
    assert bash_tool._validate_command("git push", policy) is not None

  def test_empty_prefix_entry_is_ignored(self):
    policy = bash_tool.BashToolPolicy(allowed_command_prefixes=("", "   "))
    assert bash_tool._validate_command("ls", policy) is not None

  def test_unparsable_prefix_entry_is_ignored_with_warning(self, caplog):
    with caplog.at_level(logging.WARNING, logger="google_adk"):
      policy = bash_tool.BashToolPolicy(
          allowed_command_prefixes=("git commit -m 'unterminated", "ls")
      )
    assert "Ignoring unparsable allowed_command_prefixes entry" in caplog.text
    assert bash_tool._validate_command("git status", policy) is not None
    assert bash_tool._validate_command("ls -la", policy) is None

  def test_unparsable_command_is_blocked(self):
    policy = bash_tool.BashToolPolicy(allowed_command_prefixes=("ls",))
    assert (
        bash_tool._validate_command("ls 'unterminated", policy)
        == "Unable to parse command."
    )

  def test_restricted_policy_blocks_others(self):
    policy = bash_tool.BashToolPolicy(allowed_command_prefixes=("ls", "cat"))
    assert bash_tool._validate_command("rm -rf .", policy) is not None
    assert bash_tool._validate_command("tree", policy) is not None
    assert "Permitted prefixes are: ls, cat" in bash_tool._validate_command(
        "tree", policy
    )

  def test_blocked_operators_validation(self):
    policy = bash_tool.BashToolPolicy(
        allowed_command_prefixes=("*",),
        blocked_operators=("|", ";", "$(", "`", "&&", "||"),
    )
    assert (
        bash_tool._validate_command("echo hello | grep h", policy)
        == "Command contains blocked operator: |"
    )
    assert (
        bash_tool._validate_command("ls ; rm -rf /", policy)
        == "Command contains blocked operator: ;"
    )


class TestExecuteBashTool:

  @pytest.mark.asyncio
  async def test_requests_confirmation(
      self, workspace, tool_context_no_confirmation
  ):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    result = await tool.run_async(
        args={"command": "ls"},
        tool_context=tool_context_no_confirmation,
    )
    assert "error" in result
    assert "requires confirmation" in result["error"]
    tool_context_no_confirmation.request_confirmation.assert_called_once()

  @pytest.mark.asyncio
  async def test_rejected(self, workspace, tool_context_rejected):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    result = await tool.run_async(
        args={"command": "ls"}, tool_context=tool_context_rejected
    )
    assert result == {"error": "This tool call is rejected."}

  @pytest.mark.asyncio
  async def test_executes_when_confirmed(
      self, workspace, tool_context_confirmed
  ):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    result = await tool.run_async(
        args={"command": "ls"},
        tool_context=tool_context_confirmed,
    )
    assert result["returncode"] == 0
    assert "pdf" in result["stdout"]

  @pytest.mark.asyncio
  async def test_cat_skill_md(self, workspace, tool_context_confirmed):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    result = await tool.run_async(
        args={"command": "cat pdf/SKILL.md"},
        tool_context=tool_context_confirmed,
    )
    assert "PDF Processing Guide" in result["stdout"]

  @pytest.mark.asyncio
  async def test_python_script(self, workspace, tool_context_confirmed):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    result = await tool.run_async(
        args={
            "command": "python3 pdf/scripts/extract_form_structure.py test.pdf"
        },
        tool_context=tool_context_confirmed,
    )
    assert "extracting from test.pdf" in result["stdout"]
    assert result["returncode"] == 0

  @pytest.mark.asyncio
  async def test_blocks_disallowed_by_policy(
      self, workspace, tool_context_no_confirmation
  ):
    policy = bash_tool.BashToolPolicy(allowed_command_prefixes=("ls",))
    tool = bash_tool.ExecuteBashTool(workspace=workspace, policy=policy)
    result = await tool.run_async(
        args={"command": "rm -rf ."},
        tool_context=tool_context_no_confirmation,
    )
    assert "error" in result
    assert "Permitted prefixes are: ls" in result["error"]
    tool_context_no_confirmation.request_confirmation.assert_not_called()

  @pytest.mark.asyncio
  async def test_captures_stderr(self, workspace, tool_context_confirmed):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    result = await tool.run_async(
        args={"command": "python3 -c 'import sys; sys.stderr.write(\"err\")'"},
        tool_context=tool_context_confirmed,
    )
    assert "err" in result["stderr"]

  @pytest.mark.asyncio
  async def test_nonzero_returncode(self, workspace, tool_context_confirmed):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    result = await tool.run_async(
        args={"command": "python3 -c 'exit(42)'"},
        tool_context=tool_context_confirmed,
    )
    assert result["returncode"] == 42

  @pytest.mark.asyncio
  async def test_timeout(self, workspace, tool_context_confirmed):
    tool = bash_tool.ExecuteBashTool(
        workspace=workspace,
        policy=bash_tool.BashToolPolicy(timeout_seconds=0),
    )
    exit_fut = asyncio.get_running_loop().create_future()
    mock_process = mock.AsyncMock()
    mock_process.pid = 12345
    mock_process.stdout = None
    mock_process.stderr = None
    mock_process.returncode = None

    async def _wait():
      if mock_process.returncode is not None:
        return mock_process.returncode
      return await exit_fut

    def _killpg(pid, sig):
      mock_process.returncode = -sig
      if not exit_fut.done():
        exit_fut.set_result(-sig)

    mock_process.wait = _wait
    mock_process._transport = mock.Mock()
    with (
        mock.patch.object(
            asyncio,
            "create_subprocess_exec",
            autospec=True,
            return_value=mock_process,
        ),
        mock.patch("os.killpg", side_effect=_killpg) as mock_killpg,
    ):
      result = await tool.run_async(
          args={"command": "python scripts/do_thing.py"},
          tool_context=tool_context_confirmed,
      )
      mock_killpg.assert_called_with(12345, signal.SIGKILL)
    assert "error" in result
    assert "timed out" in result["error"].lower()
    mock_process._transport.close.assert_called_once()

  @pytest.mark.asyncio
  @pytest.mark.parametrize(
      "close_error",
      [AttributeError("missing"), OSError(9, "Bad file descriptor")],
  )
  async def test_timeout_ignores_transport_close_error(
      self, workspace, tool_context_confirmed, close_error
  ):
    """A transport that is already closed still returns the timeout error."""
    tool = bash_tool.ExecuteBashTool(
        workspace=workspace,
        policy=bash_tool.BashToolPolicy(timeout_seconds=0),
    )
    exit_fut = asyncio.get_running_loop().create_future()
    mock_process = mock.AsyncMock()
    mock_process.pid = 12345
    mock_process.stdout = None
    mock_process.stderr = None
    mock_process.returncode = None

    async def _wait():
      if mock_process.returncode is not None:
        return mock_process.returncode
      return await exit_fut

    def _killpg(pid, sig):
      mock_process.returncode = -sig
      if not exit_fut.done():
        exit_fut.set_result(-sig)

    mock_process.wait = _wait
    mock_process._transport = mock.Mock()
    mock_process._transport.close.side_effect = close_error
    with (
        mock.patch.object(
            asyncio,
            "create_subprocess_exec",
            autospec=True,
            return_value=mock_process,
        ),
        mock.patch("os.killpg", side_effect=_killpg),
    ):
      result = await tool.run_async(
          args={"command": "python scripts/do_thing.py"},
          tool_context=tool_context_confirmed,
      )
    assert "timed out" in result["error"].lower()
    mock_process._transport.close.assert_called_once()

  @pytest.mark.asyncio
  async def test_cwd_is_workspace(self, workspace, tool_context_confirmed):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    result = await tool.run_async(
        args={"command": "python3 -c 'import os; print(os.getcwd())'"},
        tool_context=tool_context_confirmed,
    )
    assert result["stdout"].strip() == str(workspace)

  @pytest.mark.asyncio
  async def test_no_command(self, workspace, tool_context_confirmed):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    result = await tool.run_async(args={}, tool_context=tool_context_confirmed)
    assert "error" in result
    assert "required" in result["error"].lower()

  @pytest.mark.asyncio
  async def test_resource_limits_set(self, workspace, tool_context_confirmed):
    policy = bash_tool.BashToolPolicy(
        max_memory_bytes=100 * 1024 * 1024,
        max_file_size_bytes=50 * 1024 * 1024,
        max_child_processes=10,
    )
    tool = bash_tool.ExecuteBashTool(workspace=workspace, policy=policy)
    mock_process = mock.AsyncMock()
    mock_process.pid = None  # Ensure finally block doesn't try to kill it
    mock_process.stdout = None
    mock_process.stderr = None
    mock_exec = mock.AsyncMock(return_value=mock_process)

    with mock.patch("asyncio.create_subprocess_exec", mock_exec):
      await tool.run_async(
          args={"command": "ls"},
          tool_context=tool_context_confirmed,
      )
      assert "preexec_fn" in mock_exec.call_args.kwargs
      preexec_fn = mock_exec.call_args.kwargs["preexec_fn"]

      mock_setrlimit = mock.create_autospec(resource.setrlimit, instance=True)
      with mock.patch("resource.setrlimit", mock_setrlimit):
        preexec_fn()
        mock_setrlimit.assert_any_call(resource.RLIMIT_CORE, (0, 0))
        mock_setrlimit.assert_any_call(
            resource.RLIMIT_AS, (100 * 1024 * 1024, 100 * 1024 * 1024)
        )
        mock_setrlimit.assert_any_call(
            resource.RLIMIT_FSIZE, (50 * 1024 * 1024, 50 * 1024 * 1024)
        )


class TestOutputLimits:

  def test_default_limit(self):
    assert bash_tool.BashToolPolicy().max_output_bytes_per_stream == (
        1024 * 1024
    )

  @pytest.mark.asyncio
  async def test_small_output_unchanged(
      self, workspace, tool_context_confirmed
  ):
    policy = bash_tool.BashToolPolicy(max_output_bytes_per_stream=1024)
    tool = bash_tool.ExecuteBashTool(workspace=workspace, policy=policy)
    result = await tool.run_async(
        args={"command": "python3 -c 'print(\"hello\")'"},
        tool_context=tool_context_confirmed,
    )
    assert result["stdout"] == "hello\n"
    assert result["returncode"] == 0

  @pytest.mark.asyncio
  async def test_large_stdout_is_truncated(
      self, workspace, tool_context_confirmed
  ):
    policy = bash_tool.BashToolPolicy(max_output_bytes_per_stream=1000)
    tool = bash_tool.ExecuteBashTool(workspace=workspace, policy=policy)
    total = 2 * 1024 * 1024
    result = await tool.run_async(
        args={
            "command": (
                "python3 -c 'import sys;"
                f' sys.stdout.buffer.write(b"x" * {total})\''
            )
        },
        tool_context=tool_context_confirmed,
    )
    assert result["returncode"] == 0
    assert result["stdout"].startswith("x" * 1000 + "\n...[stdout truncated")
    assert f"{total - 1000} of {total} bytes omitted" in result["stdout"]
    assert len(result["stdout"]) < 1200

  @pytest.mark.asyncio
  async def test_stdout_and_stderr_are_bounded_independently(
      self, workspace, tool_context_confirmed
  ):
    policy = bash_tool.BashToolPolicy(max_output_bytes_per_stream=500)
    tool = bash_tool.ExecuteBashTool(workspace=workspace, policy=policy)
    script = (
        "import sys\n"
        "for _ in range(64):\n"
        '    sys.stdout.buffer.write(b"o" * 16384)\n'
        '    sys.stderr.buffer.write(b"e" * 16384)\n'
    )
    (workspace / "noisy.py").write_text(script)
    result = await tool.run_async(
        args={"command": "python3 noisy.py"},
        tool_context=tool_context_confirmed,
    )
    assert result["returncode"] == 0
    assert result["stdout"].startswith("o" * 500 + "\n...[stdout truncated")
    assert result["stderr"].startswith("e" * 500 + "\n...[stderr truncated")
    assert len(result["stdout"]) < 700
    assert len(result["stderr"]) < 700

  @pytest.mark.asyncio
  async def test_limit_can_be_disabled(self, workspace, tool_context_confirmed):
    policy = bash_tool.BashToolPolicy(max_output_bytes_per_stream=None)
    tool = bash_tool.ExecuteBashTool(workspace=workspace, policy=policy)
    result = await tool.run_async(
        args={
            "command": (
                'python3 -c \'import sys; sys.stdout.buffer.write(b"x" *'
                " 200000)'"
            )
        },
        tool_context=tool_context_confirmed,
    )
    assert result["stdout"] == "x" * 200000

  @pytest.mark.asyncio
  async def test_timeout_output_is_bounded_and_process_reaped(
      self, workspace, tool_context_confirmed
  ):
    policy = bash_tool.BashToolPolicy(
        timeout_seconds=1, max_output_bytes_per_stream=100
    )
    tool = bash_tool.ExecuteBashTool(workspace=workspace, policy=policy)
    (workspace / "forever.py").write_text(
        'import sys\nwhile True:\n    sys.stdout.buffer.write(b"y" * 65536)\n'
    )
    result = await tool.run_async(
        args={"command": "python3 forever.py"},
        tool_context=tool_context_confirmed,
    )
    assert "timed out" in result["error"]
    assert result["stdout"].startswith("y" * 100 + "\n...[stdout truncated")
    assert len(result["stdout"]) < 300
    assert result["returncode"] == -signal.SIGKILL

  @pytest.mark.asyncio
  async def test_timeout_setsid_child_holding_pipe_cleans_up(
      self, workspace, tool_context_confirmed
  ):
    """Timeout closes the pipe so a setsid descendant stops writing.

    A descendant that calls setsid() leaves the process group, so killpg does
    not close its inherited stdout. Cleanup cancels the explicit waiter and
    readers, then closes the subprocess transport. The descendant's next write
    fails and the writer exits instead of outliving run_async.
    """
    policy = bash_tool.BashToolPolicy(
        timeout_seconds=1, max_output_bytes_per_stream=64
    )
    tool = bash_tool.ExecuteBashTool(workspace=workspace, policy=policy)
    (workspace / "setsid_hold.py").write_text(_SETSID_PIPE_HOLDER)
    loop = asyncio.get_running_loop()
    reports = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _, context: reports.append(context))
    try:
      started = time.monotonic()
      result = await tool.run_async(
          args={"command": "python3 setsid_hold.py"},
          tool_context=tool_context_confirmed,
      )
      elapsed = time.monotonic() - started
      for _ in range(5):
        await asyncio.sleep(0)
      gc.collect()
      await asyncio.sleep(0)
      assert "timed out" in result["error"]
      # Returns after timeout plus drain grace, not the child's 60s sleep.
      assert elapsed < 5.0
      leftover = [
          t
          for t in asyncio.all_tasks()
          if not t.done()
          and t is not asyncio.current_task()
          and getattr(getattr(t, "get_coro", lambda: None)(), "__name__", None)
          in {"_drain_stream", "wait"}
      ]
      assert leftover == []
      noisy = [
          r
          for r in reports
          if "never retrieved" in r.get("message", "")
          or "shielded" in r.get("message", "").lower()
      ]
      assert noisy == []
      writer_pid = int((workspace / "writer.pid").read_text().strip())
      assert await _wait_for_pid_exit(writer_pid)
    finally:
      loop.set_exception_handler(previous_handler)
      _kill_pidfile(workspace / "writer.pid")
      _kill_pidfile(workspace / "parent.pid")

  def test_rejects_negative_output_limit(self):
    with pytest.raises(ValueError, match="max_output_bytes_per_stream"):
      bash_tool.BashToolPolicy(max_output_bytes_per_stream=-1)

  def test_zero_output_limit_is_allowed(self):
    policy = bash_tool.BashToolPolicy(max_output_bytes_per_stream=0)
    assert policy.max_output_bytes_per_stream == 0

  @pytest.mark.asyncio
  async def test_output_exactly_at_limit_is_not_truncated(
      self, workspace, tool_context_confirmed
  ):
    limit = 1000
    policy = bash_tool.BashToolPolicy(max_output_bytes_per_stream=limit)
    tool = bash_tool.ExecuteBashTool(workspace=workspace, policy=policy)
    result = await tool.run_async(
        args={
            "command": (
                "python3 -c 'import sys;"
                f' sys.stdout.buffer.write(b"x" * {limit})\''
            )
        },
        tool_context=tool_context_confirmed,
    )
    assert result["returncode"] == 0
    assert result["stdout"] == "x" * limit
    assert "truncated" not in result["stdout"]

  @pytest.mark.asyncio
  async def test_cancellation_cleans_up_drain_and_wait(
      self, workspace, tool_context_confirmed
  ):
    """Cancelling run_async must not leave drain/wait tasks pending.

    A child that stays in the process group and keeps stdout open would
    otherwise leave _drain_stream tasks hanging after CancelledError.
    """
    policy = bash_tool.BashToolPolicy(
        timeout_seconds=30, max_output_bytes_per_stream=64
    )
    tool = bash_tool.ExecuteBashTool(workspace=workspace, policy=policy)
    (workspace / "hold_pipe.py").write_text(
        "import os\n"
        "import sys\n"
        "import time\n"
        "if os.fork() == 0:\n"
        "    while True:\n"
        '        sys.stdout.buffer.write(b"z" * 1024)\n'
        "        sys.stdout.buffer.flush()\n"
        "        time.sleep(0.05)\n"
        "else:\n"
        "    time.sleep(60)\n"
    )
    task = asyncio.create_task(
        tool.run_async(
            args={"command": "python3 hold_pipe.py"},
            tool_context=tool_context_confirmed,
        )
    )
    await asyncio.sleep(0.4)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
      await task
    await asyncio.sleep(0)
    leftover_drains = [
        t
        for t in asyncio.all_tasks()
        if not t.done()
        and getattr(getattr(t, "get_coro", lambda: None)(), "__name__", None)
        == "_drain_stream"
    ]
    assert leftover_drains == []

  @pytest.mark.asyncio
  async def test_cancellation_setsid_child_holding_pipe_exits(
      self, workspace, tool_context_confirmed
  ):
    """Cancelling run_async closes the pipe so a setsid writer exits.

    The forked child leaves the process group, so killing that group does not
    stop it. Transport close after the cancelled readers are gathered must
    make the writer exit.
    """
    policy = bash_tool.BashToolPolicy(
        timeout_seconds=30, max_output_bytes_per_stream=64
    )
    tool = bash_tool.ExecuteBashTool(workspace=workspace, policy=policy)
    (workspace / "setsid_hold.py").write_text(_SETSID_PIPE_HOLDER)
    task = asyncio.create_task(
        tool.run_async(
            args={"command": "python3 setsid_hold.py"},
            tool_context=tool_context_confirmed,
        )
    )
    pid_path = workspace / "writer.pid"
    try:
      deadline = time.monotonic() + 2
      while not pid_path.is_file():
        if task.done() or time.monotonic() >= deadline:
          break
        await asyncio.sleep(0.05)
      assert pid_path.is_file()
      task.cancel()
      with pytest.raises(asyncio.CancelledError):
        await task
      writer_pid = int(pid_path.read_text().strip())
      assert await _wait_for_pid_exit(writer_pid)
    finally:
      if not task.done():
        task.cancel()
      _kill_pidfile(pid_path)
      _kill_pidfile(workspace / "parent.pid")

  @pytest.mark.asyncio
  async def test_cancellation_retrieves_completion_exception(
      self, workspace, tool_context_confirmed
  ):
    """Cancelling run_async must not report an unretrieved gather exception."""
    loop = asyncio.get_running_loop()
    reports = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _, context: reports.append(context))
    try:
      tool = bash_tool.ExecuteBashTool(
          workspace=workspace,
          policy=bash_tool.BashToolPolicy(timeout_seconds=30),
      )
      task = asyncio.create_task(
          tool.run_async(
              args={"command": "sleep 30"},
              tool_context=tool_context_confirmed,
          )
      )
      await asyncio.sleep(0.4)
      task.cancel()
      with pytest.raises(asyncio.CancelledError):
        await task
      del task
      for _ in range(3):
        await asyncio.sleep(0)
      gc.collect()
      await asyncio.sleep(0)
    finally:
      loop.set_exception_handler(previous_handler)
    assert [
        r
        for r in reports
        if "never retrieved" in r.get("message", "")
        or "shielded" in r.get("message", "").lower()
    ] == []
