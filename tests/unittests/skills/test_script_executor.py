# Copyright 2025 Google LLC
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

"""Unit tests for ScriptExecutor class."""

from __future__ import annotations

from pathlib import Path
import tempfile

from google.adk.skills import ScriptExecutionResult
from google.adk.skills import ScriptExecutor
import pytest


class TestScriptExecutionResult:
  """Tests for ScriptExecutionResult."""

  def test_default_values(self):
    """Test default values."""
    result = ScriptExecutionResult(success=True)

    assert result.success is True
    assert result.stdout == ""
    assert result.stderr == ""
    assert result.return_code == 0
    assert result.execution_time_ms == 0.0
    assert result.timed_out is False

  def test_custom_values(self):
    """Test custom values."""
    result = ScriptExecutionResult(
        success=False,
        stdout="output",
        stderr="error",
        return_code=1,
        execution_time_ms=100.5,
        timed_out=True,
    )

    assert result.success is False
    assert result.stdout == "output"
    assert result.stderr == "error"
    assert result.return_code == 1
    assert result.execution_time_ms == 100.5
    assert result.timed_out is True


class TestScriptExecutor:
  """Tests for ScriptExecutor."""

  @pytest.fixture
  def executor(self):
    """Create a default executor."""
    return ScriptExecutor()

  @pytest.fixture
  def python_script(self):
    """Create a temporary Python script."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
      f.write("""#!/usr/bin/env python3
import sys
print("Hello from Python!")
print("Args:", sys.argv[1:])
""")
      f.flush()
      script_path = Path(f.name)
    yield script_path
    script_path.unlink(missing_ok=True)

  @pytest.fixture
  def failing_script(self):
    """Create a Python script that fails."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
      f.write("""#!/usr/bin/env python3
import sys
print("Error message", file=sys.stderr)
sys.exit(1)
""")
      f.flush()
      script_path = Path(f.name)
    yield script_path
    script_path.unlink(missing_ok=True)

  @pytest.fixture
  def slow_script(self):
    """Create a slow Python script."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
      f.write("""#!/usr/bin/env python3
import time
time.sleep(10)
print("Done")
""")
      f.flush()
      script_path = Path(f.name)
    yield script_path
    script_path.unlink(missing_ok=True)

  @pytest.fixture
  def shell_script(self):
    """Create a temporary shell script."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
      f.write("""#!/bin/bash
echo "Hello from Bash!"
echo "Args: $@"
""")
      f.flush()
      script_path = Path(f.name)
    yield script_path
    script_path.unlink(missing_ok=True)

  def test_default_config(self, executor):
    """Test default configuration."""
    assert executor.timeout_seconds == 60.0
    assert executor.allow_network is False
    assert executor.memory_limit_mb == 256
    assert executor.use_container is False

  def test_custom_config(self):
    """Test custom configuration."""
    executor = ScriptExecutor(
        timeout_seconds=30.0,
        allow_network=True,
        memory_limit_mb=512,
    )

    assert executor.timeout_seconds == 30.0
    assert executor.allow_network is True
    assert executor.memory_limit_mb == 512

  @pytest.mark.asyncio
  async def test_execute_python_script(self, executor, python_script):
    """Test executing a Python script."""
    result = await executor.execute_script(python_script)

    assert result.success is True
    assert "Hello from Python!" in result.stdout
    assert result.return_code == 0
    assert result.execution_time_ms > 0

  @pytest.mark.asyncio
  async def test_execute_python_script_with_args(self, executor, python_script):
    """Test executing a Python script with arguments."""
    result = await executor.execute_script(
        python_script,
        args=["arg1", "arg2"],
    )

    assert result.success is True
    assert "arg1" in result.stdout
    assert "arg2" in result.stdout

  @pytest.mark.asyncio
  async def test_execute_shell_script(self, executor, shell_script):
    """Test executing a shell script."""
    result = await executor.execute_script(shell_script)

    assert result.success is True
    assert "Hello from Bash!" in result.stdout

  @pytest.mark.asyncio
  async def test_execute_failing_script(self, executor, failing_script):
    """Test executing a failing script."""
    result = await executor.execute_script(failing_script)

    assert result.success is False
    assert result.return_code == 1
    assert "Error message" in result.stderr

  @pytest.mark.asyncio
  async def test_execute_with_timeout(self, slow_script):
    """Test script timeout."""
    executor = ScriptExecutor(timeout_seconds=0.5)

    result = await executor.execute_script(slow_script)

    assert result.success is False
    assert result.timed_out is True
    assert "timed out" in result.stderr.lower()

  @pytest.mark.asyncio
  async def test_execute_nonexistent_script(self, executor):
    """Test executing non-existent script."""
    with pytest.raises(FileNotFoundError):
      await executor.execute_script("/nonexistent/script.py")

  @pytest.mark.asyncio
  async def test_execute_unsupported_script_type(self, executor):
    """Test executing unsupported script type."""
    with tempfile.NamedTemporaryFile(suffix=".unknown") as f:
      with pytest.raises(ValueError, match="Unsupported script type"):
        await executor.execute_script(f.name)

  @pytest.mark.asyncio
  async def test_execute_with_working_dir(self, executor, python_script):
    """Test executing with custom working directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
      result = await executor.execute_script(
          python_script,
          working_dir=tmpdir,
      )

      assert result.success is True

  @pytest.mark.asyncio
  async def test_execute_with_env(self, executor):
    """Test executing with custom environment variables."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
      f.write("""#!/usr/bin/env python3
import os
print(os.environ.get("TEST_VAR", "not found"))
""")
      script_path = Path(f.name)

    result = await executor.execute_script(
        script_path,
        env={"TEST_VAR": "test_value"},
    )

    assert result.success is True
    assert "test_value" in result.stdout

  @pytest.mark.asyncio
  async def test_execute_with_stdin(self, executor):
    """Test executing with stdin input."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
      f.write("""#!/usr/bin/env python3
import sys
data = sys.stdin.read()
print(f"Received: {data}")
""")
      script_path = Path(f.name)

    result = await executor.execute_script(
        script_path,
        stdin="hello world",
    )

    assert result.success is True
    assert "Received: hello world" in result.stdout

  def test_check_interpreter_available(self, executor):
    """Test checking interpreter availability."""
    # Python should be available
    assert executor.check_interpreter_available(".py") is True

    # Unknown extension should not be available
    assert executor.check_interpreter_available(".unknown") is False

  def test_get_available_interpreters(self, executor):
    """Test getting available interpreters."""
    interpreters = executor.get_available_interpreters()

    # Python should be available on most systems
    assert ".py" in interpreters

  @pytest.mark.asyncio
  async def test_output_truncation(self):
    """Test that large output is truncated."""
    executor = ScriptExecutor(max_output_size=100)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
      f.write("""#!/usr/bin/env python3
print("x" * 1000)
""")
      script_path = Path(f.name)

    result = await executor.execute_script(script_path)

    assert result.success is True
    assert len(result.stdout) < 1000
    assert "truncated" in result.stdout.lower()


class TestScriptExecutorValidation:
  """Tests for ScriptExecutor validation."""

  def test_invalid_timeout(self):
    """Test that invalid timeout is rejected."""
    with pytest.raises(ValueError):
      ScriptExecutor(timeout_seconds=0)

    with pytest.raises(ValueError):
      ScriptExecutor(timeout_seconds=-1)

  def test_invalid_memory_limit(self):
    """Test that invalid memory limit is rejected."""
    with pytest.raises(ValueError):
      ScriptExecutor(memory_limit_mb=0)

    with pytest.raises(ValueError):
      ScriptExecutor(memory_limit_mb=-1)

  def test_invalid_max_output_size(self):
    """Test that invalid max output size is rejected."""
    with pytest.raises(ValueError):
      ScriptExecutor(max_output_size=0)
