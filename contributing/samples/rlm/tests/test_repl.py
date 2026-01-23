"""
Tests for the LocalREPL environment.
"""

from adk_rlm.repl.local_repl import LocalREPL
import pytest


class TestBasicExecution:
  """Tests for basic code execution."""

  def test_basic_execution(self, mock_llm_query):
    """Execute simple Python code."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)
    result = repl.execute_code("x = 42")

    assert result.stderr == ""
    assert "x" in result.locals
    assert result.locals["x"] == 42

  def test_print_capture(self, mock_llm_query):
    """Capture print() output."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)
    result = repl.execute_code("print('Hello, World!')")

    assert "Hello, World!" in result.stdout
    assert result.stderr == ""

  def test_stderr_capture(self, mock_llm_query):
    """Capture exceptions in stderr."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)
    result = repl.execute_code("raise ValueError('test error')")

    assert "ValueError" in result.stderr
    assert "test error" in result.stderr

  def test_variable_persistence(self, mock_llm_query):
    """Variables persist across executions."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)
    repl.execute_code("x = 42")
    result = repl.execute_code("y = x * 2\nprint(y)")

    assert result.stdout.strip() == "84"
    assert repl.locals["x"] == 42
    assert repl.locals["y"] == 84

  def test_multiline_code(self, mock_llm_query):
    """Execute multiline code blocks."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)
    code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

result = factorial(5)
print(result)
"""
    result = repl.execute_code(code)

    assert result.stdout.strip() == "120"
    assert result.stderr == ""

  def test_execution_timing(self, mock_llm_query):
    """Execution time is tracked."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)
    result = repl.execute_code("x = 1 + 1")

    assert result.execution_time > 0
    assert result.execution_time < 1.0  # Should be fast

  def test_syntax_error_handling(self, mock_llm_query):
    """Handle syntax errors gracefully."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)
    result = repl.execute_code("if True print('bad')")

    assert "SyntaxError" in result.stderr

  def test_runtime_error_handling(self, mock_llm_query):
    """Handle runtime errors gracefully."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)
    result = repl.execute_code("x = 1 / 0")

    assert "ZeroDivisionError" in result.stderr


class TestContextLoading:
  """Tests for context loading functionality."""

  def test_context_loading_string(self, mock_llm_query, sample_context):
    """Load string context."""
    repl = LocalREPL(
        llm_query_fn=mock_llm_query, context_payload=sample_context
    )
    result = repl.execute_code("print(type(context).__name__)")

    assert "str" in result.stdout
    assert "context" in repl.locals
    assert repl.locals["context"] == sample_context

  def test_context_loading_dict(self, mock_llm_query, sample_context_dict):
    """Load dict context."""
    repl = LocalREPL(
        llm_query_fn=mock_llm_query, context_payload=sample_context_dict
    )
    result = repl.execute_code("print(context['title'])")

    assert "Test Document" in result.stdout

  def test_context_loading_list(self, mock_llm_query, sample_context_list):
    """Load list context."""
    repl = LocalREPL(
        llm_query_fn=mock_llm_query, context_payload=sample_context_list
    )
    result = repl.execute_code("print(len(context))")

    assert "3" in result.stdout

  def test_multiple_contexts(self, mock_llm_query):
    """Add multiple contexts."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)
    repl.add_context("First context", 0)
    repl.add_context("Second context", 1)

    assert repl.get_context_count() == 2
    assert "context_0" in repl.locals
    assert "context_1" in repl.locals
    assert "context" in repl.locals
    assert repl.locals["context"] == repl.locals["context_0"]

  def test_context_loading_non_serializable_object(self, mock_llm_query):
    """Load non-JSON-serializable object (e.g., custom class)."""

    class CustomObject:

      def __init__(self, value):
        self.value = value

      def get_value(self):
        return self.value

    obj = CustomObject(42)
    repl = LocalREPL(llm_query_fn=mock_llm_query, context_payload=obj)

    # Object should be accessible in the REPL
    assert "context" in repl.locals
    assert repl.locals["context"] is obj

    # Should be able to call methods on it
    result = repl.execute_code("result = context.get_value()\nprint(result)")
    assert "42" in result.stdout

  def test_context_loading_lazy_file_collection(self, mock_llm_query, tmp_path):
    """Load LazyFileCollection (the actual use case that was failing)."""
    from adk_rlm.files import FileLoader

    # Create test files
    (tmp_path / "test1.txt").write_text("Content 1")
    (tmp_path / "test2.txt").write_text("Content 2")

    # Create lazy file collection
    loader = FileLoader(base_path=tmp_path)
    files = loader.create_lazy_files(["*.txt"])

    # Load into REPL - this was failing before the fix
    repl = LocalREPL(llm_query_fn=mock_llm_query, context_payload=files)

    # Collection should be accessible
    assert "context" in repl.locals
    assert repl.locals["context"] is files

    # Should be able to use collection methods
    result = repl.execute_code("print(len(context))")
    assert "2" in result.stdout

    result = repl.execute_code("print(context.names)")
    assert "test1.txt" in result.stdout or "test2.txt" in result.stdout

  def test_context_loading_dict_with_lazy_files(self, mock_llm_query, tmp_path):
    """Load dict containing LazyFileCollection (RLM build_context format)."""
    from adk_rlm.files import FileLoader

    # Create test files
    (tmp_path / "doc.txt").write_text("Document content")

    # Create context dict like build_context does
    loader = FileLoader(base_path=tmp_path)
    files = loader.create_lazy_files(["*.txt"])
    context_dict = {
        "files": files,
        "file_count": len(files),
        "file_names": files.names,
    }

    # Load into REPL
    repl = LocalREPL(llm_query_fn=mock_llm_query, context_payload=context_dict)

    # Context should be accessible
    assert "context" in repl.locals

    # Should be able to access the files
    result = repl.execute_code("print(context['file_count'])")
    assert "1" in result.stdout

    result = repl.execute_code("print(context['files'].names)")
    assert "doc.txt" in result.stdout


class TestSafeBuiltins:
  """Tests for safe builtins."""

  def test_safe_builtins_available(self, mock_llm_query):
    """Allowed builtins work."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)

    # Test various safe builtins
    result = repl.execute_code("print(len([1, 2, 3]))")
    assert "3" in result.stdout

    result = repl.execute_code("print(str(42))")
    assert "42" in result.stdout

    result = repl.execute_code("print(list(range(3)))")
    assert "[0, 1, 2]" in result.stdout

  def test_dangerous_builtins_blocked(self, mock_llm_query):
    """Blocked builtins raise errors."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)

    result = repl.execute_code("eval('1 + 1')")
    assert (
        "Error" in result.stderr
        or "None" in result.stderr
        or "TypeError" in result.stderr
    )

    result = repl.execute_code("exec('x = 1')")
    assert (
        "Error" in result.stderr
        or "None" in result.stderr
        or "TypeError" in result.stderr
    )

  def test_import_allowed_modules(self, mock_llm_query):
    """Can import safe modules."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)

    result = repl.execute_code("import re\nprint(re.match('a', 'abc'))")
    assert result.stderr == "" or "Error" not in result.stderr

    result = repl.execute_code("import json\nprint(json.dumps({'a': 1}))")
    assert result.stderr == "" or "Error" not in result.stderr


class TestLLMQuery:
  """Tests for LLM query functionality."""

  def test_llm_query_basic(self, mock_llm_query):
    """Basic llm_query call works."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)
    result = repl.execute_code(
        "response = llm_query('What is 2+2?')\nprint(response)"
    )

    assert "Mock response" in result.stdout
    assert result.stderr == ""

  def test_llm_query_batched(self, mock_llm_query, mock_llm_query_batched):
    """Batched llm_query works."""
    repl = LocalREPL(
        llm_query_fn=mock_llm_query, llm_query_batched_fn=mock_llm_query_batched
    )
    result = repl.execute_code(
        "responses = llm_query_batched(['Q1?', 'Q2?'])\nprint(len(responses))"
    )

    assert "2" in result.stdout
    assert result.stderr == ""

  def test_llm_query_no_function(self):
    """Error when no llm_query function configured."""
    repl = LocalREPL()
    result = repl.execute_code("response = llm_query('test')\nprint(response)")

    assert "Error" in result.stdout


class TestFinalVar:
  """Tests for FINAL_VAR functionality."""

  def test_final_var_string(self, mock_llm_query):
    """FINAL_VAR resolves string variable."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)
    repl.execute_code("my_answer = 'The answer is 42'")
    result = repl.execute_code("print(FINAL_VAR('my_answer'))")

    assert "The answer is 42" in result.stdout

  def test_final_var_missing(self, mock_llm_query):
    """FINAL_VAR with missing variable returns error."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)
    result = repl.execute_code("print(FINAL_VAR('nonexistent'))")

    assert "Error" in result.stdout


class TestHistoryManagement:
  """Tests for history management."""

  def test_add_history(self, mock_llm_query):
    """Add conversation history."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]
    repl.add_history(history)

    assert repl.get_history_count() == 1
    assert "history_0" in repl.locals
    assert "history" in repl.locals

  def test_multiple_histories(self, mock_llm_query):
    """Add multiple histories."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)
    repl.add_history([{"role": "user", "content": "First"}])
    repl.add_history([{"role": "user", "content": "Second"}])

    assert repl.get_history_count() == 2
    assert "history_0" in repl.locals
    assert "history_1" in repl.locals


class TestCleanup:
  """Tests for cleanup functionality."""

  def test_cleanup(self, mock_llm_query):
    """Cleanup removes temp files and resets state."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)
    repl.execute_code("x = 42")
    repl.cleanup()

    assert len(repl.globals) == 0
    assert len(repl.locals) == 0

  def test_context_manager(self, mock_llm_query):
    """Context manager cleans up on exit."""
    with LocalREPL(llm_query_fn=mock_llm_query) as repl:
      repl.execute_code("x = 42")
      assert "x" in repl.locals

    # After exit, cleanup should have been called
    assert len(repl.locals) == 0

  def test_reset(self, mock_llm_query):
    """Reset clears state but keeps the REPL usable."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)
    repl.execute_code("x = 42")
    repl.add_context("test context")
    repl.reset()

    assert "x" not in repl.locals
    assert repl.get_context_count() == 0
    assert repl.get_history_count() == 0

    # Should still be usable
    result = repl.execute_code("y = 100\nprint(y)")
    assert "100" in result.stdout
