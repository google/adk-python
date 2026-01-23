"""
Tests for code block parsing and final answer detection.
"""

from adk_rlm.callbacks.code_execution import find_code_blocks
from adk_rlm.callbacks.code_execution import find_final_answer
from adk_rlm.callbacks.code_execution import format_execution_result
from adk_rlm.callbacks.code_execution import format_iteration
from adk_rlm.repl.local_repl import LocalREPL
from adk_rlm.types import CodeBlock
from adk_rlm.types import REPLResult
from adk_rlm.types import RLMIteration
import pytest


class TestFindCodeBlocks:
  """Tests for find_code_blocks function."""

  def test_find_single_code_block(self):
    """One repl block."""
    text = """
Let me analyze this:
```repl
x = 42
print(x)
```
That should work.
"""
    blocks = find_code_blocks(text)
    assert len(blocks) == 1
    assert "x = 42" in blocks[0]
    assert "print(x)" in blocks[0]

  def test_find_multiple_code_blocks(self):
    """Multiple repl blocks."""
    text = """
First block:
```repl
x = 1
```

Second block:
```repl
y = 2
```
"""
    blocks = find_code_blocks(text)
    assert len(blocks) == 2
    assert "x = 1" in blocks[0]
    assert "y = 2" in blocks[1]

  def test_find_no_code_blocks(self):
    """No repl blocks."""
    text = "This is just plain text without any code blocks."
    blocks = find_code_blocks(text)
    assert blocks == []

  def test_ignore_other_languages(self):
    """Only extracts ```repl blocks."""
    text = """
```python
x = 1
```

```repl
y = 2
```

```javascript
z = 3
```
"""
    blocks = find_code_blocks(text)
    assert len(blocks) == 1
    assert "y = 2" in blocks[0]

  def test_preserve_indentation(self):
    """Code with indentation is preserved."""
    text = """
```repl
def foo():
    return 42

result = foo()
```
"""
    blocks = find_code_blocks(text)
    assert len(blocks) == 1
    assert "    return 42" in blocks[0]

  def test_preserve_blank_lines(self):
    """Code with blank lines."""
    text = """
```repl
x = 1

y = 2
```
"""
    blocks = find_code_blocks(text)
    assert len(blocks) == 1
    assert "\n\n" in blocks[0] or blocks[0].count("\n") >= 2


class TestFindFinalAnswer:
  """Tests for find_final_answer function."""

  def test_find_final_simple(self):
    """FINAL(answer)."""
    text = "Based on my analysis:\nFINAL(The answer is 42)"
    answer = find_final_answer(text)
    assert answer == "The answer is 42"

  def test_find_final_multiword(self):
    """FINAL with multiple words."""
    text = "FINAL(This is a longer answer with multiple words)"
    answer = find_final_answer(text)
    assert answer == "This is a longer answer with multiple words"

  def test_find_final_with_quotes(self):
    """FINAL with quoted content."""
    text = 'FINAL("quoted answer")'
    answer = find_final_answer(text)
    assert '"quoted answer"' in answer or "quoted answer" in answer

  def test_no_final_answer(self):
    """No FINAL pattern."""
    text = "This is just regular text without any final answer."
    answer = find_final_answer(text)
    assert answer is None

  def test_final_at_line_start(self):
    """FINAL must be at line start."""
    text = "Some text FINAL(not a final)"
    answer = find_final_answer(text)
    # Should not match if not at line start
    assert answer is None

  def test_final_with_leading_whitespace(self):
    """FINAL with leading whitespace is allowed."""
    text = "Some intro\n  FINAL(The answer)"
    answer = find_final_answer(text)
    assert answer == "The answer"

  def test_find_final_var(self, mock_llm_query):
    """FINAL_VAR resolves variable."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)
    repl.execute_code("my_result = 'Computed answer'")

    text = "After computation:\nFINAL_VAR(my_result)"
    answer = find_final_answer(text, repl)
    assert answer == "Computed answer"

  def test_find_final_var_quoted(self, mock_llm_query):
    """FINAL_VAR with quoted variable name."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)
    repl.execute_code("my_result = 'Answer here'")

    text = 'FINAL_VAR("my_result")'
    answer = find_final_answer(text, repl)
    assert answer == "Answer here"

  def test_find_final_var_missing(self, mock_llm_query):
    """FINAL_VAR with missing variable returns None (no final answer)."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)

    text = "FINAL_VAR(nonexistent)"
    answer = find_final_answer(text, repl)
    assert answer is None

  def test_multiple_finals_returns_first(self):
    """Multiple FINAL patterns returns first."""
    text = "FINAL(first)\nFINAL(second)"
    answer = find_final_answer(text)
    assert answer == "first"

  def test_final_var_takes_precedence(self, mock_llm_query):
    """FINAL_VAR is checked before FINAL."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)
    repl.execute_code("var = 'from variable'")

    text = "FINAL_VAR(var)\nFINAL(direct)"
    answer = find_final_answer(text, repl)
    assert answer == "from variable"


class TestFormatExecutionResult:
  """Tests for format_execution_result function."""

  def test_format_with_stdout(self):
    """Format result with stdout."""
    result = REPLResult(
        stdout="Hello, World!", stderr="", locals={"x": 42}, execution_time=0.1
    )
    formatted = format_execution_result(result)
    assert "Hello, World!" in formatted

  def test_format_with_stderr(self):
    """Format result with stderr."""
    result = REPLResult(
        stdout="", stderr="Error occurred", locals={}, execution_time=0.1
    )
    formatted = format_execution_result(result)
    assert "Error occurred" in formatted

  def test_format_with_variables(self):
    """Format result showing variables."""
    result = REPLResult(
        stdout="", stderr="", locals={"x": 42, "y": "hello"}, execution_time=0.1
    )
    formatted = format_execution_result(result)
    assert "REPL variables" in formatted or "x" in formatted or "y" in formatted

  def test_format_no_output(self):
    """Format result with no output."""
    result = REPLResult(stdout="", stderr="", locals={}, execution_time=0.1)
    formatted = format_execution_result(result)
    assert formatted  # Should return something


class TestFormatIteration:
  """Tests for format_iteration function."""

  def test_format_iteration_response(self):
    """Format assistant response."""
    iteration = RLMIteration(
        prompt="test", response="This is my response.", code_blocks=[]
    )
    messages = format_iteration(iteration)

    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"
    assert "This is my response." in messages[0]["content"]

  def test_format_iteration_with_code(self):
    """Format iteration with code execution."""
    result = REPLResult(
        stdout="42", stderr="", locals={"x": 42}, execution_time=0.1
    )
    code_block = CodeBlock(code="print(42)", result=result)
    iteration = RLMIteration(
        prompt="test", response="Let me calculate:", code_blocks=[code_block]
    )
    messages = format_iteration(iteration)

    assert len(messages) == 2
    assert messages[0]["role"] == "assistant"
    assert messages[1]["role"] == "user"
    assert "Code executed" in messages[1]["content"]

  def test_truncate_long_output(self):
    """Long output is truncated."""
    long_output = "x" * 30000
    result = REPLResult(
        stdout=long_output, stderr="", locals={}, execution_time=0.1
    )
    code_block = CodeBlock(code="print('x' * 30000)", result=result)
    iteration = RLMIteration(
        prompt="test", response="Computing...", code_blocks=[code_block]
    )

    messages = format_iteration(iteration, max_character_length=1000)

    # Should be truncated
    assert len(messages[1]["content"]) < 25000
