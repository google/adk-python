"""
Tests for multi-turn persistence in ADK-RLM.

These tests verify that:
1. REPL environments persist across calls
2. Contexts accumulate (context_0, context_1, ...)
3. Histories accumulate (history_0, history_1, ...)
4. Variables persist across calls
"""

from adk_rlm.repl.local_repl import LocalREPL
import pytest


class TestLocalREPLMultiContext:
  """Tests for multi-context support."""

  def test_add_context_versioning(self, mock_llm_query):
    """Add_context creates versioned variables."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)
    repl.add_context("First", 0)
    repl.add_context("Second", 1)

    assert repl.locals["context_0"] == "First"
    assert repl.locals["context_1"] == "Second"
    assert repl.locals["context"] == "First"  # Alias to first
    assert repl.get_context_count() == 2

  def test_add_context_auto_increment(self, mock_llm_query):
    """Add_context auto-increments when no index provided."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)
    idx1 = repl.add_context("First")
    idx2 = repl.add_context("Second")

    assert idx1 == 0
    assert idx2 == 1
    assert repl.locals["context_0"] == "First"
    assert repl.locals["context_1"] == "Second"
    assert repl.get_context_count() == 2

  def test_contexts_accessible_in_code(self, mock_llm_query):
    """Multiple contexts can be accessed in code execution."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)
    repl.add_context("Document A content")
    repl.add_context("Document B content")

    result = repl.execute_code("combined = f'{context_0} + {context_1}'")
    assert result.stderr == ""
    assert repl.locals["combined"] == "Document A content + Document B content"

  def test_context_alias_points_to_first(self, mock_llm_query):
    """'context' always aliases context_0."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)
    repl.add_context("First")
    repl.add_context("Second")
    repl.add_context("Third")

    result = repl.execute_code("is_first = context == context_0")
    assert result.stderr == ""
    assert repl.locals["is_first"] is True


class TestLocalREPLHistory:
  """Tests for message history storage."""

  def test_add_history_basic(self, mock_llm_query):
    """Add_history stores message history correctly."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)

    history = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    index = repl.add_history(history)

    assert index == 0
    assert "history_0" in repl.locals
    assert "history" in repl.locals
    assert repl.locals["history_0"] == history
    assert repl.locals["history"] == history
    assert repl.get_history_count() == 1

  def test_add_multiple_histories(self, mock_llm_query):
    """Adding multiple conversation histories."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)

    history1 = [{"role": "user", "content": "First conversation"}]
    history2 = [{"role": "user", "content": "Second conversation"}]

    repl.add_history(history1)
    repl.add_history(history2)

    assert repl.get_history_count() == 2
    assert repl.locals["history_0"] == history1
    assert repl.locals["history_1"] == history2
    assert repl.locals["history"] == history1  # Alias stays on first

  def test_history_accessible_via_code(self, mock_llm_query):
    """Stored history is accessible via code execution."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)

    history = [{"role": "user", "content": "Test message"}]
    repl.add_history(history)

    result = repl.execute_code("msg = history[0]['content']")
    assert result.stderr == ""
    assert repl.locals["msg"] == "Test message"

  def test_history_is_copy(self, mock_llm_query):
    """Stored history is a copy, not a reference."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)

    history = [{"role": "user", "content": "Original"}]
    repl.add_history(history)

    # Modify original
    history[0]["content"] = "Modified"

    # Stored copy should be unchanged
    assert repl.locals["history_0"][0]["content"] == "Original"

  def test_can_iterate_histories_in_code(self, mock_llm_query):
    """Iterating through multiple histories in code."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)

    repl.add_history([{"role": "user", "content": "Query 1"}])
    repl.add_history([{"role": "user", "content": "Query 2"}])
    repl.add_history([{"role": "user", "content": "Query 3"}])

    code = """
all_contents = [
    history_0[0]['content'],
    history_1[0]['content'],
    history_2[0]['content'],
]
"""
    result = repl.execute_code(code)
    assert result.stderr == ""
    assert repl.locals["all_contents"] == ["Query 1", "Query 2", "Query 3"]


class TestLocalREPLPersistentState:
  """Tests for state persistence across operations."""

  def test_variables_persist_with_contexts(self, mock_llm_query):
    """Variables and contexts coexist."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)

    repl.add_context("My context data")
    repl.execute_code("summary = context.upper()")
    assert repl.locals["summary"] == "MY CONTEXT DATA"

    repl.add_context("New context")

    # Previous variable should still exist
    assert repl.locals["summary"] == "MY CONTEXT DATA"
    assert repl.locals["context_1"] == "New context"

  def test_variables_persist_with_histories(self, mock_llm_query):
    """Variables and histories coexist."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)

    repl.add_history([{"role": "user", "content": "Hello"}])
    repl.execute_code("extracted = history[0]['content']")
    assert repl.locals["extracted"] == "Hello"

    repl.add_history([{"role": "user", "content": "World"}])

    # Previous variable should still exist
    assert repl.locals["extracted"] == "Hello"
    assert repl.locals["history_1"][0]["content"] == "World"

  def test_full_persistent_session_simulation(self, mock_llm_query):
    """Simulate a multi-turn persistent session."""
    repl = LocalREPL(llm_query_fn=mock_llm_query)

    # Turn 1: Load first document
    repl.add_context("Document: Sales were $1000")
    repl.execute_code("sales = 1000")

    # Turn 2: Load second document, use previous variable
    repl.add_context("Document: Costs were $400")
    result = repl.execute_code("profit = sales - 400")
    assert result.stderr == ""
    assert repl.locals["profit"] == 600

    # Turn 3: Store history and reference everything
    repl.add_history([
        {"role": "user", "content": "What were the sales?"},
        {"role": "assistant", "content": "Sales were $1000"},
    ])

    code = """
summary = f"Sales: {context_0}, Costs: {context_1}, Profit: {profit}"
prev_question = history_0[0]['content']
"""
    result = repl.execute_code(code)
    assert result.stderr == ""
    assert "Profit: 600" in repl.locals["summary"]
    assert repl.locals["prev_question"] == "What were the sales?"

    assert repl.get_context_count() == 2
    assert repl.get_history_count() == 1


class TestNonPersistentBehavior:
  """Tests simulating non-persistent RLM behavior."""

  def test_simulated_non_persistent_completions(self, mock_llm_query):
    """Simulate 2 RLM completions to show env resets between calls."""
    # Completion 1
    completion_1_env = LocalREPL(llm_query_fn=mock_llm_query)
    completion_1_env.execute_code("important_result = 42")
    assert completion_1_env.locals["important_result"] == 42
    completion_1_env.cleanup()

    # Completion 2 - fresh environment
    completion_2_env = LocalREPL(llm_query_fn=mock_llm_query)
    result = completion_2_env.execute_code("print(important_result)")

    assert "NameError" in result.stderr
    assert "important_result" in result.stderr
    completion_2_env.cleanup()

  def test_simulated_non_persistent_functions(self, mock_llm_query):
    """Simulate 2 RLM completions to show functions don't persist."""
    # Completion 1
    completion_1_env = LocalREPL(llm_query_fn=mock_llm_query)
    completion_1_env.execute_code("def my_helper(): return 'useful'")
    assert (
        completion_1_env.execute_code("print(my_helper())").stdout.strip()
        == "useful"
    )
    completion_1_env.cleanup()

    # Completion 2 - fresh environment
    completion_2_env = LocalREPL(llm_query_fn=mock_llm_query)
    result = completion_2_env.execute_code("my_helper()")

    assert "NameError" in result.stderr
    assert "my_helper" in result.stderr
    completion_2_env.cleanup()
