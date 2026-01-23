"""
End-to-end tests for ADK-RLM.

These tests use real Gemini API calls and are slow.
They are skipped by default unless RLM_E2E_TESTS=true is set.
"""

from functools import wraps
import time

from adk_rlm import completion
from adk_rlm import RLM
from adk_rlm import RLMEventType
import pytest


def retry_on_api_error(max_retries: int = 3, delay: float = 5.0):
  """Decorator to retry tests on transient API errors."""

  def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
      last_error = None
      for attempt in range(max_retries):
        try:
          return func(*args, **kwargs)
        except Exception as e:
          error_str = str(e).lower()
          # Retry on transient errors
          if any(
              x in error_str
              for x in [
                  "quota",
                  "rate",
                  "cancelled",
                  "503",
                  "500",
                  "overloaded",
              ]
          ):
            last_error = e
            if attempt < max_retries - 1:
              time.sleep(delay * (attempt + 1))
              continue
          raise
      raise last_error

    return wrapper

  return decorator


async def run_query(rlm: RLM, context: str, prompt: str) -> str:
  """Helper to run a query and return the final answer."""
  final_answer = None
  async for event in rlm.run_streaming(context, prompt):
    if event.custom_metadata:
      event_type = event.custom_metadata.get("event_type")
      if event_type == RLMEventType.FINAL_ANSWER.value:
        final_answer = event.custom_metadata.get("answer")
  return final_answer or ""


@pytest.mark.e2e
@pytest.mark.timeout(180)
class TestE2EBasicFunctionality:
  """Basic E2E tests."""

  @retry_on_api_error(max_retries=3, delay=5.0)
  def test_simple_computation(self):
    """Test that RLM can do simple computation via REPL."""
    result = completion(
        context="Calculate: 17 * 23",
        prompt=(
            "Compute 17 * 23 using Python code in the REPL. Return the result"
            " with FINAL()."
        ),
        model="gemini-3-flash-preview",
        max_iterations=10,
    )

    assert "391" in result.response

  @retry_on_api_error(max_retries=3, delay=5.0)
  def test_context_access(self, sample_context):
    """Test that RLM can access and analyze context."""
    result = completion(
        context=sample_context,
        prompt=(
            "Read the context variable and find the magic number. Print it"
            " using the REPL, then return it with FINAL()."
        ),
        model="gemini-3-flash-preview",
        max_iterations=10,
    )

    assert "42" in result.response

  @retry_on_api_error(max_retries=3, delay=5.0)
  def test_uses_llm_query(self, fixtures_dir):
    """Test that RLM uses llm_query for analysis."""
    context_file = fixtures_dir / "contexts" / "medium.txt"
    if not context_file.exists():
      pytest.skip("Medium context file not found")

    context = context_file.read_text()

    result = completion(
        context=context,
        prompt="What are the main topics covered in this document? Be brief.",
        model="gemini-3-flash-preview",
        sub_model="gemini-3-flash-preview",
        max_iterations=15,
    )

    # Check that we got a reasonable response
    assert len(result.response) > 50


@pytest.mark.e2e
@pytest.mark.timeout(300)
class TestE2EMultiTurn:
  """Multi-turn conversation tests."""

  async def test_context_accumulation(self):
    """Test that contexts accumulate across turns."""
    rlm = RLM(
        model="gemini-3-flash-preview",
        max_iterations=10,
        persistent=True,
        verbose=False,
    )
    try:
      # First turn - explicitly use REPL to read context
      result1 = await run_query(
          rlm,
          context="Alice is 30 years old.",
          prompt=(
              "Print the context variable and extract the age number. Return it"
              " with FINAL()."
          ),
      )
      assert "30" in result1

      # Second turn - should have access to first context via context_0
      result2 = await run_query(
          rlm,
          context="Bob is 25 years old.",
          prompt=(
              "Print context_0 to get Alice's age, and context_1 to get Bob's"
              " age. Who is older? Return just the name with FINAL()."
          ),
      )
      assert "Alice" in result2
    finally:
      rlm.close()


@pytest.mark.e2e
@pytest.mark.timeout(180)
class TestE2ELogging:
  """Test logging and tracing functionality."""

  @retry_on_api_error(max_retries=3, delay=5.0)
  def test_jsonl_logging(self, temp_log_dir):
    """Test that JSONL logs are created correctly."""
    import json
    from pathlib import Path

    result = completion(
        context="Test context",
        prompt="Just say FINAL(ok).",
        model="gemini-3-flash-preview",
        log_dir=temp_log_dir,
        max_iterations=5,
    )

    # Check log file was created
    log_files = list(Path(temp_log_dir).glob("*.jsonl"))
    assert len(log_files) == 1

    # Check log contents
    with open(log_files[0]) as f:
      lines = f.readlines()

    entries = [json.loads(line) for line in lines]
    assert entries[0]["type"] == "metadata"
    assert any(e["type"] == "iteration" for e in entries)
