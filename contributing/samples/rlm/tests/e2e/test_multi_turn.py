"""
E2E tests for multi-turn conversations and file handling.

These tests verify conversation context, file loading, and complex
interaction patterns with a real server.
"""

import os
from pathlib import Path
import re
import tempfile

from playwright.sync_api import expect
from playwright.sync_api import Page
import pytest

from .conftest import get_answer_text
from .conftest import get_message_count
from .conftest import submit_query
from .conftest import wait_for_query_complete

pytestmark = [
    pytest.mark.e2e_web,
    pytest.mark.skipif(
        os.environ.get("RLM_E2E_TESTS") != "true",
        reason="E2E tests disabled. Set RLM_E2E_TESTS=true to enable.",
    ),
]


class TestMultiTurnConversation:
  """Tests for multi-turn conversation handling."""

  def test_conversation_history_grows(self, e2e_page: Page):
    """Test that conversation history grows with each turn."""
    # First turn
    submit_query(e2e_page, "Hello, my name is Alice")
    wait_for_query_complete(e2e_page, timeout=60000)

    first_count = get_message_count(e2e_page)
    assert first_count >= 2, "Should have user message and response"

    # Second turn
    submit_query(e2e_page, "What is my name?")
    wait_for_query_complete(e2e_page, timeout=60000)

    second_count = get_message_count(e2e_page)
    assert second_count >= 4, f"Should have 4+ messages, got {second_count}"

    # Third turn
    submit_query(e2e_page, "Thanks for remembering!")
    wait_for_query_complete(e2e_page, timeout=60000)

    third_count = get_message_count(e2e_page)
    assert third_count >= 6, f"Should have 6+ messages, got {third_count}"

  def test_context_maintained_across_turns(self, e2e_page: Page):
    """Test that context is maintained across conversation turns."""
    # Establish context
    submit_query(e2e_page, "Let x = 100")
    wait_for_query_complete(e2e_page, timeout=60000)

    # Reference context
    submit_query(e2e_page, "What is x + 50?")
    wait_for_query_complete(e2e_page, timeout=60000)

    # Answer should reference the context
    answer = get_answer_text(e2e_page)
    # Should contain 150 or reference to x
    assert (
        "150" in answer or "x" in answer.lower()
    ), f"Unexpected answer: {answer}"

  def test_follow_up_questions(self, e2e_page: Page):
    """Test asking follow-up questions about previous answers."""
    # Initial question
    submit_query(e2e_page, "What is the capital of France?")
    wait_for_query_complete(e2e_page, timeout=60000)

    # Follow-up
    submit_query(e2e_page, "What is its population?")
    wait_for_query_complete(e2e_page, timeout=60000)

    # Should understand "its" refers to Paris
    answer = get_answer_text(e2e_page)
    # Should mention numbers (population) or Paris
    assert (
        any(c.isdigit() for c in answer) or "paris" in answer.lower()
    ), f"Expected population info, got: {answer}"


class TestCodeExecutionAcrossTurns:
  """Tests for code execution across multiple turns."""

  def test_variables_persist_across_turns(self, e2e_page: Page):
    """Test that REPL variables persist across conversation turns."""
    # Define a variable
    submit_query(e2e_page, "Calculate result = 2 ** 10 and show me the value")
    wait_for_query_complete(e2e_page, timeout=60000)

    first_answer = get_answer_text(e2e_page)
    assert "1024" in first_answer, f"Expected 1024 in answer: {first_answer}"

    # Use the variable in next turn
    submit_query(e2e_page, "Now divide result by 2")
    wait_for_query_complete(e2e_page, timeout=60000)

    second_answer = get_answer_text(e2e_page)
    assert "512" in second_answer, f"Expected 512 in answer: {second_answer}"

  def test_function_definition_persists(self, e2e_page: Page):
    """Test that function definitions persist across turns."""
    # Define a function
    submit_query(
        e2e_page,
        "Define a function called double that returns its argument times 2",
    )
    wait_for_query_complete(e2e_page, timeout=60000)

    # Use the function
    submit_query(e2e_page, "Use the double function on 21")
    wait_for_query_complete(e2e_page, timeout=60000)

    answer = get_answer_text(e2e_page)
    assert "42" in answer, f"Expected 42 in answer: {answer}"


class TestFileHandling:
  """Tests for file handling functionality."""

  @pytest.fixture
  def test_files(self, tmp_path: Path) -> dict:
    """Create temporary test files."""
    # Create a text file
    txt_file = tmp_path / "test_data.txt"
    txt_file.write_text("This is test content.\nLine 2.\nLine 3.")

    # Create a markdown file
    md_file = tmp_path / "readme.md"
    md_file.write_text("# Test Document\n\nThis is a test markdown file.")

    # Create a Python file
    py_file = tmp_path / "sample.py"
    py_file.write_text("def hello():\n    return 'Hello, World!'\n")

    return {
        "txt": str(txt_file),
        "md": str(md_file),
        "py": str(py_file),
        "dir": str(tmp_path),
    }

  def test_add_files_via_settings(self, e2e_page: Page, test_files: dict):
    """Test adding files via the settings modal."""
    # Open settings
    e2e_page.locator("#config-btn").click()

    # Add file pattern
    files_input = e2e_page.locator("#config-files")
    files_input.fill(f"{test_files['dir']}/*.txt")

    # Save
    e2e_page.locator("#config-form button[type='submit']").click()

    e2e_page.wait_for_timeout(500)

    # Files section should be visible
    files_section = e2e_page.locator("#files-section")
    expect(files_section).not_to_have_class(re.compile(r"hidden"))

  def test_query_with_file_context(self, e2e_page: Page, test_files: dict):
    """Test querying with file context."""
    # Open settings and add files
    e2e_page.locator("#config-btn").click()
    e2e_page.locator("#config-files").fill(f"{test_files['dir']}/*.txt")
    e2e_page.locator("#config-form button[type='submit']").click()

    e2e_page.wait_for_timeout(500)

    # Query about the file content
    submit_query(e2e_page, "How many lines are in the text file?")
    wait_for_query_complete(e2e_page, timeout=60000)

    answer = get_answer_text(e2e_page)
    # Should mention 3 lines
    assert "3" in answer, f"Expected '3' in answer about lines: {answer}"

  def test_invalid_file_pattern_shows_error(self, e2e_page: Page):
    """Test that invalid file patterns show an error."""
    # Open settings and add non-existent pattern
    e2e_page.locator("#config-btn").click()
    e2e_page.locator("#config-files").fill("/nonexistent/path/*.xyz")
    e2e_page.locator("#config-form button[type='submit']").click()

    e2e_page.wait_for_timeout(500)

    # Files section should remain hidden (no files matched)
    files_section = e2e_page.locator("#files-section")
    expect(files_section).to_have_class(re.compile(r"hidden"))


class TestComplexInteractions:
  """Tests for complex interaction patterns."""

  def test_rapid_queries(self, e2e_page: Page):
    """Test that the system handles queries correctly (one at a time)."""
    # Submit first query
    submit_query(e2e_page, "What is 1+1?")
    wait_for_query_complete(e2e_page, timeout=60000)

    # Submit second query immediately after
    submit_query(e2e_page, "What is 2+2?")
    wait_for_query_complete(e2e_page, timeout=60000)

    # Should have 4 messages (2 user + 2 assistant)
    assert get_message_count(e2e_page) >= 4

  def test_long_conversation(self, e2e_page: Page):
    """Test a longer conversation with multiple turns."""
    queries = [
        "Let's count. Start with 1.",
        "Add 1 to get the next number.",
        "Add 1 again.",
        "What number are we at now?",
    ]

    for query in queries:
      submit_query(e2e_page, query)
      wait_for_query_complete(e2e_page, timeout=60000)

    # Should have 8 messages (4 user + 4 assistant)
    message_count = get_message_count(e2e_page)
    assert message_count >= 8, f"Expected 8+ messages, got {message_count}"

    # Final answer should mention 4 (or 3, depending on interpretation)
    answer = get_answer_text(e2e_page)
    assert any(
        n in answer for n in ["3", "4"]
    ), f"Expected 3 or 4 in answer: {answer}"


class TestEdgeCases:
  """Tests for edge cases and boundary conditions."""

  def test_empty_response_handling(self, e2e_page: Page):
    """Test handling of queries that might produce empty responses."""
    submit_query(e2e_page, "Just say 'OK' and nothing else")
    wait_for_query_complete(e2e_page, timeout=60000)

    # Should still have an answer
    answer_panels = e2e_page.locator(".answer-panel")
    expect(answer_panels).to_have_count(1)

  def test_special_characters_in_query(self, e2e_page: Page):
    """Test queries with special characters."""
    submit_query(e2e_page, "What is 'hello' + \" world\" in Python?")
    wait_for_query_complete(e2e_page, timeout=60000)

    # Should handle special characters
    answer = get_answer_text(e2e_page)
    assert "hello" in answer.lower() or "world" in answer.lower()

  def test_unicode_in_query(self, e2e_page: Page):
    """Test queries with unicode characters."""
    submit_query(e2e_page, "Print the emoji: \U0001F600")
    wait_for_query_complete(e2e_page, timeout=60000)

    # Should complete without error
    answer_panels = e2e_page.locator(".answer-panel")
    expect(answer_panels).to_have_count(1)

  def test_very_long_query(self, e2e_page: Page):
    """Test handling of a very long query."""
    long_query = "Calculate the sum of: " + ", ".join(
        str(i) for i in range(1, 51)
    )
    submit_query(e2e_page, long_query)
    wait_for_query_complete(e2e_page, timeout=90000)

    # Should complete and have an answer
    answer = get_answer_text(e2e_page)
    # Sum of 1 to 50 is 1275
    assert "1275" in answer, f"Expected 1275 in answer: {answer}"
