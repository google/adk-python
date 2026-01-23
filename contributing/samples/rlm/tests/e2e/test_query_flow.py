"""
E2E tests for the full query flow.

These tests run against a real server with real WebSocket connections
and real session persistence. The LLM calls go to the real API.

Note: These tests require API access and may be slow. They are marked
with @pytest.mark.e2e_web and can be skipped with -m "not e2e_web".
"""

import os
import re

from playwright.sync_api import expect
from playwright.sync_api import Page
import pytest

from .conftest import get_answer_text
from .conftest import get_event_count
from .conftest import get_message_count
from .conftest import submit_query
from .conftest import wait_for_answer
from .conftest import wait_for_query_complete

pytestmark = [
    pytest.mark.e2e_web,
    pytest.mark.skipif(
        os.environ.get("RLM_E2E_TESTS") != "true",
        reason="E2E tests disabled. Set RLM_E2E_TESTS=true to enable.",
    ),
]


class TestBasicQuery:
  """Tests for basic query submission and response."""

  def test_simple_query_flow(self, e2e_page: Page):
    """
    Test a simple query that should complete successfully.

    This test submits a query and verifies:
    - User message appears
    - Processing indicator shows
    - Events are generated
    - Answer is displayed
    """
    # Submit a simple query
    submit_query(e2e_page, "What is 2 + 2? Just give me the number.")

    # Verify user message appeared
    user_message = e2e_page.locator(".message.user")
    expect(user_message).to_be_visible()
    expect(user_message).to_contain_text("What is 2 + 2")

    # Wait for completion (with generous timeout for LLM)
    wait_for_query_complete(e2e_page, timeout=60000)

    # Verify we got some kind of response (answer or error)
    # Either answer-panel or an error message
    response = e2e_page.locator(
        ".answer-panel, .message.assistant .message-content"
    )
    expect(response.first).to_be_visible(timeout=10000)

    # Verify events were generated
    event_count = get_event_count(e2e_page)
    assert event_count > 0, "Should have generated some events"

  def test_query_generates_events(self, e2e_page: Page):
    """Test that queries generate streaming events in the event log."""
    submit_query(e2e_page, "Calculate 10 * 5")

    # Wait for some events to appear
    e2e_page.wait_for_function(
        "() => parseInt(document.querySelector('#event-count')?.textContent ||"
        " '0') > 0",
        timeout=30000,
    )

    # Verify event items are visible
    event_log = e2e_page.locator("#event-log-content")
    expect(event_log).not_to_contain_text("Events will appear here")

    # Should have agent/iteration groups
    wait_for_query_complete(e2e_page, timeout=60000)
    event_count = get_event_count(e2e_page)
    assert event_count >= 3, f"Expected at least 3 events, got {event_count}"

  def test_processing_state_during_query(self, e2e_page: Page):
    """Test that processing indicator shows during query execution."""
    prompt_input = e2e_page.locator("#prompt-input")
    prompt_input.fill("What is the square root of 144?")

    # Submit and immediately check processing state
    prompt_input.press("Enter")

    # Processing should be visible
    processing = e2e_page.locator("#processing")
    expect(processing).not_to_have_class(re.compile(r"hidden"))

    # Send button should be disabled
    send_btn = e2e_page.locator("#send-btn")
    expect(send_btn).to_be_disabled()

    # Wait for completion
    wait_for_query_complete(e2e_page, timeout=60000)

    # Processing should be hidden
    expect(processing).to_have_class(re.compile(r"hidden"))

    # Send button should be enabled
    expect(send_btn).to_be_enabled()


class TestMultipleQueries:
  """Tests for multiple sequential queries."""

  def test_two_queries_in_sequence(self, e2e_page: Page):
    """Test submitting two queries in sequence."""
    # First query
    submit_query(e2e_page, "What is 5 + 5?")
    wait_for_query_complete(e2e_page, timeout=60000)

    # Verify first answer
    expect(e2e_page.locator(".answer-panel")).to_have_count(1)

    # Second query
    submit_query(e2e_page, "What is 10 + 10?")
    wait_for_query_complete(e2e_page, timeout=60000)

    # Should have two user messages and two answers
    user_messages = e2e_page.locator(".message.user")
    expect(user_messages).to_have_count(2)

    answer_panels = e2e_page.locator(".answer-panel")
    expect(answer_panels).to_have_count(2)

  def test_conversation_context_maintained(self, e2e_page: Page):
    """Test that conversation context is maintained across queries."""
    # First query establishes context
    submit_query(e2e_page, "Remember this number: 42")
    wait_for_query_complete(e2e_page, timeout=60000)

    # Second query references context
    submit_query(e2e_page, "What number did I ask you to remember?")
    wait_for_query_complete(e2e_page, timeout=60000)

    # The answer should reference 42
    answer_text = get_answer_text(e2e_page)
    assert "42" in answer_text, f"Expected '42' in answer, got: {answer_text}"


class TestEventLogDuringQuery:
  """Tests for event log behavior during query execution."""

  def test_events_stream_in_realtime(self, e2e_page: Page):
    """Test that events stream in real-time during query execution."""
    submit_query(e2e_page, "Count from 1 to 5")

    # Wait for at least one event
    e2e_page.wait_for_function(
        "() => parseInt(document.querySelector('#event-count')?.textContent ||"
        " '0') > 0",
        timeout=30000,
    )

    first_count = get_event_count(e2e_page)
    assert first_count > 0, "Should have at least one event"

    # Wait for more events
    e2e_page.wait_for_timeout(1000)

    # If query is still running, count should increase
    # (or be the same if query completed quickly)
    wait_for_query_complete(e2e_page, timeout=60000)

    final_count = get_event_count(e2e_page)
    assert final_count >= first_count, "Event count should not decrease"

  def test_event_details_viewable(self, e2e_page: Page):
    """Test that event details can be viewed by clicking events."""
    submit_query(e2e_page, "What is 3 * 3?")
    wait_for_query_complete(e2e_page, timeout=60000)

    # Click on first event
    event_item = e2e_page.locator(".event-item").first
    event_item.click()

    # Modal should open
    event_modal = e2e_page.locator("#event-modal")
    expect(event_modal).not_to_have_class(re.compile(r"hidden"))

    # Modal should have content
    modal_body = e2e_page.locator("#modal-body")
    expect(modal_body).to_contain_text("Event Type")


class TestErrorHandling:
  """Tests for error handling during queries."""

  def test_query_with_invalid_code_recovers(self, e2e_page: Page):
    """Test that the system can recover from code execution errors."""
    # This query might generate code that errors, but should still complete
    submit_query(
        e2e_page,
        "Try to calculate something that might fail, then recover and give"
        " me 42",
    )

    # Should eventually complete (possibly with error recovery)
    wait_for_query_complete(e2e_page, timeout=90000)

    # Should have an answer (even if it's about the error)
    answer_panels = e2e_page.locator(".answer-panel")
    # Either we get an answer or an error message
    messages = e2e_page.locator(".message.assistant")
    assert messages.count() > 0, "Should have some response"


class TestUIStateAfterQuery:
  """Tests for UI state after query completion."""

  def test_input_cleared_after_submit(self, e2e_page: Page):
    """Test that input is cleared after submitting query."""
    prompt_input = e2e_page.locator("#prompt-input")
    prompt_input.fill("Test query")
    prompt_input.press("Enter")

    # Input should be cleared immediately
    expect(prompt_input).to_have_value("")

  def test_can_submit_new_query_after_completion(self, e2e_page: Page):
    """Test that new queries can be submitted after completion."""
    # First query
    submit_query(e2e_page, "Say hello")
    wait_for_query_complete(e2e_page, timeout=60000)

    # Should be able to type and submit again
    prompt_input = e2e_page.locator("#prompt-input")
    prompt_input.fill("Say goodbye")

    send_btn = e2e_page.locator("#send-btn")
    expect(send_btn).to_be_enabled()

    send_btn.click()

    # Should start processing
    processing = e2e_page.locator("#processing")
    expect(processing).not_to_have_class(re.compile(r"hidden"))

    wait_for_query_complete(e2e_page, timeout=60000)

  def test_session_title_updates_from_first_message(self, e2e_page: Page):
    """Test that session title is auto-generated from first message."""
    # Get initial title
    session_title = e2e_page.locator("#session-title")
    initial_title = session_title.text_content()

    # Submit first query
    submit_query(e2e_page, "This is my test question about Python programming")
    wait_for_query_complete(e2e_page, timeout=60000)

    # Title should be updated to reflect the query
    e2e_page.wait_for_timeout(500)  # Allow UI to update
    new_title = session_title.text_content()

    # Title should have changed and contain part of the query
    assert new_title != initial_title or "Python" in (new_title or "")
