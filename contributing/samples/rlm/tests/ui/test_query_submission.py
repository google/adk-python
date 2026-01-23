"""
UI tests for query submission and processing.

These tests verify the query input, submission, and response handling behavior.
"""

import re

from playwright.sync_api import expect
from playwright.sync_api import Page
import pytest

from .conftest import create_mock_event
from .conftest import create_mock_session
from .conftest import create_mock_sessions_list
from .conftest import WebSocketInterceptor

pytestmark = pytest.mark.ui


class TestInputArea:
  """Tests for the input textarea and send button."""

  def test_input_accepts_text(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Input textarea should accept text."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)
    mock_ws.wait_for_connection()

    prompt_input = page.locator("#prompt-input")
    prompt_input.fill("What is 2+2?")

    expect(prompt_input).to_have_value("What is 2+2?")

  def test_input_placeholder(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Input should have placeholder text."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)

    prompt_input = page.locator("#prompt-input")
    expect(prompt_input).to_have_attribute("placeholder", "Ask a question...")

  def test_enter_submits_message(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Pressing Enter should submit the message."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)
    mock_ws.wait_for_connection()

    # Clear received messages
    page.evaluate("() => window._mockWsReceived = []")

    prompt_input = page.locator("#prompt-input")
    prompt_input.fill("Test query")
    prompt_input.press("Enter")

    page.wait_for_timeout(100)

    received = mock_ws.get_received_messages()
    query_msgs = [m for m in received if m.get("action") == "query"]

    assert len(query_msgs) == 1
    assert query_msgs[0]["prompt"] == "Test query"

  def test_shift_enter_adds_newline(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Pressing Shift+Enter should add a newline, not submit."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)
    mock_ws.wait_for_connection()

    # Clear received messages
    page.evaluate("() => window._mockWsReceived = []")

    prompt_input = page.locator("#prompt-input")
    prompt_input.fill("Line 1")
    prompt_input.press("Shift+Enter")
    prompt_input.type("Line 2")

    page.wait_for_timeout(100)

    # Value should contain newline
    value = prompt_input.input_value()
    assert "Line 1" in value
    assert "Line 2" in value

    # Should NOT have sent a query
    received = mock_ws.get_received_messages()
    query_msgs = [m for m in received if m.get("action") == "query"]
    assert len(query_msgs) == 0

  def test_send_button_submits_message(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Clicking send button should submit the message."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)
    mock_ws.wait_for_connection()

    # Clear received messages
    page.evaluate("() => window._mockWsReceived = []")

    prompt_input = page.locator("#prompt-input")
    send_btn = page.locator("#send-btn")

    prompt_input.fill("Button test query")
    send_btn.click()

    page.wait_for_timeout(100)

    received = mock_ws.get_received_messages()
    query_msgs = [m for m in received if m.get("action") == "query"]

    assert len(query_msgs) == 1
    assert query_msgs[0]["prompt"] == "Button test query"

  def test_input_cleared_after_submit(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Input should be cleared after submitting."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)
    mock_ws.wait_for_connection()

    prompt_input = page.locator("#prompt-input")
    prompt_input.fill("Test query")
    prompt_input.press("Enter")

    page.wait_for_timeout(100)

    expect(prompt_input).to_have_value("")

  def test_empty_input_not_submitted(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Empty input should not be submitted."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)
    mock_ws.wait_for_connection()

    # Clear received messages
    page.evaluate("() => window._mockWsReceived = []")

    prompt_input = page.locator("#prompt-input")
    prompt_input.press("Enter")

    page.wait_for_timeout(100)

    received = mock_ws.get_received_messages()
    query_msgs = [m for m in received if m.get("action") == "query"]

    assert len(query_msgs) == 0


class TestUserMessage:
  """Tests for user message display."""

  def test_user_message_displayed(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Submitted query should appear as user message."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)
    mock_ws.wait_for_connection()

    prompt_input = page.locator("#prompt-input")
    prompt_input.fill("What is the capital of France?")
    prompt_input.press("Enter")

    page.wait_for_timeout(100)

    # User message should appear
    user_message = page.locator(".message.user")
    expect(user_message).to_be_visible()
    expect(user_message).to_contain_text("What is the capital of France?")

  def test_empty_state_hidden_after_message(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Empty state should be hidden after sending a message."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)
    mock_ws.wait_for_connection()

    # Empty state should be visible initially
    empty_state = page.locator("#empty-state")
    expect(empty_state).to_be_visible()

    # Send a message
    prompt_input = page.locator("#prompt-input")
    prompt_input.fill("Hello")
    prompt_input.press("Enter")

    page.wait_for_timeout(100)

    # Empty state should be hidden
    expect(empty_state).not_to_be_visible()


class TestProcessingState:
  """Tests for processing state during query execution."""

  def test_processing_indicator_shows(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Processing indicator should appear during query execution."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])
    mock_ws.set_auto_response(
        "query",
        [
            {"type": "query_start", "prompt": "Test"},
        ],
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()

    prompt_input = page.locator("#prompt-input")
    prompt_input.fill("Test query")
    prompt_input.press("Enter")

    page.wait_for_timeout(100)

    # Processing indicator should be visible
    processing = page.locator("#processing")
    expect(processing).not_to_have_class(re.compile(r"hidden"))

  def test_send_button_disabled_during_processing(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Send button should be disabled during processing."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])
    mock_ws.set_auto_response(
        "query",
        [
            {"type": "query_start", "prompt": "Test"},
        ],
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()

    send_btn = page.locator("#send-btn")
    prompt_input = page.locator("#prompt-input")

    prompt_input.fill("Test query")
    prompt_input.press("Enter")

    page.wait_for_timeout(100)

    # Send button should be disabled
    expect(send_btn).to_be_disabled()

  def test_processing_text_updates(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Processing text should update based on events."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])
    mock_ws.set_auto_response(
        "query",
        [
            {"type": "query_start", "prompt": "Test"},
            create_mock_event("rlm.run.start", iteration=0, event_id=0),
            create_mock_event("rlm.iteration.start", iteration=1, event_id=1),
        ],
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()

    prompt_input = page.locator("#prompt-input")
    prompt_input.fill("Test query")
    prompt_input.press("Enter")

    page.wait_for_timeout(200)

    # Processing text should show iteration
    processing_text = page.locator("#processing-text")
    expect(processing_text).to_contain_text("Iteration 1")


class TestQueryResponse:
  """Tests for query response handling."""

  def test_final_answer_displayed(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Final answer should be displayed in answer panel."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])
    mock_ws.set_auto_response(
        "query",
        [
            {"type": "query_start", "prompt": "What is 2+2?"},
            create_mock_event("rlm.run.start", iteration=0, event_id=0),
            create_mock_event("rlm.iteration.start", iteration=1, event_id=1),
            create_mock_event("rlm.final.detected", iteration=1, event_id=2),
            create_mock_event("rlm.run.end", iteration=1, event_id=3),
            {
                "type": "query_complete",
                "elapsed_seconds": 1.5,
                "total_events": 4,
                "final_answer": "The answer is 4",
                "title": "What is 2+2?",
            },
        ],
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()

    prompt_input = page.locator("#prompt-input")
    prompt_input.fill("What is 2+2?")
    prompt_input.press("Enter")

    page.wait_for_timeout(300)

    # Answer panel should be visible with answer
    answer_panel = page.locator(".answer-panel")
    expect(answer_panel).to_be_visible()
    expect(answer_panel).to_contain_text("The answer is 4")

  def test_processing_hidden_after_completion(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Processing indicator should be hidden after completion."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])
    mock_ws.set_auto_response(
        "query",
        [
            {"type": "query_start", "prompt": "Test"},
            {
                "type": "query_complete",
                "elapsed_seconds": 1.0,
                "total_events": 0,
                "final_answer": "Done",
                "title": "Test",
            },
        ],
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()

    prompt_input = page.locator("#prompt-input")
    prompt_input.fill("Test")
    prompt_input.press("Enter")

    page.wait_for_timeout(300)

    # Processing should be hidden
    processing = page.locator("#processing")
    expect(processing).to_have_class(re.compile(r"hidden"))

  def test_send_button_enabled_after_completion(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Send button should be re-enabled after completion."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])
    mock_ws.set_auto_response(
        "query",
        [
            {"type": "query_start", "prompt": "Test"},
            {
                "type": "query_complete",
                "elapsed_seconds": 1.0,
                "total_events": 0,
                "final_answer": "Done",
                "title": "Test",
            },
        ],
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()

    send_btn = page.locator("#send-btn")
    prompt_input = page.locator("#prompt-input")

    prompt_input.fill("Test")
    prompt_input.press("Enter")

    page.wait_for_timeout(300)

    # Send button should be enabled
    expect(send_btn).to_be_enabled()


class TestErrorHandling:
  """Tests for error handling during query execution."""

  def test_error_message_displayed(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Error message should be displayed when query fails."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])
    mock_ws.set_auto_response(
        "query",
        [
            {"type": "query_start", "prompt": "Test"},
            {"type": "error", "message": "Something went wrong!"},
        ],
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()

    prompt_input = page.locator("#prompt-input")
    prompt_input.fill("Test")
    prompt_input.press("Enter")

    page.wait_for_timeout(200)

    # Error message should be displayed
    error_message = page.locator(".message.assistant").last
    expect(error_message).to_contain_text("Something went wrong!")

  def test_processing_ends_on_error(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Processing should end when error occurs."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])
    mock_ws.set_auto_response(
        "query",
        [
            {"type": "query_start", "prompt": "Test"},
            {"type": "error", "message": "Error occurred"},
        ],
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()

    prompt_input = page.locator("#prompt-input")
    prompt_input.fill("Test")
    prompt_input.press("Enter")

    page.wait_for_timeout(200)

    # Processing should be hidden
    processing = page.locator("#processing")
    expect(processing).to_have_class(re.compile(r"hidden"))

    # Send button should be enabled
    send_btn = page.locator("#send-btn")
    expect(send_btn).to_be_enabled()


class TestConversationRestore:
  """Tests for conversation restoration."""

  def test_conversation_restored_on_load(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Existing conversation should be restored on page load."""
    mock_ws.setup()
    mock_ws.set_auto_response(
        "get_status",
        [
            create_mock_session(
                conversation=[
                    {
                        "role": "user",
                        "content": "Hello",
                        "timestamp": "2024-01-15T10:00:00",
                    },
                    {
                        "role": "assistant",
                        "content": "Hi there!",
                        "timestamp": "2024-01-15T10:00:05",
                    },
                    {
                        "role": "user",
                        "content": "How are you?",
                        "timestamp": "2024-01-15T10:01:00",
                    },
                    {
                        "role": "assistant",
                        "content": "I'm doing well!",
                        "timestamp": "2024-01-15T10:01:05",
                    },
                ]
            )
        ],
    )
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)
    mock_ws.wait_for_connection()
    page.wait_for_timeout(300)

    # Should have 4 messages
    messages = page.locator(".message")
    expect(messages).to_have_count(4)

    # Verify content
    user_messages = page.locator(".message.user")
    expect(user_messages).to_have_count(2)
    expect(user_messages.first).to_contain_text("Hello")

    assistant_messages = page.locator(".message.assistant")
    expect(assistant_messages).to_have_count(2)
    expect(assistant_messages.first).to_contain_text("Hi there!")
