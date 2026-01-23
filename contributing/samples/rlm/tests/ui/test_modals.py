"""
UI tests for modal dialogs.

These tests verify the settings modal and event detail modal behavior.
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


class TestSettingsModal:
  """Tests for the settings/config modal."""

  def test_settings_button_opens_modal(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Clicking settings button should open config modal."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)
    mock_ws.wait_for_connection()

    config_modal = page.locator("#config-modal")
    settings_btn = page.locator("#config-btn")

    # Modal should be hidden initially
    expect(config_modal).to_have_class(re.compile(r"hidden"))

    # Click settings button
    settings_btn.click()

    # Modal should be visible
    expect(config_modal).not_to_have_class(re.compile(r"hidden"))

  def test_modal_displays_session_title(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Settings modal should display current session title."""
    mock_ws.setup()
    mock_ws.set_auto_response(
        "get_status", [create_mock_session(title="My Session")]
    )
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)
    mock_ws.wait_for_connection()
    page.wait_for_timeout(200)

    page.locator("#config-btn").click()

    title_input = page.locator("#config-title")
    expect(title_input).to_have_value("My Session")

  def test_modal_displays_model(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Settings modal should display current model."""
    mock_ws.setup()
    mock_ws.set_auto_response(
        "get_status", [create_mock_session(model="gemini-3-flash-preview")]
    )
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)
    mock_ws.wait_for_connection()
    page.wait_for_timeout(200)

    page.locator("#config-btn").click()

    model_input = page.locator("#config-model")
    expect(model_input).to_have_value("gemini-3-flash-preview")

  def test_close_button_closes_modal(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Close button should close the modal."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)
    mock_ws.wait_for_connection()

    config_modal = page.locator("#config-modal")
    page.locator("#config-btn").click()
    expect(config_modal).not_to_have_class(re.compile(r"hidden"))

    # Click close button
    page.locator("#config-modal-close").click()

    expect(config_modal).to_have_class(re.compile(r"hidden"))

  def test_cancel_button_closes_modal(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Cancel button should close the modal."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)
    mock_ws.wait_for_connection()

    config_modal = page.locator("#config-modal")
    page.locator("#config-btn").click()
    expect(config_modal).not_to_have_class(re.compile(r"hidden"))

    # Click cancel button
    page.locator("#config-cancel").click()

    expect(config_modal).to_have_class(re.compile(r"hidden"))

  def test_click_outside_closes_modal(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Clicking outside modal should close it."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)
    mock_ws.wait_for_connection()

    config_modal = page.locator("#config-modal")
    page.locator("#config-btn").click()
    expect(config_modal).not_to_have_class(re.compile(r"hidden"))

    # Click on modal overlay (outside the modal content)
    config_modal.click(position={"x": 10, "y": 10})

    expect(config_modal).to_have_class(re.compile(r"hidden"))

  def test_save_sends_config_action(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Saving config should send config action via WebSocket."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])
    mock_ws.set_auto_response(
        "config", [{"type": "status", "message": "Configuration updated"}]
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()
    page.wait_for_timeout(200)

    # Open settings
    page.locator("#config-btn").click()

    # Modify values
    page.locator("#config-title").fill("New Title")
    page.locator("#config-model").fill("gemini-3-flash-preview")
    page.locator("#config-iterations").fill("50")

    # Clear received messages
    page.evaluate("() => window._mockWsReceived = []")

    # Submit form
    page.locator("#config-form button[type='submit']").click()

    page.wait_for_timeout(100)

    received = mock_ws.get_received_messages()
    config_msgs = [m for m in received if m.get("action") == "config"]

    assert len(config_msgs) == 1
    assert config_msgs[0]["title"] == "New Title"
    assert config_msgs[0]["model"] == "gemini-3-flash-preview"
    assert config_msgs[0]["max_iterations"] == 50

  def test_save_closes_modal(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Saving should close the modal."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])
    mock_ws.set_auto_response(
        "config", [{"type": "status", "message": "Configuration updated"}]
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()

    config_modal = page.locator("#config-modal")
    page.locator("#config-btn").click()
    expect(config_modal).not_to_have_class(re.compile(r"hidden"))

    # Submit form
    page.locator("#config-form button[type='submit']").click()

    expect(config_modal).to_have_class(re.compile(r"hidden"))

  def test_clear_session_button(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Clear session button should clear conversation with confirmation."""
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
                ]
            )
        ],
    )
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])
    mock_ws.set_auto_response(
        "clear", [{"type": "status", "message": "Session cleared"}]
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()
    page.wait_for_timeout(200)

    # Should have message initially
    messages = page.locator(".message")
    expect(messages).to_have_count(1)

    # Set up dialog handler to accept confirmation
    page.on("dialog", lambda dialog: dialog.accept())

    # Open settings and click clear
    page.locator("#config-btn").click()
    page.locator("#config-clear").click()

    page.wait_for_timeout(200)

    # Messages should be cleared
    empty_state = page.locator("#empty-state")
    expect(empty_state).to_be_visible()

  def test_clear_cancelled_no_action(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Cancelling clear confirmation should not clear."""
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
                ]
            )
        ],
    )
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)
    mock_ws.wait_for_connection()
    page.wait_for_timeout(200)

    # Set up dialog handler to dismiss confirmation
    page.on("dialog", lambda dialog: dialog.dismiss())

    # Clear received messages
    page.evaluate("() => window._mockWsReceived = []")

    # Open settings and click clear
    page.locator("#config-btn").click()
    page.locator("#config-clear").click()

    page.wait_for_timeout(100)

    # Should NOT have sent clear action
    received = mock_ws.get_received_messages()
    clear_msgs = [m for m in received if m.get("action") == "clear"]
    assert len(clear_msgs) == 0

    # Message should still exist
    messages = page.locator(".message")
    expect(messages).to_have_count(1)


class TestFilesInSettings:
  """Tests for file handling in settings modal."""

  def test_files_input_present(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Files input field should be present in settings."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)
    mock_ws.wait_for_connection()

    page.locator("#config-btn").click()

    files_input = page.locator("#config-files")
    expect(files_input).to_be_visible()

  def test_adding_files_sends_action(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Adding files should send add_files action."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])
    mock_ws.set_auto_response(
        "config", [{"type": "status", "message": "Configuration updated"}]
    )
    mock_ws.set_auto_response(
        "add_files",
        [{
            "type": "files_added",
            "patterns": ["./docs/*.md"],
            "count": 5,
            "names": [
                "file1.md",
                "file2.md",
                "file3.md",
                "file4.md",
                "file5.md",
            ],
            "total": 5,
        }],
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()
    page.wait_for_timeout(200)

    page.locator("#config-btn").click()
    page.locator("#config-files").fill("./docs/*.md ./data/*.csv")

    # Clear received messages
    page.evaluate("() => window._mockWsReceived = []")

    # Submit form
    page.locator("#config-form button[type='submit']").click()

    page.wait_for_timeout(100)

    received = mock_ws.get_received_messages()
    add_files_msgs = [m for m in received if m.get("action") == "add_files"]

    assert len(add_files_msgs) == 1
    assert "./docs/*.md" in add_files_msgs[0]["patterns"]
    assert "./data/*.csv" in add_files_msgs[0]["patterns"]


class TestFilesDisplay:
  """Tests for files display section."""

  def test_files_section_hidden_when_empty(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Files section should be hidden when no files loaded."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session(files=[])])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)
    mock_ws.wait_for_connection()

    files_section = page.locator("#files-section")
    expect(files_section).to_have_class(re.compile(r"hidden"))

  def test_files_section_visible_with_files(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Files section should be visible when files are loaded."""
    mock_ws.setup()
    mock_ws.set_auto_response(
        "get_status", [create_mock_session(files=["./docs/*.md"])]
    )
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)
    mock_ws.wait_for_connection()
    page.wait_for_timeout(200)

    files_section = page.locator("#files-section")
    expect(files_section).not_to_have_class(re.compile(r"hidden"))

  def test_file_chips_displayed(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """File chips should be displayed for each file."""
    mock_ws.setup()
    mock_ws.set_auto_response(
        "get_status", [create_mock_session(files=["file1.md", "file2.md"])]
    )
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)
    mock_ws.wait_for_connection()
    page.wait_for_timeout(200)

    file_chips = page.locator(".file-chip")
    expect(file_chips).to_have_count(2)
    expect(file_chips.first).to_contain_text("file1.md")


class TestEventDetailModal:
  """Tests for the event detail modal."""

  def test_modal_shows_event_type(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Event modal should show the event type."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])
    mock_ws.set_auto_response(
        "query",
        [
            {"type": "query_start", "prompt": "Test"},
            create_mock_event("rlm.run.start", iteration=0, event_id=0),
        ],
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()

    prompt_input = page.locator("#prompt-input")
    prompt_input.fill("Test")
    prompt_input.press("Enter")

    page.wait_for_timeout(200)

    # Click on event
    page.locator(".event-item").first.click()

    # Modal should show event type
    modal_body = page.locator("#modal-body")
    expect(modal_body).to_contain_text("rlm.run.start")

  def test_modal_shows_timestamp(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Event modal should show the timestamp."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])
    mock_ws.set_auto_response(
        "query",
        [
            {"type": "query_start", "prompt": "Test"},
            create_mock_event("rlm.run.start", iteration=0, event_id=0),
        ],
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()

    prompt_input = page.locator("#prompt-input")
    prompt_input.fill("Test")
    prompt_input.press("Enter")

    page.wait_for_timeout(200)

    # Click on event
    page.locator(".event-item").first.click()

    # Modal should show timestamp section
    modal_body = page.locator("#modal-body")
    expect(modal_body).to_contain_text("Timestamp")

  def test_modal_shows_code_block(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Event modal should show code block for code events."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])
    mock_ws.set_auto_response(
        "query",
        [
            {"type": "query_start", "prompt": "Test"},
            create_mock_event(
                "rlm.code.found",
                iteration=1,
                event_id=0,
                code="result = 2 + 2\nprint(result)",
            ),
        ],
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()

    prompt_input = page.locator("#prompt-input")
    prompt_input.fill("Test")
    prompt_input.press("Enter")

    page.wait_for_timeout(200)

    # Click on event
    page.locator(".event-item").first.click()

    # Modal should show code
    modal_body = page.locator("#modal-body")
    expect(modal_body).to_contain_text("Code")
    expect(modal_body).to_contain_text("result = 2 + 2")

  def test_modal_shows_output(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Event modal should show output for execution events."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])
    mock_ws.set_auto_response(
        "query",
        [
            {"type": "query_start", "prompt": "Test"},
            create_mock_event(
                "rlm.code.end",
                iteration=1,
                event_id=0,
                output="4\n",
            ),
        ],
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()

    prompt_input = page.locator("#prompt-input")
    prompt_input.fill("Test")
    prompt_input.press("Enter")

    page.wait_for_timeout(200)

    # Click on event
    page.locator(".event-item").first.click()

    # Modal should show output
    modal_body = page.locator("#modal-body")
    expect(modal_body).to_contain_text("Output")

  def test_modal_close_button(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Close button should close the event modal."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])
    mock_ws.set_auto_response(
        "query",
        [
            {"type": "query_start", "prompt": "Test"},
            create_mock_event("rlm.run.start", iteration=0, event_id=0),
        ],
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()

    prompt_input = page.locator("#prompt-input")
    prompt_input.fill("Test")
    prompt_input.press("Enter")

    page.wait_for_timeout(200)

    # Click on event to open modal
    page.locator(".event-item").first.click()

    event_modal = page.locator("#event-modal")
    expect(event_modal).not_to_have_class(re.compile(r"hidden"))

    # Click close button
    page.locator("#modal-close").click()

    expect(event_modal).to_have_class(re.compile(r"hidden"))

  def test_click_outside_closes_modal(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Clicking outside event modal should close it."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])
    mock_ws.set_auto_response(
        "query",
        [
            {"type": "query_start", "prompt": "Test"},
            create_mock_event("rlm.run.start", iteration=0, event_id=0),
        ],
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()

    prompt_input = page.locator("#prompt-input")
    prompt_input.fill("Test")
    prompt_input.press("Enter")

    page.wait_for_timeout(200)

    # Click on event to open modal
    page.locator(".event-item").first.click()

    event_modal = page.locator("#event-modal")
    expect(event_modal).not_to_have_class(re.compile(r"hidden"))

    # Click on modal overlay (outside the modal content)
    event_modal.click(position={"x": 10, "y": 10})

    expect(event_modal).to_have_class(re.compile(r"hidden"))
