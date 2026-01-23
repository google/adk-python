"""
UI tests for page load and WebSocket connection.

These tests verify the initial page load behavior and WebSocket connection handling.
"""

import re

from playwright.sync_api import expect
from playwright.sync_api import Page
import pytest

from .conftest import create_mock_session
from .conftest import create_mock_sessions_list
from .conftest import WebSocketInterceptor

pytestmark = pytest.mark.ui


class TestPageLoad:
  """Tests for initial page load."""

  def test_page_loads_with_correct_title(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Page should load with 'ADK-RLM' title."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)

    expect(page).to_have_title("ADK-RLM")

  def test_header_displays_logo(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Header should display logo icon and 'ADK-RLM' text."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)

    logo_icon = page.locator(".logo-icon")
    logo_text = page.locator(".logo-text")

    expect(logo_icon).to_be_visible()
    expect(logo_text).to_have_text("ADK-RLM")

  def test_empty_state_shown_initially(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Empty state message should be shown when no conversation exists."""
    mock_ws.setup()
    mock_ws.set_auto_response(
        "get_status", [create_mock_session(conversation=[])]
    )
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)
    mock_ws.wait_for_connection()

    empty_state = page.locator("#empty-state")
    expect(empty_state).to_be_visible()
    expect(empty_state).to_contain_text("Recursive Language Model")

  def test_session_sidebar_visible(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Session sidebar should be visible with 'Sessions' title."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)

    sidebar = page.locator("#session-sidebar")
    sidebar_title = page.locator(".sidebar-title")

    expect(sidebar).to_be_visible()
    expect(sidebar_title).to_have_text("Sessions")

  def test_settings_button_visible(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Settings button should be visible in header."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)

    settings_btn = page.locator("#config-btn")
    expect(settings_btn).to_be_visible()
    expect(settings_btn).to_contain_text("Settings")

  def test_input_area_visible(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Input textarea and send button should be visible."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)

    input_area = page.locator("#prompt-input")
    send_btn = page.locator("#send-btn")

    expect(input_area).to_be_visible()
    expect(send_btn).to_be_visible()

  def test_event_log_panel_visible(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Event log panel should be visible by default."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)

    event_log = page.locator("#event-log-panel")
    event_log_title = page.locator(".event-log-title")

    expect(event_log).to_be_visible()
    expect(event_log_title).to_have_text("Event Log")


class TestWebSocketConnection:
  """Tests for WebSocket connection behavior."""

  @pytest.mark.skip(
      reason="Mock WebSocket connects too fast to test transient state"
  )
  def test_status_shows_connecting_initially(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Status badge should show 'Connecting...' initially."""
    mock_ws.setup()
    # Don't set auto-responses yet to test initial state

    page.goto(live_server)

    status_badge = page.locator("#status-badge")
    # Initially should show Connecting...
    expect(status_badge).to_have_text("Connecting...")

  def test_status_shows_connected_after_websocket_connects(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Status badge should show 'Connected' after WebSocket connects."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)
    mock_ws.wait_for_connection()

    status_badge = page.locator("#status-badge")
    expect(status_badge).to_have_text("Connected")
    expect(status_badge).to_have_class(re.compile(r"connected"))

  @pytest.mark.skip(
      reason="Mock WebSocket connects too fast to test disconnected state"
  )
  def test_send_button_disabled_when_disconnected(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Send button should be disabled when not connected."""
    # Don't set up mock_ws to simulate no connection
    page.goto(live_server)

    # Wait briefly for initial state
    page.wait_for_timeout(100)

    send_btn = page.locator("#send-btn")
    # Button should be disabled initially before connection
    expect(send_btn).to_be_disabled()

  def test_send_button_enabled_when_connected(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Send button should be enabled when connected."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)
    mock_ws.wait_for_connection()

    send_btn = page.locator("#send-btn")
    expect(send_btn).to_be_enabled()

  def test_initial_get_status_sent(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Client should send get_status action on connection."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)
    mock_ws.wait_for_connection()

    # Wait for messages to be sent
    page.wait_for_timeout(200)

    received = mock_ws.get_received_messages()
    actions = [msg.get("action") for msg in received]

    assert "get_status" in actions

  def test_initial_list_sessions_sent(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Client should send list_sessions action on connection."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)
    mock_ws.wait_for_connection()

    # Wait for messages to be sent
    page.wait_for_timeout(200)

    received = mock_ws.get_received_messages()
    actions = [msg.get("action") for msg in received]

    assert "list_sessions" in actions

  def test_session_title_populated_from_status_response(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Session title should be populated from status_response."""
    mock_ws.setup()
    mock_ws.set_auto_response(
        "get_status", [create_mock_session(title="My Test Session")]
    )
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)
    mock_ws.wait_for_connection()

    # Wait for status response to be processed
    page.wait_for_timeout(200)

    session_title = page.locator("#session-title")
    expect(session_title).to_have_text("My Test Session")


class TestSessionListPopulation:
  """Tests for session list population from WebSocket."""

  def test_sessions_list_populated(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Session list should be populated from sessions_list response."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response(
        "list_sessions",
        [
            create_mock_sessions_list(
                sessions=[
                    {
                        "session_id": "sess-1",
                        "title": "Session One",
                        "updated_at": "2024-01-15T10:00:00",
                        "message_count": 3,
                    },
                    {
                        "session_id": "sess-2",
                        "title": "Session Two",
                        "updated_at": "2024-01-14T09:00:00",
                        "message_count": 7,
                    },
                ]
            )
        ],
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()

    # Wait for sessions list to be rendered
    page.wait_for_timeout(300)

    session_items = page.locator(".session-item")
    expect(session_items).to_have_count(2)

    first_session = session_items.first
    expect(first_session).to_contain_text("Session One")

  def test_empty_sessions_message_when_no_sessions(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Should show 'No sessions yet' when session list is empty."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response(
        "list_sessions", [create_mock_sessions_list(sessions=[])]
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()

    # Wait for sessions list to be rendered
    page.wait_for_timeout(300)

    empty_message = page.locator(".empty-sessions")
    expect(empty_message).to_have_text("No sessions yet")
