"""
UI tests for session management.

These tests verify session creation, loading, deletion, and sidebar interactions.
"""

import re

from playwright.sync_api import expect
from playwright.sync_api import Page
import pytest

from .conftest import create_mock_session
from .conftest import create_mock_sessions_list
from .conftest import WebSocketInterceptor

pytestmark = pytest.mark.ui


class TestSessionSidebar:
  """Tests for session sidebar interactions."""

  def test_collapse_sidebar(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Sidebar should collapse when close button is clicked."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)
    mock_ws.wait_for_connection()

    sidebar = page.locator("#session-sidebar")
    close_btn = page.locator("#toggle-sidebar-close")

    # Sidebar should be visible initially
    expect(sidebar).not_to_have_class(re.compile(r"collapsed"))

    # Click close button
    close_btn.click()

    # Sidebar should be collapsed
    expect(sidebar).to_have_class(re.compile(r"collapsed"))

  def test_expand_sidebar(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Sidebar should expand when open button is clicked."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)
    mock_ws.wait_for_connection()

    sidebar = page.locator("#session-sidebar")
    close_btn = page.locator("#toggle-sidebar-close")
    open_btn = page.locator("#toggle-sidebar-open")

    # Collapse first
    close_btn.click()
    expect(sidebar).to_have_class(re.compile(r"collapsed"))

    # Click open button
    open_btn.click()

    # Sidebar should be expanded
    expect(sidebar).not_to_have_class(re.compile(r"collapsed"))


class TestNewSession:
  """Tests for creating new sessions."""

  def test_new_session_button_visible(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """New session button should be visible in sidebar."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)

    new_session_btn = page.locator("#new-session-btn")
    expect(new_session_btn).to_be_visible()

  def test_new_session_sends_action(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Clicking new session button should send new_session action."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])
    mock_ws.set_auto_response(
        "new_session",
        [{
            "type": "session_created",
            "session_id": "new-session-id",
            "title": "Session 2024-01-15 12:00",
        }],
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()

    # Clear received messages
    page.evaluate("() => window._mockWsReceived = []")

    # Click new session button
    new_session_btn = page.locator("#new-session-btn")
    new_session_btn.click()

    # Wait for message to be sent
    page.wait_for_timeout(100)

    received = mock_ws.get_received_messages()
    actions = [msg.get("action") for msg in received]

    assert "new_session" in actions

  def test_new_session_clears_ui(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Creating new session should clear the conversation UI."""
    mock_ws.setup()
    # Start with a session that has conversation
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
                ]
            )
        ],
    )
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])
    mock_ws.set_auto_response(
        "new_session",
        [{
            "type": "session_created",
            "session_id": "new-session-id",
            "title": "New Session",
        }],
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()
    page.wait_for_timeout(200)

    # Should have messages initially
    messages = page.locator(".message")
    expect(messages).to_have_count(2)

    # Click new session
    page.locator("#new-session-btn").click()
    page.wait_for_timeout(200)

    # Messages should be cleared, empty state should show
    empty_state = page.locator("#empty-state")
    expect(empty_state).to_be_visible()


class TestLoadSession:
  """Tests for loading existing sessions."""

  def test_click_session_sends_load_action(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Clicking a session item should send load_session action."""
    mock_ws.setup()
    mock_ws.set_auto_response(
        "get_status", [create_mock_session(session_id="current")]
    )
    mock_ws.set_auto_response(
        "list_sessions",
        [
            create_mock_sessions_list(
                sessions=[
                    {
                        "session_id": "other-session",
                        "title": "Other Session",
                        "updated_at": "2024-01-15T10:00:00",
                        "message_count": 3,
                    },
                ]
            )
        ],
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()
    page.wait_for_timeout(300)

    # Clear received messages
    page.evaluate("() => window._mockWsReceived = []")

    # Click the session item
    session_item = page.locator(".session-item").first
    session_item.click()

    page.wait_for_timeout(100)

    received = mock_ws.get_received_messages()

    # Should have sent load_session with the session_id
    load_msgs = [m for m in received if m.get("action") == "load_session"]
    assert len(load_msgs) == 1
    assert load_msgs[0]["session_id"] == "other-session"

  def test_session_loaded_updates_ui(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Loading a session should update the UI with session data."""
    mock_ws.setup()
    mock_ws.set_auto_response(
        "get_status",
        [
            create_mock_session(
                session_id="current",
                title="Current Session",
                conversation=[],
            )
        ],
    )
    mock_ws.set_auto_response(
        "list_sessions",
        [
            create_mock_sessions_list(
                sessions=[
                    {
                        "session_id": "other-session",
                        "title": "Loaded Session",
                        "updated_at": "2024-01-15T10:00:00",
                        "message_count": 2,
                    },
                ]
            )
        ],
    )
    mock_ws.set_auto_response(
        "load_session",
        [{
            "type": "session_loaded",
            "session_id": "other-session",
            "title": "Loaded Session",
            "model": "gemini-3-pro-preview",
            "sub_model": "gemini-3-pro-preview",
            "max_iterations": 30,
            "files": [],
            "conversation": [
                {
                    "role": "user",
                    "content": "Question?",
                    "timestamp": "2024-01-15T10:00:00",
                },
                {
                    "role": "assistant",
                    "content": "Answer!",
                    "timestamp": "2024-01-15T10:00:05",
                },
            ],
            "events": [],
        }],
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()
    page.wait_for_timeout(300)

    # Click the session item
    session_item = page.locator(".session-item").first
    session_item.click()

    page.wait_for_timeout(300)

    # Session title should be updated
    session_title = page.locator("#session-title")
    expect(session_title).to_have_text("Loaded Session")

    # Conversation should be restored
    messages = page.locator(".message")
    expect(messages).to_have_count(2)


class TestDeleteSession:
  """Tests for deleting sessions."""

  def test_delete_button_visible_on_hover(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Delete button should become visible on session item hover."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response(
        "list_sessions",
        [
            create_mock_sessions_list(
                sessions=[
                    {
                        "session_id": "test-session",
                        "title": "Test Session",
                        "updated_at": "2024-01-15T10:00:00",
                        "message_count": 1,
                    },
                ]
            )
        ],
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()
    page.wait_for_timeout(300)

    session_item = page.locator(".session-item").first
    delete_btn = session_item.locator(".session-item-delete")

    # Delete button should be hidden initially (opacity 0)
    expect(delete_btn).to_have_css("opacity", "0")

    # Hover over session item
    session_item.hover()

    # Delete button should be visible
    expect(delete_btn).to_have_css("opacity", "1")

  def test_delete_session_sends_action(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Clicking delete button should send delete_session action after confirmation."""
    mock_ws.setup()
    mock_ws.set_auto_response(
        "get_status", [create_mock_session(session_id="current")]
    )
    mock_ws.set_auto_response(
        "list_sessions",
        [
            create_mock_sessions_list(
                sessions=[
                    {
                        "session_id": "to-delete",
                        "title": "Session To Delete",
                        "updated_at": "2024-01-15T10:00:00",
                        "message_count": 1,
                    },
                ]
            )
        ],
    )
    mock_ws.set_auto_response(
        "delete_session",
        [{
            "type": "session_deleted",
            "session_id": "to-delete",
            "success": True,
        }],
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()
    page.wait_for_timeout(300)

    # Set up dialog handler to accept confirmation
    page.on("dialog", lambda dialog: dialog.accept())

    # Clear received messages
    page.evaluate("() => window._mockWsReceived = []")

    # Hover and click delete
    session_item = page.locator(".session-item").first
    delete_btn = session_item.locator(".session-item-delete")
    session_item.hover()
    delete_btn.click()

    page.wait_for_timeout(100)

    received = mock_ws.get_received_messages()

    # Should have sent delete_session
    delete_msgs = [m for m in received if m.get("action") == "delete_session"]
    assert len(delete_msgs) == 1
    assert delete_msgs[0]["session_id"] == "to-delete"

  def test_delete_cancelled_no_action(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Cancelling delete confirmation should not send action."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response(
        "list_sessions",
        [
            create_mock_sessions_list(
                sessions=[
                    {
                        "session_id": "test-session",
                        "title": "Test Session",
                        "updated_at": "2024-01-15T10:00:00",
                        "message_count": 1,
                    },
                ]
            )
        ],
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()
    page.wait_for_timeout(300)

    # Set up dialog handler to dismiss confirmation
    page.on("dialog", lambda dialog: dialog.dismiss())

    # Clear received messages
    page.evaluate("() => window._mockWsReceived = []")

    # Hover and click delete
    session_item = page.locator(".session-item").first
    delete_btn = session_item.locator(".session-item-delete")
    session_item.hover()
    delete_btn.click()

    page.wait_for_timeout(100)

    received = mock_ws.get_received_messages()

    # Should NOT have sent delete_session
    delete_msgs = [m for m in received if m.get("action") == "delete_session"]
    assert len(delete_msgs) == 0


class TestSessionActive:
  """Tests for active session highlighting."""

  def test_current_session_highlighted(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Current session should be highlighted with 'active' class."""
    mock_ws.setup()
    mock_ws.set_auto_response(
        "get_status", [create_mock_session(session_id="active-session")]
    )
    mock_ws.set_auto_response(
        "list_sessions",
        [
            create_mock_sessions_list(
                sessions=[
                    {
                        "session_id": "active-session",
                        "title": "Active Session",
                        "updated_at": "2024-01-15T10:00:00",
                        "message_count": 1,
                    },
                    {
                        "session_id": "other-session",
                        "title": "Other Session",
                        "updated_at": "2024-01-14T10:00:00",
                        "message_count": 2,
                    },
                ]
            )
        ],
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()
    page.wait_for_timeout(300)

    # Find session items
    active_item = page.locator(
        ".session-item[data-session-id='active-session']"
    )
    other_item = page.locator(".session-item[data-session-id='other-session']")

    # Active session should have 'active' class
    expect(active_item).to_have_class(re.compile(r"active"))
    expect(other_item).not_to_have_class(re.compile(r"active"))
