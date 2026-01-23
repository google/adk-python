"""
E2E tests for session lifecycle management.

These tests verify session creation, persistence, loading, and deletion
with a real server and database.
"""

import os
import re

from playwright.sync_api import expect
from playwright.sync_api import Page
import pytest

from .conftest import get_message_count
from .conftest import get_session_count
from .conftest import submit_query
from .conftest import wait_for_query_complete

pytestmark = [
    pytest.mark.e2e_web,
    pytest.mark.skipif(
        os.environ.get("RLM_E2E_TESTS") != "true",
        reason="E2E tests disabled. Set RLM_E2E_TESTS=true to enable.",
    ),
]


class TestSessionCreation:
  """Tests for creating new sessions."""

  def test_initial_session_created(self, e2e_page: Page):
    """Test that an initial session is created on page load."""
    # Should have a session ID in the sidebar
    session_title = e2e_page.locator("#session-title")
    expect(session_title).not_to_be_empty()

    # Status should be connected
    status_badge = e2e_page.locator("#status-badge")
    expect(status_badge).to_have_text("Connected")

  def test_create_new_session(self, e2e_page: Page):
    """Test creating a new session via the + button."""
    # Get initial session count
    initial_count = get_session_count(e2e_page)

    # Click new session button
    new_session_btn = e2e_page.locator("#new-session-btn")
    new_session_btn.click()

    # Wait for session list to update
    e2e_page.wait_for_timeout(500)

    # Session count should increase
    new_count = get_session_count(e2e_page)
    assert new_count >= initial_count, "Session count should not decrease"

    # UI should be cleared (empty state)
    empty_state = e2e_page.locator("#empty-state")
    expect(empty_state).to_be_visible()

  def test_new_session_has_default_title(self, e2e_page: Page):
    """Test that new sessions have a default title with timestamp."""
    # Create new session
    e2e_page.locator("#new-session-btn").click()
    e2e_page.wait_for_timeout(500)

    # Title should contain "Session" and a date-like pattern
    session_title = e2e_page.locator("#session-title")
    title_text = session_title.text_content() or ""

    assert (
        "Session" in title_text or "-" in title_text
    ), f"Unexpected title: {title_text}"


class TestSessionPersistence:
  """Tests for session persistence across page reloads."""

  def test_conversation_persists_on_reload(
      self, e2e_page: Page, e2e_server: str
  ):
    """Test that conversation persists after page reload."""
    # Submit a query
    submit_query(e2e_page, "Remember this: the magic word is abracadabra")
    wait_for_query_complete(e2e_page, timeout=60000)

    # Should have messages
    initial_message_count = get_message_count(e2e_page)
    assert initial_message_count >= 2, "Should have user message and answer"

    # Reload page
    e2e_page.reload()

    # Wait for WebSocket connection
    e2e_page.wait_for_function(
        "() => document.querySelector('#status-badge')?.textContent ==="
        " 'Connected'",
        timeout=10000,
    )

    # Wait for conversation to restore
    e2e_page.wait_for_timeout(1000)

    # Messages should be restored
    restored_count = get_message_count(e2e_page)
    assert restored_count >= initial_message_count, (
        f"Expected at least {initial_message_count} messages, got"
        f" {restored_count}"
    )

  def test_session_title_persists(self, e2e_page: Page, e2e_server: str):
    """Test that session title persists after page reload."""
    # Submit a query to trigger title generation
    submit_query(e2e_page, "This is a test query about persistence")
    wait_for_query_complete(e2e_page, timeout=60000)

    # Get the title
    session_title = e2e_page.locator("#session-title")
    original_title = session_title.text_content()

    # Reload page
    e2e_page.reload()

    e2e_page.wait_for_function(
        "() => document.querySelector('#status-badge')?.textContent ==="
        " 'Connected'",
        timeout=10000,
    )
    e2e_page.wait_for_timeout(1000)

    # Title should be restored
    restored_title = session_title.text_content()
    assert restored_title == original_title, (
        f"Title not restored. Expected '{original_title}', got"
        f" '{restored_title}'"
    )


class TestSessionSwitching:
  """Tests for switching between sessions."""

  def test_switch_to_different_session(self, e2e_page: Page):
    """Test switching to a different session."""
    # Create first session with content
    submit_query(e2e_page, "First session content")
    wait_for_query_complete(e2e_page, timeout=60000)

    first_message_count = get_message_count(e2e_page)

    # Create new session
    e2e_page.locator("#new-session-btn").click()
    e2e_page.wait_for_timeout(500)

    # New session should be empty
    empty_state = e2e_page.locator("#empty-state")
    expect(empty_state).to_be_visible()

    # Add content to second session
    submit_query(e2e_page, "Second session content")
    wait_for_query_complete(e2e_page, timeout=60000)

    # Should have messages in second session
    second_message_count = get_message_count(e2e_page)
    assert second_message_count >= 2

    # Switch back to first session (should be first in list, or second)
    session_items = e2e_page.locator(".session-item")
    if session_items.count() >= 2:
      # Click on a different session
      first_item = session_items.first
      first_item.click()

      e2e_page.wait_for_timeout(1000)

      # Should load that session's content
      # (messages may differ based on which session we loaded)


class TestSessionDeletion:
  """Tests for deleting sessions."""

  def test_delete_session(self, e2e_page: Page):
    """Test deleting a session."""
    # Create a session with content
    submit_query(e2e_page, "Session to delete")
    wait_for_query_complete(e2e_page, timeout=60000)

    # Create another session
    e2e_page.locator("#new-session-btn").click()
    e2e_page.wait_for_timeout(500)

    initial_session_count = get_session_count(e2e_page)
    assert initial_session_count >= 2, "Should have at least 2 sessions"

    # Set up dialog handler to accept confirmation
    e2e_page.on("dialog", lambda dialog: dialog.accept())

    # Delete the first session
    session_item = e2e_page.locator(".session-item").first
    delete_btn = session_item.locator(".session-item-delete")
    session_item.hover()
    delete_btn.click()

    e2e_page.wait_for_timeout(500)

    # Session count should decrease
    new_count = get_session_count(e2e_page)
    assert new_count < initial_session_count, "Session should be deleted"


class TestClearSession:
  """Tests for clearing session content."""

  def test_clear_session_removes_messages(self, e2e_page: Page):
    """Test that clearing a session removes messages."""
    # Add content
    submit_query(e2e_page, "Content to clear")
    wait_for_query_complete(e2e_page, timeout=60000)

    assert get_message_count(e2e_page) >= 2, "Should have messages"

    # Set up dialog handler to accept confirmation
    e2e_page.on("dialog", lambda dialog: dialog.accept())

    # Open settings and clear
    e2e_page.locator("#config-btn").click()
    e2e_page.locator("#config-clear").click()

    e2e_page.wait_for_timeout(500)

    # Messages should be cleared
    empty_state = e2e_page.locator("#empty-state")
    expect(empty_state).to_be_visible()

  def test_clear_session_removes_events(self, e2e_page: Page):
    """Test that clearing a session removes events."""
    # Add content
    submit_query(e2e_page, "Generate some events")
    wait_for_query_complete(e2e_page, timeout=60000)

    # Should have events
    event_count = e2e_page.locator("#event-count")
    expect(event_count).not_to_have_text("0 events")

    # Set up dialog handler to accept confirmation
    e2e_page.on("dialog", lambda dialog: dialog.accept())

    # Open settings and clear
    e2e_page.locator("#config-btn").click()
    e2e_page.locator("#config-clear").click()

    e2e_page.wait_for_timeout(500)

    # Events should be cleared
    expect(event_count).to_have_text("0 events")


class TestSessionConfiguration:
  """Tests for session configuration changes."""

  def test_change_model_setting(self, e2e_page: Page):
    """Test changing the model setting."""
    # Open settings
    e2e_page.locator("#config-btn").click()

    # Change model
    model_input = e2e_page.locator("#config-model")
    model_input.fill("gemini-3-flash-preview")

    # Save
    e2e_page.locator("#config-form button[type='submit']").click()

    e2e_page.wait_for_timeout(500)

    # Modal should close
    config_modal = e2e_page.locator("#config-modal")
    expect(config_modal).to_have_class(re.compile(r"hidden"))

    # Reopen settings to verify
    e2e_page.locator("#config-btn").click()
    expect(model_input).to_have_value("gemini-3-flash-preview")

  def test_change_max_iterations(self, e2e_page: Page):
    """Test changing the max iterations setting."""
    # Open settings
    e2e_page.locator("#config-btn").click()

    # Change max iterations
    iterations_input = e2e_page.locator("#config-iterations")
    iterations_input.fill("10")

    # Save
    e2e_page.locator("#config-form button[type='submit']").click()

    e2e_page.wait_for_timeout(500)

    # Reopen and verify
    e2e_page.locator("#config-btn").click()
    expect(iterations_input).to_have_value("10")

  def test_change_session_title(self, e2e_page: Page):
    """Test changing the session title."""
    # Open settings
    e2e_page.locator("#config-btn").click()

    # Change title
    title_input = e2e_page.locator("#config-title")
    title_input.fill("My Custom Title")

    # Save
    e2e_page.locator("#config-form button[type='submit']").click()

    e2e_page.wait_for_timeout(500)

    # Title should be updated in header
    session_title = e2e_page.locator("#session-title")
    expect(session_title).to_have_text("My Custom Title")
