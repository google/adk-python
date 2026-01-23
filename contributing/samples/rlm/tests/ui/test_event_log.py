"""
UI tests for the event log panel.

These tests verify event log display, toggling, and event item interactions.
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


class TestEventLogPanel:
  """Tests for the event log panel layout and toggle."""

  def test_event_log_visible_by_default(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Event log panel should be visible by default."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)

    event_log = page.locator("#event-log-panel")
    expect(event_log).to_be_visible()
    expect(event_log).not_to_have_class(re.compile(r"collapsed"))

  def test_event_log_has_title(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Event log should display 'Event Log' title."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)

    title = page.locator(".event-log-title")
    expect(title).to_have_text("Event Log")

  def test_event_count_shows_zero_initially(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Event count should show '0 events' initially."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session(events=[])])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)
    mock_ws.wait_for_connection()

    event_count = page.locator("#event-count")
    expect(event_count).to_have_text("0 events")

  def test_toggle_collapses_panel(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Clicking toggle button should collapse the event log."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)

    event_log = page.locator("#event-log-panel")
    toggle_btn = page.locator("#toggle-log-btn")

    # Should be expanded initially
    expect(event_log).not_to_have_class(re.compile(r"collapsed"))

    # Click toggle
    toggle_btn.click()

    # Should be collapsed
    expect(event_log).to_have_class(re.compile(r"collapsed"))

  def test_toggle_expands_panel(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Clicking toggle button again should expand the event log."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)

    event_log = page.locator("#event-log-panel")
    toggle_btn = page.locator("#toggle-log-btn")

    # Collapse first
    toggle_btn.click()
    expect(event_log).to_have_class(re.compile(r"collapsed"))

    # Click again to expand
    toggle_btn.click()
    expect(event_log).not_to_have_class(re.compile(r"collapsed"))

  def test_empty_state_shown_when_no_events(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Empty state should be shown when no events exist."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session(events=[])])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    page.goto(live_server)
    mock_ws.wait_for_connection()

    empty_state = page.locator("#event-log-content .empty-state")
    expect(empty_state).to_be_visible()
    expect(empty_state).to_contain_text("Events will appear here")


class TestEventDisplay:
  """Tests for individual event display."""

  def test_events_displayed_during_query(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Events should be displayed as they arrive during query."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])
    mock_ws.set_auto_response(
        "query",
        [
            {"type": "query_start", "prompt": "Test"},
            create_mock_event("rlm.run.start", iteration=0, event_id=0),
            create_mock_event("rlm.iteration.start", iteration=1, event_id=1),
            create_mock_event("rlm.llm.start", iteration=1, event_id=2),
            create_mock_event("rlm.llm.end", iteration=1, event_id=3),
        ],
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()

    prompt_input = page.locator("#prompt-input")
    prompt_input.fill("Test")
    prompt_input.press("Enter")

    page.wait_for_timeout(300)

    # Event count should be updated
    event_count = page.locator("#event-count")
    expect(event_count).to_contain_text("4 events")

  def test_event_items_have_icon(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Event items should display icons."""
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

    # Event items should have icons
    event_icons = page.locator(".event-icon")
    expect(event_icons.first).to_be_visible()

  def test_event_items_have_label(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Event items should display labels."""
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

    # Event items should have labels
    event_labels = page.locator(".event-label")
    expect(event_labels.first).to_be_visible()

  def test_event_items_have_timestamp(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Event items should display timestamp."""
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

    # Event items should have timestamps
    event_times = page.locator(".event-time")
    expect(event_times.first).to_be_visible()
    # Should contain seconds notation
    expect(event_times.first).to_contain_text("s")

  def test_event_with_preview(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Events with content should show preview."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])
    mock_ws.set_auto_response(
        "query",
        [
            {"type": "query_start", "prompt": "Test"},
            create_mock_event(
                "rlm.llm.end",
                iteration=1,
                event_id=0,
                response_preview="This is a preview of the response...",
            ),
        ],
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()

    prompt_input = page.locator("#prompt-input")
    prompt_input.fill("Test")
    prompt_input.press("Enter")

    page.wait_for_timeout(200)

    # Should show preview
    preview = page.locator(".event-preview")
    expect(preview).to_be_visible()
    expect(preview).to_contain_text("This is a preview")


class TestAgentGroups:
  """Tests for agent group display in event log."""

  def test_agent_groups_created(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Agent groups should be created for events."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])
    mock_ws.set_auto_response(
        "query",
        [
            {"type": "query_start", "prompt": "Test"},
            {
                **create_mock_event(
                    "rlm.iteration.start", iteration=1, event_id=0
                ),
                "metadata": {"agent_name": "rlm_agent", "agent_depth": 0},
            },
            {
                **create_mock_event("rlm.llm.start", iteration=1, event_id=1),
                "metadata": {"agent_name": "rlm_agent", "agent_depth": 0},
            },
        ],
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()

    prompt_input = page.locator("#prompt-input")
    prompt_input.fill("Test")
    prompt_input.press("Enter")

    page.wait_for_timeout(300)

    # Agent group should be created
    agent_group = page.locator(".agent-group")
    expect(agent_group.first).to_be_visible()

  def test_agent_group_expandable(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Agent groups should be expandable/collapsible."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])
    mock_ws.set_auto_response(
        "query",
        [
            {"type": "query_start", "prompt": "Test"},
            {
                **create_mock_event(
                    "rlm.iteration.start", iteration=1, event_id=0
                ),
                "metadata": {"agent_name": "rlm_agent", "agent_depth": 0},
            },
        ],
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()

    prompt_input = page.locator("#prompt-input")
    prompt_input.fill("Test")
    prompt_input.press("Enter")

    page.wait_for_timeout(300)

    agent_group = page.locator(".agent-group").first
    agent_header = agent_group.locator(".agent-header")

    # Should be expanded by default
    expect(agent_group).to_have_class(re.compile(r"expanded"))

    # Click to collapse
    agent_header.click()

    # Should be collapsed
    expect(agent_group).not_to_have_class(re.compile(r"expanded"))


class TestIterationGroups:
  """Tests for iteration group display within agents."""

  def test_iteration_groups_created(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Iteration groups should be created within agents."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])
    mock_ws.set_auto_response(
        "query",
        [
            {"type": "query_start", "prompt": "Test"},
            {
                **create_mock_event(
                    "rlm.iteration.start", iteration=1, event_id=0
                ),
                "metadata": {
                    "agent_name": "rlm_agent",
                    "agent_depth": 0,
                    "iteration": 1,
                },
            },
            {
                **create_mock_event("rlm.llm.start", iteration=1, event_id=1),
                "metadata": {
                    "agent_name": "rlm_agent",
                    "agent_depth": 0,
                    "iteration": 1,
                },
            },
        ],
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()

    prompt_input = page.locator("#prompt-input")
    prompt_input.fill("Test")
    prompt_input.press("Enter")

    page.wait_for_timeout(300)

    # Iteration group should be created
    iteration_group = page.locator(".agent-iteration")
    expect(iteration_group.first).to_be_visible()

  def test_multiple_iterations_displayed(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Multiple iterations should be displayed separately."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])
    mock_ws.set_auto_response(
        "query",
        [
            {"type": "query_start", "prompt": "Test"},
            {
                **create_mock_event(
                    "rlm.iteration.start", iteration=1, event_id=0
                ),
                "metadata": {
                    "agent_name": "rlm_agent",
                    "agent_depth": 0,
                    "iteration": 1,
                },
            },
            {
                **create_mock_event(
                    "rlm.iteration.end", iteration=1, event_id=1
                ),
                "metadata": {
                    "agent_name": "rlm_agent",
                    "agent_depth": 0,
                    "iteration": 1,
                },
            },
            {
                **create_mock_event(
                    "rlm.iteration.start", iteration=2, event_id=2
                ),
                "metadata": {
                    "agent_name": "rlm_agent",
                    "agent_depth": 0,
                    "iteration": 2,
                },
            },
            {
                **create_mock_event(
                    "rlm.iteration.end", iteration=2, event_id=3
                ),
                "metadata": {
                    "agent_name": "rlm_agent",
                    "agent_depth": 0,
                    "iteration": 2,
                },
            },
        ],
    )

    page.goto(live_server)
    mock_ws.wait_for_connection()

    prompt_input = page.locator("#prompt-input")
    prompt_input.fill("Test")
    prompt_input.press("Enter")

    page.wait_for_timeout(300)

    # Should have 2 iteration groups
    iteration_groups = page.locator(".agent-iteration")
    expect(iteration_groups).to_have_count(2)


class TestEventClick:
  """Tests for clicking on event items."""

  def test_event_click_opens_modal(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Clicking an event should open the detail modal."""
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

    # Click on event item
    event_item = page.locator(".event-item").first
    event_item.click()

    # Modal should be visible
    modal = page.locator("#event-modal")
    expect(modal).not_to_have_class(re.compile(r"hidden"))


class TestEventLogScroll:
  """Tests for event log scrolling behavior."""

  def test_event_log_scrolls_to_bottom(
      self, page: Page, mock_ws: WebSocketInterceptor, live_server: str
  ):
    """Event log should auto-scroll to bottom on new events."""
    mock_ws.setup()
    mock_ws.set_auto_response("get_status", [create_mock_session()])
    mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

    # Create many events to trigger scroll
    events = [{"type": "query_start", "prompt": "Test"}]
    for i in range(20):
      events.append({
          **create_mock_event("rlm.llm.end", iteration=1, event_id=i),
          "metadata": {
              "agent_name": "rlm_agent",
              "agent_depth": 0,
              "iteration": 1,
              "response_preview": f"Response {i}",
          },
      })

    mock_ws.set_auto_response("query", events)

    page.goto(live_server)
    mock_ws.wait_for_connection()

    prompt_input = page.locator("#prompt-input")
    prompt_input.fill("Test")
    prompt_input.press("Enter")

    page.wait_for_timeout(500)

    # Check that event log content is scrolled
    event_log_content = page.locator("#event-log-content")
    scroll_height = event_log_content.evaluate("el => el.scrollHeight")
    scroll_top = event_log_content.evaluate("el => el.scrollTop")
    client_height = event_log_content.evaluate("el => el.clientHeight")

    # Should be scrolled near bottom (allow some tolerance)
    assert scroll_top + client_height >= scroll_height - 100
