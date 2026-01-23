"""
Interactive CLI for ADK-RLM with real-time streaming output.

A conversational REPL that uses the ADK Runner and renders events
to the terminal using Rich. Supports slash commands for configuration.
Sessions are persisted using ADK's DatabaseSessionService.
"""

import argparse
import asyncio
from datetime import datetime
import os
from pathlib import Path
import time
import uuid

from adk_rlm import RLM
from adk_rlm import RLMEventType
from google.adk.events import Event
from google.adk.events import EventActions
from google.adk.sessions import DatabaseSessionService
from google.adk.sessions import Session
from rich.console import Console
from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.style import Style
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

# Default configuration
DEFAULT_DB_URL = os.environ.get(
    "RLM_CLI_DB_URL", "sqlite+aiosqlite:///./cli_sessions.db"
)
APP_NAME = "adk_rlm_cli"
DEFAULT_USER_ID = "default_user"

# Tokyo Night Color Theme
COLORS = {
    "primary": "#7AA2F7",
    "secondary": "#BB9AF7",
    "success": "#9ECE6A",
    "warning": "#E0AF68",
    "error": "#F7768E",
    "text": "#A9B1D6",
    "muted": "#565F89",
    "accent": "#7DCFFF",
    "border": "#3B4261",
}


class RLMDisplay:
  """Manages the Rich display for RLM execution."""

  def __init__(self, console: Console):
    self.console = console
    self.current_iteration = 0
    self.status_text = "Initializing..."
    self.last_response_preview = ""
    self.last_code = ""
    self.last_output = ""
    self.final_answer = None
    self.total_iterations = 0
    self.execution_time_ms = 0

  def reset(self):
    """Reset display state for a new query."""
    self.current_iteration = 0
    self.status_text = "Thinking..."
    self.last_response_preview = ""
    self.last_code = ""
    self.last_output = ""
    self.final_answer = None
    self.total_iterations = 0
    self.execution_time_ms = 0

  def build_display(self) -> Panel:
    """Build the current display panel."""
    content_parts = []

    # Status line
    status = Text()
    status.append("◆ ", style=Style(color=COLORS["accent"]))
    status.append(self.status_text, style=Style(color=COLORS["text"]))
    content_parts.append(status)

    # Current iteration
    if self.current_iteration > 0:
      iter_text = Text()
      iter_text.append(
          f"\nIteration {self.current_iteration}",
          style=Style(color=COLORS["primary"], bold=True),
      )
      content_parts.append(iter_text)

    # Last response preview
    if self.last_response_preview:
      response_text = Text()
      response_text.append(
          "\n\nLLM Response: ", style=Style(color=COLORS["muted"])
      )
      preview = self.last_response_preview[:300]
      if len(self.last_response_preview) > 300:
        preview += "..."
      response_text.append(preview, style=Style(color=COLORS["text"]))
      content_parts.append(response_text)

    # Last code block
    if self.last_code:
      code_header = Text("\n\nCode: ", style=Style(color=COLORS["success"]))
      content_parts.append(code_header)
      code_preview = self.last_code[:200]
      if len(self.last_code) > 200:
        code_preview += "\n..."
      content_parts.append(
          Text(code_preview, style=Style(color=COLORS["success"], dim=True))
      )

    # Last output
    if self.last_output:
      output_text = Text()
      output_text.append("\n\nOutput: ", style=Style(color=COLORS["muted"]))
      output_preview = self.last_output[:200]
      if len(self.last_output) > 200:
        output_preview += "..."
      output_text.append(output_preview, style=Style(color=COLORS["accent"]))
      content_parts.append(output_text)

    title = Text()
    title.append("◆ ", style=Style(color=COLORS["accent"]))
    title.append("RLM", style=Style(color=COLORS["primary"], bold=True))
    title.append(" ━ Processing", style=Style(color=COLORS["muted"]))

    return Panel(
        Group(*content_parts),
        title=title,
        title_align="left",
        border_style=COLORS["border"],
        padding=(1, 2),
    )

  def update_from_event(self, event) -> None:
    """Update display state from an event."""
    if not event.custom_metadata:
      return

    event_type = event.custom_metadata.get("event_type")

    if event_type == RLMEventType.RUN_START.value:
      self.status_text = "Starting..."

    elif event_type == RLMEventType.ITERATION_START.value:
      self.current_iteration = event.custom_metadata.get("iteration", 0)
      self.status_text = f"Iteration {self.current_iteration} - Thinking..."
      self.last_code = ""
      self.last_output = ""

    elif event_type == RLMEventType.LLM_CALL_START.value:
      self.status_text = f"Iteration {self.current_iteration} - Calling LLM..."

    elif event_type == RLMEventType.LLM_CALL_END.value:
      self.last_response_preview = event.custom_metadata.get(
          "response_preview", ""
      )
      self.status_text = f"Iteration {self.current_iteration} - Processing..."

    elif event_type == RLMEventType.CODE_FOUND.value:
      self.last_code = event.custom_metadata.get("code", "")
      self.status_text = f"Iteration {self.current_iteration} - Found code..."

    elif event_type == RLMEventType.CODE_EXEC_START.value:
      self.status_text = f"Iteration {self.current_iteration} - Executing..."

    elif event_type == RLMEventType.CODE_EXEC_END.value:
      output = event.custom_metadata.get("output", "")
      error = event.custom_metadata.get("error", "")
      self.last_output = output or error or "(no output)"
      self.status_text = f"Iteration {self.current_iteration} - Code executed"

    elif event_type == RLMEventType.FINAL_DETECTED.value:
      self.status_text = "Final answer detected!"

    elif event_type == RLMEventType.FINAL_ANSWER.value:
      self.final_answer = event.custom_metadata.get("answer", "")
      self.total_iterations = event.custom_metadata.get("total_iterations", 0)
      self.execution_time_ms = event.custom_metadata.get("execution_time_ms", 0)

    elif event_type == RLMEventType.RUN_END.value:
      self.status_text = "Complete!"

    elif event_type == RLMEventType.RUN_ERROR.value:
      error = event.custom_metadata.get("error", "Unknown error")
      self.status_text = f"Error: {error}"


class InteractiveCLI:
  """Interactive REPL for ADK-RLM with session persistence."""

  DEFAULT_LOG_DIR = "./logs"

  def __init__(
      self,
      model: str = "gemini-3-pro-preview",
      sub_model: str | None = None,
      max_iterations: int = 30,
      verbose: bool = False,
      log_dir: str | None = None,
      db_url: str = DEFAULT_DB_URL,
      session_id: str | None = None,
  ):
    self.console = Console()
    self.model = model
    self.sub_model = sub_model
    self.max_iterations = max_iterations
    self.verbose = verbose
    self.log_dir = log_dir or self.DEFAULT_LOG_DIR
    self.db_url = db_url

    # Session management
    self.session_service: DatabaseSessionService | None = None
    self.session: Session | None = None
    self.session_id = session_id or str(uuid.uuid4())

    # State (will be synced with session)
    self.files: list[str] = []
    self.conversation: list[dict] = []
    self.rlm: RLM | None = None
    self.display = RLMDisplay(self.console)

  def _init_rlm(self):
    """Initialize or reinitialize the RLM instance."""
    if self.rlm:
      self.rlm.close()

    self.rlm = RLM(
        model=self.model,
        sub_model=self.sub_model,
        max_iterations=self.max_iterations,
        persistent=True,  # Keep REPL state across turns
        log_dir=self.log_dir,
    )

  async def _init_session_service(self):
    """Initialize the session service."""
    if self.session_service is None:
      self.session_service = DatabaseSessionService(db_url=self.db_url)

  async def _load_or_create_session(self) -> Session:
    """Load existing session or create a new one."""
    await self._init_session_service()

    session = await self.session_service.get_session(
        app_name=APP_NAME,
        user_id=DEFAULT_USER_ID,
        session_id=self.session_id,
    )

    if session is None:
      session = await self.session_service.create_session(
          app_name=APP_NAME,
          user_id=DEFAULT_USER_ID,
          session_id=self.session_id,
          state={
              "title": f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
              "model": self.model,
              "sub_model": self.sub_model,
              "max_iterations": self.max_iterations,
              "files": [],
              "conversation": [],
          },
      )
    else:
      # Restore state from session
      self.model = session.state.get("model", self.model)
      self.sub_model = session.state.get("sub_model", self.sub_model)
      self.max_iterations = session.state.get(
          "max_iterations", self.max_iterations
      )
      self.files = session.state.get("files", [])
      self.conversation = session.state.get("conversation", [])

    self.session = session
    return session

  async def _update_session_state(self, state_updates: dict):
    """Update session state and persist to database."""
    if not self.session or not self.session_service:
      return

    # Update in-memory state
    self.session.state.update(state_updates)

    # Create a state-update event to persist changes
    event = Event(
        author="system",
        timestamp=time.time(),
        actions=EventActions(state_delta=state_updates),
    )

    await self.session_service.append_event(self.session, event)

  async def _sync_state_to_session(self):
    """Sync current state to session."""
    await self._update_session_state({
        "model": self.model,
        "sub_model": self.sub_model,
        "max_iterations": self.max_iterations,
        "files": self.files,
        "conversation": self.conversation,
    })

  def print_welcome(self):
    """Print welcome message."""
    title = Text()
    title.append("◆ ", style=Style(color=COLORS["accent"]))
    title.append("ADK-RLM", style=Style(color=COLORS["primary"], bold=True))
    title.append(" ━ Interactive Mode", style=Style(color=COLORS["muted"]))

    help_text = Text()
    help_text.append(
        "Type a message to chat, or use slash commands:\n\n",
        style=Style(color=COLORS["text"]),
    )
    help_text.append("  /files ", style=Style(color=COLORS["accent"]))
    help_text.append("<patterns>  ", style=Style(color=COLORS["muted"]))
    help_text.append(
        "Add files to context\n", style=Style(color=COLORS["text"])
    )
    help_text.append(
        "  /clear              ", style=Style(color=COLORS["accent"])
    )
    help_text.append(
        "Clear files and conversation\n", style=Style(color=COLORS["text"])
    )
    help_text.append(
        "  /status             ", style=Style(color=COLORS["accent"])
    )
    help_text.append(
        "Show current configuration\n", style=Style(color=COLORS["text"])
    )
    help_text.append("  /model ", style=Style(color=COLORS["accent"]))
    help_text.append("<name>      ", style=Style(color=COLORS["muted"]))
    help_text.append("Change model\n", style=Style(color=COLORS["text"]))
    help_text.append("  /iterations ", style=Style(color=COLORS["accent"]))
    help_text.append("<n>   ", style=Style(color=COLORS["muted"]))
    help_text.append("Set max iterations\n", style=Style(color=COLORS["text"]))
    help_text.append(
        "  /logs               ", style=Style(color=COLORS["accent"])
    )
    help_text.append("Show log file path\n", style=Style(color=COLORS["text"]))
    help_text.append("\n", style=Style(color=COLORS["text"]))
    help_text.append(
        "  /sessions           ", style=Style(color=COLORS["secondary"])
    )
    help_text.append("List saved sessions\n", style=Style(color=COLORS["text"]))
    help_text.append(
        "  /new                ", style=Style(color=COLORS["secondary"])
    )
    help_text.append("Create new session\n", style=Style(color=COLORS["text"]))
    help_text.append("  /load ", style=Style(color=COLORS["secondary"]))
    help_text.append("<id>        ", style=Style(color=COLORS["muted"]))
    help_text.append("Load a session\n", style=Style(color=COLORS["text"]))
    help_text.append("  /delete ", style=Style(color=COLORS["secondary"]))
    help_text.append("<id>      ", style=Style(color=COLORS["muted"]))
    help_text.append("Delete a session\n", style=Style(color=COLORS["text"]))
    help_text.append("  /title ", style=Style(color=COLORS["secondary"]))
    help_text.append("<name>      ", style=Style(color=COLORS["muted"]))
    help_text.append("Set session title\n", style=Style(color=COLORS["text"]))
    help_text.append("\n", style=Style(color=COLORS["text"]))
    help_text.append(
        "  /help               ", style=Style(color=COLORS["accent"])
    )
    help_text.append("Show this help\n", style=Style(color=COLORS["text"]))
    help_text.append(
        "  /quit               ", style=Style(color=COLORS["accent"])
    )
    help_text.append("Exit\n", style=Style(color=COLORS["text"]))

    panel = Panel(
        help_text,
        title=title,
        title_align="left",
        border_style=COLORS["border"],
        padding=(1, 2),
    )

    self.console.print()
    self.console.print(panel)
    self.console.print()

  def print_status(self):
    """Print current configuration status."""
    table = Table(
        show_header=False,
        show_edge=False,
        box=None,
        padding=(0, 2),
    )
    table.add_column("key", style=Style(color=COLORS["muted"]), width=16)
    table.add_column("value", style=Style(color=COLORS["text"]))

    # Session info
    session_title = (
        self.session.state.get("title", "Untitled")
        if self.session
        else "Untitled"
    )
    table.add_row(
        "Session", Text(session_title, style=Style(color=COLORS["primary"]))
    )
    table.add_row(
        "Session ID",
        Text(self.session_id[:8] + "...", style=Style(color=COLORS["muted"])),
    )
    table.add_row(
        "Messages",
        Text(
            str(len(self.conversation)), style=Style(color=COLORS["secondary"])
        ),
    )

    table.add_row(
        "Model", Text(self.model, style=Style(color=COLORS["accent"]))
    )
    table.add_row(
        "Sub-model",
        Text(
            self.sub_model or self.model, style=Style(color=COLORS["secondary"])
        ),
    )
    table.add_row(
        "Max iterations",
        Text(str(self.max_iterations), style=Style(color=COLORS["warning"])),
    )

    if self.files:
      files_str = ", ".join(self.files[:5])
      if len(self.files) > 5:
        files_str += f" (+{len(self.files) - 5} more)"
      table.add_row(
          "Files", Text(files_str, style=Style(color=COLORS["success"]))
      )
    else:
      table.add_row("Files", Text("(none)", style=Style(color=COLORS["muted"])))

    log_path = self.rlm.log_path if self.rlm else None
    if log_path:
      table.add_row(
          "Log file", Text(log_path, style=Style(color=COLORS["muted"]))
      )

    title = Text()
    title.append("◇ ", style=Style(color=COLORS["secondary"]))
    title.append("Status", style=Style(color=COLORS["secondary"]))

    panel = Panel(
        table,
        title=title,
        title_align="left",
        border_style=COLORS["muted"],
        padding=(1, 2),
    )
    self.console.print(panel)

  def print_answer(self, answer: str, iterations: int, time_ms: float):
    """Print the final answer."""
    title = Text()
    title.append("★ ", style=Style(color=COLORS["warning"]))
    title.append("Answer", style=Style(color=COLORS["warning"], bold=True))
    title.append(
        f"  ({iterations} iterations, {time_ms/1000:.1f}s)",
        style=Style(color=COLORS["muted"]),
    )

    # Try to render as markdown
    try:
      content = Markdown(answer)
    except Exception:
      content = Text(answer, style=Style(color=COLORS["text"]))

    panel = Panel(
        content,
        title=title,
        title_align="left",
        border_style=COLORS["warning"],
        padding=(1, 2),
    )

    self.console.print()
    self.console.print(panel)

  def print_error(self, message: str):
    """Print an error message."""
    self.console.print(f"[red]Error:[/red] {message}")

  def print_info(self, message: str):
    """Print an info message."""
    self.console.print(f"[{COLORS['accent']}]ℹ[/{COLORS['accent']}] {message}")

  async def handle_command(self, command: str) -> bool:
    """
    Handle a slash command.

    Returns True if the REPL should continue, False to exit.
    """
    parts = command.split(maxsplit=1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if cmd in ("/quit", "/exit", "/q"):
      self.console.print(f"[{COLORS['muted']}]Goodbye![/{COLORS['muted']}]")
      return False

    elif cmd == "/help":
      self.print_welcome()

    elif cmd == "/status":
      self.print_status()

    elif cmd == "/clear":
      self.files = []
      self.conversation = []
      self._init_rlm()
      await self._update_session_state({
          "files": [],
          "conversation": [],
      })
      self.print_info("Cleared files and conversation state.")

    elif cmd == "/files":
      if not args:
        if self.files:
          self.console.print(
              f"[{COLORS['muted']}]Current files:[/{COLORS['muted']}]"
          )
          for f in self.files:
            self.console.print(f"  • {f}")
        else:
          self.print_info("No files loaded. Use /files <pattern> to add files.")
      else:
        # Add new files
        new_files = args.split()
        try:
          resolved = self.rlm.file_loader.create_lazy_files(new_files)
          if len(resolved) == 0:
            self.print_error(f"No files found matching: {' '.join(new_files)}")
          else:
            self.files.extend(new_files)
            await self._update_session_state({"files": self.files})
            self.print_info(
                f"Added {len(resolved)} file(s). Total patterns:"
                f" {len(self.files)}"
            )
            for f in resolved.names[:5]:
              self.console.print(
                  f"  [{COLORS['success']}]✓[/{COLORS['success']}] {f}"
              )
            if len(resolved) > 5:
              self.console.print(
                  f"  [{COLORS['muted']}]... and {len(resolved) - 5}"
                  f" more[/{COLORS['muted']}]"
              )
        except Exception as e:
          self.print_error(f"Could not resolve files: {e}")

    elif cmd == "/model":
      if not args:
        self.console.print(
            "Current model:"
            f" [{COLORS['accent']}]{self.model}[/{COLORS['accent']}]"
        )
      else:
        self.model = args.strip()
        self._init_rlm()
        await self._update_session_state({"model": self.model})
        self.print_info(f"Model changed to: {self.model}")

    elif cmd == "/submodel":
      if not args:
        self.console.print(
            "Current sub-model:"
            f" [{COLORS['accent']}]{self.sub_model or self.model}[/{COLORS['accent']}]"
        )
      else:
        self.sub_model = args.strip()
        self._init_rlm()
        await self._update_session_state({"sub_model": self.sub_model})
        self.print_info(f"Sub-model changed to: {self.sub_model}")

    elif cmd == "/iterations":
      if not args:
        self.console.print(
            "Max iterations:"
            f" [{COLORS['accent']}]{self.max_iterations}[/{COLORS['accent']}]"
        )
      else:
        try:
          self.max_iterations = int(args.strip())
          self._init_rlm()
          await self._update_session_state(
              {"max_iterations": self.max_iterations}
          )
          self.print_info(f"Max iterations set to: {self.max_iterations}")
        except ValueError:
          self.print_error("Invalid number")

    elif cmd == "/logs":
      log_path = self.rlm.log_path if self.rlm else None
      if log_path:
        self.console.print(
            f"Log file: [{COLORS['accent']}]{log_path}[/{COLORS['accent']}]"
        )
      else:
        self.console.print(
            "Log directory:"
            f" [{COLORS['accent']}]{self.log_dir}[/{COLORS['accent']}]"
        )

    # Session management commands
    elif cmd == "/sessions":
      await self._list_sessions()

    elif cmd == "/new":
      await self._new_session()

    elif cmd == "/load":
      if not args:
        self.print_error("Usage: /load <session_id>")
      else:
        await self._load_session(args.strip())

    elif cmd == "/delete":
      if not args:
        self.print_error("Usage: /delete <session_id>")
      else:
        await self._delete_session(args.strip())

    elif cmd == "/title":
      if not args:
        title = (
            self.session.state.get("title", "Untitled")
            if self.session
            else "Untitled"
        )
        self.console.print(
            f"Session title: [{COLORS['primary']}]{title}[/{COLORS['primary']}]"
        )
      else:
        await self._update_session_state({"title": args.strip()})
        self.print_info(f"Session title set to: {args.strip()}")

    else:
      self.print_error(
          f"Unknown command: {cmd}. Type /help for available commands."
      )

    return True

  async def _list_sessions(self):
    """List all saved sessions."""
    await self._init_session_service()
    response = await self.session_service.list_sessions(
        app_name=APP_NAME,
        user_id=DEFAULT_USER_ID,
    )

    if not response.sessions:
      self.print_info("No saved sessions.")
      return

    table = Table(
        show_header=True,
        header_style=Style(color=COLORS["primary"], bold=True),
        border_style=COLORS["border"],
    )
    table.add_column("ID", style=Style(color=COLORS["muted"]), width=10)
    table.add_column("Title", style=Style(color=COLORS["text"]))
    table.add_column(
        "Messages", style=Style(color=COLORS["secondary"]), justify="right"
    )
    table.add_column("Updated", style=Style(color=COLORS["muted"]))

    # Sort by last update time descending
    sessions = sorted(
        response.sessions,
        key=lambda s: s.last_update_time or 0,
        reverse=True,
    )

    for s in sessions:
      conv = s.state.get("conversation", [])
      updated = (
          datetime.fromtimestamp(s.last_update_time).strftime("%Y-%m-%d %H:%M")
          if s.last_update_time
          else "Unknown"
      )
      is_current = "→ " if s.id == self.session_id else "  "
      table.add_row(
          is_current + s.id[:8],
          s.state.get("title", "Untitled"),
          str(len(conv)),
          updated,
      )

    self.console.print()
    self.console.print(table)
    self.console.print(
        f"\n[{COLORS['muted']}]Use /load <id> to switch"
        f" sessions[/{COLORS['muted']}]"
    )

  async def _new_session(self):
    """Create a new session."""
    await self._init_session_service()

    # Close current RLM
    if self.rlm:
      self.rlm.close()
      self.rlm = None

    # Create new session
    self.session_id = str(uuid.uuid4())
    self.files = []
    self.conversation = []

    self.session = await self.session_service.create_session(
        app_name=APP_NAME,
        user_id=DEFAULT_USER_ID,
        session_id=self.session_id,
        state={
            "title": f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "model": self.model,
            "sub_model": self.sub_model,
            "max_iterations": self.max_iterations,
            "files": [],
            "conversation": [],
        },
    )

    self._init_rlm()
    self.print_info(f"Created new session: {self.session_id[:8]}...")

  async def _load_session(self, session_id_prefix: str):
    """Load a session by ID or prefix."""
    await self._init_session_service()

    # Find session matching prefix
    response = await self.session_service.list_sessions(
        app_name=APP_NAME,
        user_id=DEFAULT_USER_ID,
    )

    matching = [
        s for s in response.sessions if s.id.startswith(session_id_prefix)
    ]

    if not matching:
      self.print_error(f"No session found matching: {session_id_prefix}")
      return
    if len(matching) > 1:
      self.print_error(
          f"Multiple sessions match '{session_id_prefix}'. Be more specific."
      )
      return

    # Close current RLM
    if self.rlm:
      self.rlm.close()
      self.rlm = None

    # Load the session
    target = matching[0]
    self.session_id = target.id
    self.session = target
    self.model = target.state.get("model", self.model)
    self.sub_model = target.state.get("sub_model", self.sub_model)
    self.max_iterations = target.state.get(
        "max_iterations", self.max_iterations
    )
    self.files = target.state.get("files", [])
    self.conversation = target.state.get("conversation", [])

    self._init_rlm()
    title = target.state.get("title", "Untitled")
    self.print_info(f"Loaded session: {title} ({self.session_id[:8]}...)")
    self.print_info(
        f"  {len(self.conversation)} messages, {len(self.files)} file patterns"
    )

  async def _delete_session(self, session_id_prefix: str):
    """Delete a session by ID or prefix."""
    await self._init_session_service()

    # Find session matching prefix
    response = await self.session_service.list_sessions(
        app_name=APP_NAME,
        user_id=DEFAULT_USER_ID,
    )

    matching = [
        s for s in response.sessions if s.id.startswith(session_id_prefix)
    ]

    if not matching:
      self.print_error(f"No session found matching: {session_id_prefix}")
      return
    if len(matching) > 1:
      self.print_error(
          f"Multiple sessions match '{session_id_prefix}'. Be more specific."
      )
      return

    target = matching[0]

    if target.id == self.session_id:
      self.print_error(
          "Cannot delete the current session. Switch to another session first."
      )
      return

    await self.session_service.delete_session(
        app_name=APP_NAME,
        user_id=DEFAULT_USER_ID,
        session_id=target.id,
    )
    title = target.state.get("title", "Untitled")
    self.print_info(f"Deleted session: {title} ({target.id[:8]}...)")

  async def run_query(self, prompt: str):
    """Run a query and stream the results."""
    self.display.reset()

    # Add user message to conversation
    self.conversation.append({
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().isoformat(),
    })

    # Extract conversation history for the agent (exclude current message)
    # Only include role and content, not timestamp
    conversation_history = None
    if len(self.conversation) > 1:
      conversation_history = [
          {"role": msg["role"], "content": msg["content"]}
          for msg in self.conversation[:-1]
      ]

    # Build file context
    if self.files:
      try:
        file_ctx = self.rlm.file_loader.build_context(self.files, lazy=True)
        file_count = file_ctx.get("file_count", 0)
        if file_count == 0:
          self.print_error(
              f"No files found matching patterns: {' '.join(self.files)}"
          )
          self.print_info(
              "Check that the paths exist. Use /files to see current patterns."
          )
          # Remove the user message we just added
          self.conversation.pop()
          return
        ctx = file_ctx
      except Exception as e:
        self.print_error(f"Failed to load files: {e}")
        self.conversation.pop()
        return
    else:
      ctx = {
          "info": "No files loaded. The user is asking a question.",
      }

    try:
      with Live(
          self.display.build_display(),
          console=self.console,
          refresh_per_second=10,
          transient=True,
      ) as live:
        async for event in self.rlm.run_streaming(
            ctx, prompt, conversation_history
        ):
          self.display.update_from_event(event)
          live.update(self.display.build_display())

      # Print final answer and save to conversation
      if self.display.final_answer:
        self.conversation.append({
            "role": "assistant",
            "content": self.display.final_answer,
            "timestamp": datetime.now().isoformat(),
        })

        # Auto-generate title from first exchange (like web.py)
        title = self.session.state.get("title", "") if self.session else ""
        if title.startswith("Session ") and len(self.conversation) == 2:
          first_msg = self.conversation[0]["content"]
          title = first_msg[:50] + ("..." if len(first_msg) > 50 else "")
          await self._update_session_state({"title": title})

        # Save conversation to session
        await self._update_session_state({"conversation": self.conversation})

        self.print_answer(
            self.display.final_answer,
            self.display.total_iterations,
            self.display.execution_time_ms,
        )
      else:
        # Remove user message if no answer
        self.conversation.pop()
        self.print_error("No answer received")

    except KeyboardInterrupt:
      self.conversation.pop()
      self.console.print(
          f"\n[{COLORS['warning']}]Interrupted[/{COLORS['warning']}]"
      )
    except Exception as e:
      self.conversation.pop()
      self.print_error(str(e))

  async def run(self):
    """Main REPL loop."""
    # Initialize session
    await self._load_or_create_session()
    self._init_rlm()

    self.print_welcome()

    # Handle pending files from command line
    pending_files = getattr(self, "_pending_files", None)
    if pending_files:
      try:
        resolved = self.rlm.file_loader.create_lazy_files(pending_files)
        if len(resolved) == 0:
          self.print_error(
              f"No files found matching: {' '.join(pending_files)}"
          )
          self.print_info(
              "Use /files <pattern> to add files, or check your paths"
          )
        else:
          self.files.extend(pending_files)
          await self._update_session_state({"files": self.files})
          self.print_info(
              f"Loaded {len(resolved)} file(s) from {len(pending_files)}"
              " pattern(s)"
          )
          for f in resolved.names[:5]:
            self.console.print(
                f"  [{COLORS['success']}]✓[/{COLORS['success']}] {f}"
            )
          if len(resolved) > 5:
            self.console.print(
                f"  [{COLORS['muted']}]... and {len(resolved) - 5}"
                f" more[/{COLORS['muted']}]"
            )
      except Exception as e:
        self.print_error(f"Could not load files: {e}")
      self._pending_files = None

    # Show session info on startup
    title = (
        self.session.state.get("title", "Untitled")
        if self.session
        else "Untitled"
    )
    msg_count = len(self.conversation)
    if msg_count > 0:
      self.print_info(f"Resumed session: {title} ({msg_count} messages)")
    else:
      self.print_info(f"Session: {title}")

    while True:
      try:
        # Get input
        self.console.print()
        user_input = Prompt.ask(
            f"[{COLORS['accent']}]>[/{COLORS['accent']}]",
            console=self.console,
        )

        if not user_input.strip():
          continue

        # Handle slash commands
        if user_input.startswith("/"):
          if not await self.handle_command(user_input):
            break
          continue

        # Run query
        await self.run_query(user_input)

      except KeyboardInterrupt:
        self.console.print(
            f"\n[{COLORS['muted']}]Use /quit to exit[/{COLORS['muted']}]"
        )
      except EOFError:
        break

    # Cleanup
    if self.rlm:
      self.rlm.close()


async def run_interactive(
    model: str = "gemini-3-pro-preview",
    sub_model: str | None = None,
    max_iterations: int = 30,
    files: list[str] | None = None,
    log_dir: str | None = None,
    db_url: str = DEFAULT_DB_URL,
    session_id: str | None = None,
):
  """Run the interactive CLI."""
  cli = InteractiveCLI(
      model=model,
      sub_model=sub_model,
      max_iterations=max_iterations,
      log_dir=log_dir,
      db_url=db_url,
      session_id=session_id,
  )

  # Pre-load any files specified on command line (will be added after session init)
  if files:
    cli._pending_files = files
  else:
    cli._pending_files = None

  # Run will initialize session and RLM
  await cli.run()


def main():
  """Main entry point for the CLI."""
  parser = argparse.ArgumentParser(
      description="ADK-RLM: Interactive Recursive Language Model CLI",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog="""
Examples:
  # Start interactive mode
  python -m adk_rlm.cli

  # Start with files pre-loaded
  python -m adk_rlm.cli --files "./docs/**/*.md" "./data/*.csv"

  # Use a specific model
  python -m adk_rlm.cli --model gemini-3-pro-preview

  # Resume a specific session
  python -m adk_rlm.cli --session abc12345
        """,
  )

  parser.add_argument(
      "--files",
      "-f",
      type=str,
      nargs="+",
      help="File paths or glob patterns to pre-load",
  )
  parser.add_argument(
      "--model",
      "-m",
      type=str,
      default="gemini-3-pro-preview",
      help="Main model to use (default: gemini-3-pro-preview)",
  )
  parser.add_argument(
      "--sub-model",
      "-s",
      type=str,
      help="Sub-model for recursive calls (defaults to main model)",
  )
  parser.add_argument(
      "--max-iterations",
      "-i",
      type=int,
      default=30,
      help="Maximum number of iterations (default: 30)",
  )
  parser.add_argument(
      "--log-dir",
      "-l",
      type=str,
      default="./logs",
      help="Directory for JSONL logs (default: ./logs)",
  )
  parser.add_argument(
      "--db-url",
      type=str,
      default=DEFAULT_DB_URL,
      help=(
          "SQLAlchemy database URL for sessions (default:"
          " sqlite+aiosqlite:///./cli_sessions.db)"
      ),
  )
  parser.add_argument(
      "--session",
      type=str,
      help="Session ID to resume (creates new session if not specified)",
  )

  args = parser.parse_args()

  # Run the interactive CLI
  asyncio.run(
      run_interactive(
          model=args.model,
          sub_model=args.sub_model,
          max_iterations=args.max_iterations,
          files=args.files,
          log_dir=args.log_dir,
          db_url=args.db_url,
          session_id=args.session,
      )
  )


if __name__ == "__main__":
  main()
