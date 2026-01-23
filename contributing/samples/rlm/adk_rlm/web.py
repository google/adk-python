"""
Web interface for ADK-RLM using FastAPI and Jinja2.

Provides a browser-based UI with real-time streaming events via WebSocket.
Events are displayed in an expandable log with iteration lineage.
Sessions are persisted using ADK's DatabaseSessionService.
"""

from contextlib import asynccontextmanager
from datetime import datetime
import logging
import os
from pathlib import Path
import time
from typing import Any
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("adk_rlm.web")

from adk_rlm import RLM
from adk_rlm import RLMEventType
from fastapi import FastAPI
from fastapi import Request
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from google.adk.sessions import DatabaseSessionService
from google.adk.sessions import Session

# Template directory
TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE_DIR.mkdir(exist_ok=True)

# Default configuration (can be overridden via environment or create_app)
DEFAULT_DB_URL = os.environ.get(
    "RLM_DB_URL", "sqlite+aiosqlite:///./sessions.db"
)
DEFAULT_MODEL = os.environ.get("RLM_MODEL", "gemini-3-pro-preview")
DEFAULT_SUB_MODEL = os.environ.get("RLM_SUB_MODEL")
DEFAULT_MAX_ITERATIONS = int(os.environ.get("RLM_MAX_ITERATIONS", "30"))
DEFAULT_LOG_DIR = os.environ.get("RLM_LOG_DIR", "./logs")

# Module-level config that persists across imports
_config = {
    "db_url": DEFAULT_DB_URL,
    "model": DEFAULT_MODEL,
    "sub_model": DEFAULT_SUB_MODEL,
    "max_iterations": DEFAULT_MAX_ITERATIONS,
    "log_dir": DEFAULT_LOG_DIR,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
  """Initialize session service on startup."""
  global session_service

  # Use module-level config
  db_url = _config["db_url"]

  if session_service is None:
    logger.info(f"Initializing DatabaseSessionService with: {db_url}")
    session_service = DatabaseSessionService(db_url=db_url)

  # Warm up the database connection by doing a simple query
  # This ensures the first WebSocket connection doesn't have to wait
  logger.info("Warming up database connection...")
  try:
    await session_service.list_sessions(
        app_name=APP_NAME,
        user_id=DEFAULT_USER_ID,
    )
    logger.info("Database warmup complete")
  except Exception as e:
    logger.warning(f"Database warmup failed (this is OK for first run): {e}")

  yield

  # Cleanup on shutdown
  for rlm in active_rlm.values():
    rlm.close()
  active_rlm.clear()


app = FastAPI(title="ADK-RLM Web Interface", lifespan=lifespan)

# Setup templates
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# Global session service (initialized in create_app or main)
session_service: DatabaseSessionService | None = None

# App name for ADK sessions
APP_NAME = "adk_rlm_web"
DEFAULT_USER_ID = "default_user"

# Store active RLM instances per session
active_rlm: dict[str, RLM] = {}


# Tokyo Night Color Theme (matching CLI)
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
    "bg": "#1A1B26",
    "bg_dark": "#16161E",
    "bg_highlight": "#292E42",
}


def get_or_create_rlm(session: Session) -> RLM:
  """Get or create an RLM instance for a session."""
  if session.id not in active_rlm:
    # Get config from session state
    model = session.state.get("model", "gemini-3-pro-preview")
    sub_model = session.state.get("sub_model")
    max_iterations = session.state.get("max_iterations", 30)
    log_dir = session.state.get("log_dir", "./logs")

    active_rlm[session.id] = RLM(
        model=model,
        sub_model=sub_model,
        max_iterations=max_iterations,
        persistent=True,
        log_dir=log_dir,
    )

    # Register GCS source for gs:// URIs
    try:
      from adk_rlm.files.sources.gcs import GCSFileSource

      gcs_source = GCSFileSource()
      active_rlm[session.id].file_loader.register_source("gcs", gcs_source)
    except ImportError:
      logger.warning(
          "GCS support not available (google-cloud-storage not installed)"
      )
  return active_rlm[session.id]


def close_rlm(session_id: str):
  """Close and remove an RLM instance."""
  if session_id in active_rlm:
    active_rlm[session_id].close()
    del active_rlm[session_id]


def get_event_icon(event_type: str) -> str:
  """Get icon for event type."""
  icons = {
      RLMEventType.RUN_START.value: "play_arrow",
      RLMEventType.RUN_END.value: "stop",
      RLMEventType.RUN_ERROR.value: "error",
      RLMEventType.ITERATION_START.value: "loop",
      RLMEventType.ITERATION_END.value: "check_circle",
      RLMEventType.LLM_CALL_START.value: "psychology",
      RLMEventType.LLM_CALL_END.value: "psychology",
      RLMEventType.CODE_FOUND.value: "code",
      RLMEventType.CODE_EXEC_START.value: "terminal",
      RLMEventType.CODE_EXEC_END.value: "terminal",
      RLMEventType.SUB_LLM_START.value: "call_split",
      RLMEventType.SUB_LLM_END.value: "call_merge",
      RLMEventType.FINAL_DETECTED.value: "star",
      RLMEventType.FINAL_ANSWER.value: "check",
  }
  return icons.get(event_type, "circle")


def get_event_color(event_type: str) -> str:
  """Get color for event type."""
  colors = {
      RLMEventType.RUN_START.value: COLORS["primary"],
      RLMEventType.RUN_END.value: COLORS["success"],
      RLMEventType.RUN_ERROR.value: COLORS["error"],
      RLMEventType.ITERATION_START.value: COLORS["primary"],
      RLMEventType.ITERATION_END.value: COLORS["muted"],
      RLMEventType.LLM_CALL_START.value: COLORS["secondary"],
      RLMEventType.LLM_CALL_END.value: COLORS["secondary"],
      RLMEventType.CODE_FOUND.value: COLORS["success"],
      RLMEventType.CODE_EXEC_START.value: COLORS["accent"],
      RLMEventType.CODE_EXEC_END.value: COLORS["accent"],
      RLMEventType.SUB_LLM_START.value: COLORS["warning"],
      RLMEventType.SUB_LLM_END.value: COLORS["warning"],
      RLMEventType.FINAL_DETECTED.value: COLORS["warning"],
      RLMEventType.FINAL_ANSWER.value: COLORS["warning"],
  }
  return colors.get(event_type, COLORS["text"])


def format_event_label(event_type: str) -> str:
  """Format event type for display."""
  label = event_type.replace("rlm.", "").replace(".", " ").title()
  return label


def format_event_for_ui(
    event_data: dict, event_id: int, start_time: float, iteration: int
) -> dict:
  """Format an event for the UI."""
  event_type = event_data.get("event_type", "")
  return {
      "id": event_id,
      "type": "event",
      "event_type": event_type,
      "iteration": iteration,
      "timestamp": time.time() - start_time,
      "icon": get_event_icon(event_type),
      "color": get_event_color(event_type),
      "label": format_event_label(event_type),
      "metadata": {
          k: v
          for k, v in event_data.items()
          if k not in ("event_type",) and v is not None
      },
  }


async def update_session_state(
    session: Session, state_updates: dict[str, Any]
) -> Session:
  """
  Update session state and persist to database.

  This is a convenience wrapper that updates state via a no-op event.
  """
  from google.adk.events import Event
  from google.adk.events import EventActions

  # Update in-memory state
  session.state.update(state_updates)

  # Create a state-update event to persist changes
  event = Event(
      author="system",
      timestamp=time.time(),
      actions=EventActions(state_delta=state_updates),
  )

  # Persist via append_event
  await session_service.append_event(session, event)

  return session


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
  """Render the main page."""
  return templates.TemplateResponse(
      "index.html",
      {
          "request": request,
          "colors": COLORS,
      },
  )


@app.get("/health")
async def health():
  """Health check endpoint to verify session service."""
  try:
    if session_service is None:
      return {"status": "error", "message": "session_service is None"}

    # Try a simple operation
    logger.info("Health check: testing session service...")
    sessions = await session_service.list_sessions(
        app_name=APP_NAME,
        user_id=DEFAULT_USER_ID,
    )
    logger.info(f"Health check: got {len(sessions.sessions)} sessions")
    return {
        "status": "ok",
        "session_service": str(type(session_service)),
        "session_count": len(sessions.sessions),
    }
  except Exception as e:
    logger.exception(f"Health check failed: {e}")
    return {"status": "error", "message": str(e)}


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
  """WebSocket endpoint for real-time streaming."""
  logger.info(f"WebSocket connection request for session: {session_id}")
  await websocket.accept()
  logger.info(f"WebSocket accepted for session: {session_id}")

  # Get or create session
  try:
    session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=DEFAULT_USER_ID,
        session_id=session_id,
    )
    logger.info(f"Got session: {session}")

    if session is None:
      # Create new session with default state
      logger.info(f"Creating new session: {session_id}")
      session = await session_service.create_session(
          app_name=APP_NAME,
          user_id=DEFAULT_USER_ID,
          session_id=session_id,
          state={
              "title": f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
              "model": "gemini-3-pro-preview",
              "sub_model": None,
              "max_iterations": 30,
              "files": [],
              "conversation": [],  # List of {role, content, timestamp}
              "ui_events": [],  # Formatted events for UI
          },
      )
      logger.info(f"Created session: {session.id}")
  except Exception as e:
    logger.exception(f"Error getting/creating session: {e}")
    await websocket.close(code=1011, reason=str(e))
    return

  try:
    while True:
      data = await websocket.receive_json()
      action = data.get("action")

      if action == "query":
        prompt = data.get("prompt", "")
        await run_query(websocket, session, prompt)

      elif action == "add_files":
        patterns = data.get("patterns", [])
        await add_files(websocket, session, patterns)

      elif action == "clear":
        # Clear conversation and events
        await update_session_state(
            session,
            {
                "conversation": [],
                "ui_events": [],
                "files": [],
            },
        )
        close_rlm(session.id)
        await websocket.send_json({
            "type": "status",
            "message": "Session cleared",
        })

      elif action == "config":
        updates = {}
        if "model" in data:
          updates["model"] = data["model"]
        if "sub_model" in data:
          updates["sub_model"] = data["sub_model"]
        if "max_iterations" in data:
          updates["max_iterations"] = data["max_iterations"]
        if "title" in data:
          updates["title"] = data["title"]

        if updates:
          await update_session_state(session, updates)
          # Recreate RLM with new config
          close_rlm(session.id)

        await websocket.send_json({
            "type": "status",
            "message": "Configuration updated",
        })

      elif action == "get_status":
        # Refresh session from DB
        session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=DEFAULT_USER_ID,
            session_id=session.id,
        )
        await websocket.send_json({
            "type": "status_response",
            "session_id": session.id,
            "title": session.state.get("title", "Untitled"),
            "model": session.state.get("model", "gemini-3-pro-preview"),
            "sub_model": (
                session.state.get("sub_model")
                or session.state.get("model", "gemini-3-pro-preview")
            ),
            "max_iterations": session.state.get("max_iterations", 30),
            "files": session.state.get("files", []),
            "conversation": session.state.get("conversation", []),
            "events": session.state.get("ui_events", []),
        })

      elif action == "load_session":
        new_session_id = data.get("session_id")
        if new_session_id and new_session_id != session.id:
          new_session = await session_service.get_session(
              app_name=APP_NAME,
              user_id=DEFAULT_USER_ID,
              session_id=new_session_id,
          )
          if new_session:
            session = new_session
            await websocket.send_json({
                "type": "session_loaded",
                "session_id": session.id,
                "title": session.state.get("title", "Untitled"),
                "model": session.state.get("model", "gemini-3-pro-preview"),
                "sub_model": (
                    session.state.get("sub_model") or session.state.get("model")
                ),
                "max_iterations": session.state.get("max_iterations", 30),
                "files": session.state.get("files", []),
                "conversation": session.state.get("conversation", []),
                "events": session.state.get("ui_events", []),
            })
          else:
            await websocket.send_json({
                "type": "error",
                "message": f"Session {new_session_id} not found",
            })

      elif action == "new_session":
        # Create new session
        new_session_id = str(uuid.uuid4())
        session = await session_service.create_session(
            app_name=APP_NAME,
            user_id=DEFAULT_USER_ID,
            session_id=new_session_id,
            state={
                "title": f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "model": "gemini-3-pro-preview",
                "sub_model": None,
                "max_iterations": 30,
                "files": [],
                "conversation": [],
                "ui_events": [],
            },
        )
        await websocket.send_json({
            "type": "session_created",
            "session_id": session.id,
            "title": session.state.get("title"),
        })

      elif action == "list_sessions":
        response = await session_service.list_sessions(
            app_name=APP_NAME,
            user_id=DEFAULT_USER_ID,
        )
        sessions_data = []
        for s in response.sessions:
          conv = s.state.get("conversation", [])
          sessions_data.append({
              "session_id": s.id,
              "title": s.state.get("title", "Untitled"),
              "updated_at": (
                  datetime.fromtimestamp(s.last_update_time).isoformat()
                  if s.last_update_time
                  else None
              ),
              "message_count": len(conv),
          })
        # Sort by updated_at descending
        sessions_data.sort(
            key=lambda x: x.get("updated_at") or "", reverse=True
        )
        await websocket.send_json({
            "type": "sessions_list",
            "sessions": sessions_data,
        })

      elif action == "delete_session":
        del_session_id = data.get("session_id")
        if del_session_id:
          close_rlm(del_session_id)
          await session_service.delete_session(
              app_name=APP_NAME,
              user_id=DEFAULT_USER_ID,
              session_id=del_session_id,
          )
          await websocket.send_json({
              "type": "session_deleted",
              "session_id": del_session_id,
              "success": True,
          })

  except WebSocketDisconnect:
    pass  # Session already persisted


async def add_files(
    websocket: WebSocket, session: Session, patterns: list[str]
):
  """Add files to the session."""
  rlm = get_or_create_rlm(session)
  try:
    resolved = rlm.file_loader.create_lazy_files(patterns)
    if len(resolved) == 0:
      await websocket.send_json({
          "type": "error",
          "message": f"No files found matching: {' '.join(patterns)}",
      })
    else:
      # Update session state
      current_files = session.state.get("files", [])
      current_files.extend(patterns)
      await update_session_state(session, {"files": current_files})

      await websocket.send_json({
          "type": "files_added",
          "patterns": patterns,
          "count": len(resolved),
          "names": resolved.names[:10],
          "total": len(resolved),
      })
  except Exception as e:
    await websocket.send_json({
        "type": "error",
        "message": f"Could not resolve files: {e}",
    })


async def run_query(websocket: WebSocket, session: Session, prompt: str):
  """Run an RLM query and stream events."""
  rlm = get_or_create_rlm(session)

  # Add user message to conversation
  conversation = list(session.state.get("conversation", []))
  conversation.append({
      "role": "user",
      "content": prompt,
      "timestamp": datetime.now().isoformat(),
  })

  # Extract conversation history for the agent (exclude current message)
  # Only include role and content, not timestamp
  conversation_history = None
  if len(conversation) > 1:
    conversation_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in conversation[:-1]
    ]

  # Build file context
  files = session.state.get("files", [])
  if files:
    try:
      file_ctx = rlm.file_loader.build_context(files, lazy=True)
      file_count = file_ctx.get("file_count", 0)
      if file_count == 0:
        await websocket.send_json({
            "type": "error",
            "message": f"No files found matching patterns: {' '.join(files)}",
        })
        return
      ctx = file_ctx
    except Exception as e:
      await websocket.send_json({
          "type": "error",
          "message": f"Failed to load files: {e}",
      })
      return
  else:
    ctx = {
        "info": "No files loaded. The user is asking a question.",
    }

  # Send query start
  start_time = time.time()
  await websocket.send_json({
      "type": "query_start",
      "prompt": prompt,
  })

  try:
    event_id = 0
    current_iteration = 0
    final_answer = None
    ui_events = []

    async for event in rlm.run_streaming(ctx, prompt, conversation_history):
      if not event.custom_metadata:
        continue

      event_type = event.custom_metadata.get("event_type")
      if not event_type:
        continue

      if event_type == RLMEventType.ITERATION_START.value:
        current_iteration = event.custom_metadata.get("iteration", 0)

      # Format event for UI
      ui_event = format_event_for_ui(
          event.custom_metadata,
          event_id,
          start_time,
          current_iteration,
      )
      ui_events.append(ui_event)

      # Check for final answer
      if event.custom_metadata.get("answer"):
        final_answer = event.custom_metadata["answer"]

      await websocket.send_json(ui_event)
      event_id += 1

    # Add assistant message to conversation
    title = session.state.get("title", "")
    if final_answer:
      conversation.append({
          "role": "assistant",
          "content": final_answer,
          "timestamp": datetime.now().isoformat(),
      })

      # Auto-generate title from first exchange
      if title.startswith("Session ") and len(conversation) == 2:
        first_msg = conversation[0]["content"]
        title = first_msg[:50] + ("..." if len(first_msg) > 50 else "")

    # Update session state
    await update_session_state(
        session,
        {
            "conversation": conversation,
            "ui_events": ui_events,
            "title": title,
        },
    )

    # Send completion
    elapsed = time.time() - start_time
    await websocket.send_json({
        "type": "query_complete",
        "elapsed_seconds": elapsed,
        "total_events": event_id,
        "final_answer": final_answer,
        "title": session.state.get("title"),
    })

  except Exception as e:
    await websocket.send_json({
        "type": "error",
        "message": str(e),
    })


def create_app(
    model: str = "gemini-3-pro-preview",
    sub_model: str | None = None,
    max_iterations: int = 30,
    log_dir: str | None = None,
    db_url: str = "sqlite+aiosqlite:///./sessions.db",
) -> FastAPI:
  """Create a configured FastAPI app."""
  # Update module-level config
  _config["db_url"] = db_url
  _config["model"] = model
  _config["sub_model"] = sub_model
  _config["max_iterations"] = max_iterations
  _config["log_dir"] = log_dir

  # Also store in app.state for easy access
  app.state.default_model = model
  app.state.default_sub_model = sub_model
  app.state.default_max_iterations = max_iterations
  app.state.default_log_dir = log_dir
  return app


def main():
  """Run the web server."""
  import argparse

  import uvicorn

  parser = argparse.ArgumentParser(
      description="ADK-RLM Web Interface",
  )
  parser.add_argument(
      "--host",
      type=str,
      default="127.0.0.1",
      help="Host to bind to (default: 127.0.0.1)",
  )
  parser.add_argument(
      "--port",
      type=int,
      default=8000,
      help="Port to bind to (default: 8000)",
  )
  parser.add_argument(
      "--model",
      "-m",
      type=str,
      default="gemini-3-pro-preview",
      help="Default model (default: gemini-3-pro-preview)",
  )
  parser.add_argument(
      "--sub-model",
      "-s",
      type=str,
      help="Default sub-model (defaults to main model)",
  )
  parser.add_argument(
      "--max-iterations",
      "-i",
      type=int,
      default=30,
      help="Default max iterations (default: 30)",
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
      default="sqlite+aiosqlite:///./sessions.db",
      help=(
          "SQLAlchemy database URL for sessions (default:"
          " sqlite+aiosqlite:///./sessions.db)"
      ),
  )
  parser.add_argument(
      "--reload",
      action="store_true",
      help="Enable auto-reload for development",
  )

  args = parser.parse_args()

  if args.reload:
    # When using reload, set environment variables so config persists
    # across module reimports
    os.environ["RLM_DB_URL"] = args.db_url
    os.environ["RLM_MODEL"] = args.model
    os.environ["RLM_MAX_ITERATIONS"] = str(args.max_iterations)
    if args.log_dir:
      os.environ["RLM_LOG_DIR"] = args.log_dir
    if args.sub_model:
      os.environ["RLM_SUB_MODEL"] = args.sub_model

    uvicorn.run(
        "adk_rlm.web:app",
        host=args.host,
        port=args.port,
        reload=True,
    )
  else:
    # When not using reload, configure app directly
    configured_app = create_app(
        model=args.model,
        sub_model=args.sub_model,
        max_iterations=args.max_iterations,
        log_dir=args.log_dir,
        db_url=args.db_url,
    )

    uvicorn.run(
        configured_app,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
  main()
