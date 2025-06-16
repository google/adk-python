# app/live_server.py
import asyncio
import json
import logging
import os
import uuid
import base64 # Needed for decoding/encoding audio
from pathlib import Path
import sys # To modify path for imports if needed
from contextlib import suppress # For ignoring CancelledError during cleanup

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse # Use Starlette directly for FileResponse

# --- ADK Imports ---
from google.adk.runners import Runner
from google.adk.agents.run_config import RunConfig
from google.adk.sessions import InMemorySessionService, Session # Using in-memory for simplicity
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.events import Event, EventActions # Import EventActions for state delta
# Use alias for genai types to avoid conflicts if any
from google.genai import types as genai_types

from typing import Any, Dict, Optional


# --- Configuration ---
from dotenv import load_dotenv
# Load .env file from the parent directory (project root)
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path, override=True)

# Setup logging
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO').upper(),
                   format='%(asctime)s - %(name)s - %(levelname)s - LIVE_SERVER - %(message)s')
logger = logging.getLogger(__name__)

# --- Import the Host Agent CREATION FUNCTION ---
try:
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from host_agent.agent import create_host_agent
    logger.info("Successfully imported host_agent creation function.")
except ImportError as e:
     logger.critical(f"Could not import host_agent.agent: {e}. Check sys.path and ensure host_agent/agent.py exists.", exc_info=True)
     create_host_agent = None # type: ignore

if not create_host_agent:
    logger.critical("Host agent creation function failed to load. Exiting.")
    sys.exit(1)

APP_NAME = "ProjectHorizonLive"
STATIC_DIR = Path(__file__).parent / "static"

# --- ADK Setup ---
session_service = InMemorySessionService()
runner: Optional[Runner] = None # Initialize as None

# --- FastAPI App ---
app = FastAPI(title="Project Horizon Live Server") # Create app instance early

async def initialize_adk_system():
    """Initializes the ADK system, including specialist agent discovery and HostAgent creation."""
    global runner

    if not create_host_agent:
        logger.critical("create_host_agent function not available. Cannot initialize ADK system.")
        # This should ideally prevent server from starting, but with FastAPI events,
        # we'll log critical and the app might fail to handle requests properly.
        return

    temp_host_agent = await create_host_agent()
    if not temp_host_agent:
        logger.critical("Failed to create HostAgent after discovery.")
        return
    
    runner = Runner(
        app_name=APP_NAME,
        agent=temp_host_agent,
        session_service=session_service,
    )
    logger.info(f"ADK Runner initialized with dynamically configured agent: {temp_host_agent.name}")

# --- FastAPI Startup Event ---
@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI server starting up...")
    await initialize_adk_system()
    if not runner:
        logger.critical("ADK Runner failed to initialize during startup. Server might not function correctly.")
    else:
        logger.info("ADK System initialized successfully within FastAPI startup.")


# --- WebSocket Endpoint ---
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Handles WebSocket connection for live agent interaction."""
    await websocket.accept()
    logger.info(f"WebSocket client connected for session: {session_id}")
    user_id = f"live_user_{session_id}"

    live_events: Optional[asyncio.StreamReader] = None
    live_request_queue: Optional[LiveRequestQueue] = None
    adk_run_task: Optional[asyncio.Task] = None
    client_listener_task: Optional[asyncio.Task] = None
    adk_session: Optional[Session] = None

    if not runner:
        logger.error(f"Runner not initialized for session {session_id}. Aborting WebSocket connection.")
        await websocket.close(code=1011, reason="Server not ready (Runner missing)")
        return

    try:
        adk_session = session_service.get_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )
        if not adk_session:
            adk_session = session_service.create_session(
                app_name=APP_NAME, user_id=user_id, session_id=session_id, state={}
            )
            logger.info(f"Created new ADK session: {session_id}")
        else:
             logger.info(f"Resumed existing ADK session: {session_id}")

        live_request_queue = LiveRequestQueue()
        run_config = RunConfig(response_modalities=["AUDIO"])
        live_events = runner.run_live( # type: ignore
            session=adk_session,
            live_request_queue=live_request_queue,
            run_config=run_config,
        )
        logger.info(f"ADK run_live started for session {session_id}")

        async def adk_to_client():
            nonlocal adk_session
            logger.debug(f"[{session_id}] ADK -> Client task started.")
            audio_chunk_counter = 0
            try:
                async for event in live_events: # type: ignore
                    logger.debug(f"[{session_id}] Raw ADK Event Received: {event}")
                    message_to_send = None
                    event_processed = False

                    if hasattr(event, 'interrupted') and event.interrupted:
                         logger.info(f"[{session_id}] Received interruption signal from ADK.")
                         await websocket.send_json({"type": "interrupted"})
                         logger.info(f"[{session_id}] Sent 'interrupted' signal to client.")

                    if event.content and event.content.parts:
                         part = event.content.parts[0]
                         if part.inline_data and part.inline_data.mime_type.startswith("audio/"):
                              audio_bytes = part.inline_data.data
                              audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                              message_to_send = {"type": "audio", "data": audio_b64}
                              audio_chunk_counter += 1
                              if audio_chunk_counter == 1:
                                  logger.info(f"[{session_id}] Receiving audio stream from ADK...")
                              logger.debug(f"[{session_id}] Sending audio chunk #{audio_chunk_counter} ({len(audio_bytes)} bytes) to client.")
                              await websocket.send_json(message_to_send)
                              event_processed = True

                    if event.turn_complete:
                        if audio_chunk_counter > 0:
                            logger.info(f"[{session_id}] Finished sending {audio_chunk_counter} audio chunks.")
                        audio_chunk_counter = 0
                        await websocket.send_json({"type": "turn_complete"})
                        logger.info(f"[{session_id}] Sent turn_complete signal to client.")
                        event_processed = True

                    if not event_processed and not event.actions:
                        is_tool_event = bool(event.get_function_calls() or event.get_function_responses())
                        if not is_tool_event:
                             logger.debug(f"[{session_id}] Skipping event: Author={event.author}, Partial={event.partial}, Content Type={type(event.content.parts[0]).__name__ if event.content and event.content.parts else 'None'}")
                        else:
                             if event.get_function_calls():
                                 logger.debug(f"[{session_id}] Processing event: Tool Call Requested by {event.author}")
                             elif event.get_function_responses():
                                 logger.debug(f"[{session_id}] Processing event: Tool Response from {event.author}")

                    if event.actions and (event.actions.state_delta or event.actions.artifact_delta):
                        logger.debug(f"[{session_id}] Appending event with actions: {event.actions}")
                        if adk_session:
                            session_service.append_event(adk_session, event)
                            adk_session = session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id) # type: ignore
                            logger.debug(f"[{session_id}] Session state possibly updated by event actions.")
                        else:
                            logger.warning(f"[{session_id}] adk_session is None, cannot append event actions.")
            except WebSocketDisconnect:
                 logger.info(f"[{session_id}] WebSocket disconnected during ADK event processing.")
            except asyncio.CancelledError:
                 logger.info(f"[{session_id}] ADK -> Client task cancelled.")
            except Exception as e:
                 logger.error(f"[{session_id}] Error in ADK -> Client task: {e}", exc_info=True)
            finally:
                 logger.debug(f"[{session_id}] ADK -> Client task finished.")

        async def client_to_adk():
            nonlocal adk_session
            logger.debug(f"[{session_id}] Client -> ADK task started.")
            try:
                while True:
                    data = await websocket.receive_json()
                    message_type = data.get("type")
                    logger.debug(f"[{session_id}] Received '{message_type}' from client.")

                    if message_type == "audio":
                        audio_b64 = data.get("data", "")
                        if audio_b64:
                            audio_bytes = base64.b64decode(audio_b64)
                            if audio_bytes:
                                audio_blob = genai_types.Blob(mime_type='audio/pcm', data=audio_bytes)
                                if live_request_queue: live_request_queue.send_realtime(blob=audio_blob)
                            else:
                                logger.warning(f"[{session_id}] Received empty audio data from client.")
                    elif message_type == "text":
                        text_data = data.get("data", "")
                        if text_data:
                            logger.info(f"[{session_id}] Sending text '{text_data}' to ADK.")
                            content = genai_types.Content(role='user', parts=[genai_types.Part(text=text_data)])
                            if live_request_queue: live_request_queue.send_content(content=content)
                    elif message_type == "end_of_turn":
                        logger.info(f"[{session_id}] Client indicated end of turn.")
                        logger.debug(f"[{session_id}] End of turn signal received, letting Gemini API infer turn end.")
                    elif message_type == "toggle_mock":
                         mock_value = data.get("value", False)
                         logger.info(f"[{session_id}] Setting mock_a2a_calls state to: {mock_value}")
                         state_update_event = Event(
                             author="ui_control",
                             invocation_id = f"ui_mock_toggle_{uuid.uuid4().hex[:8]}",
                             actions=EventActions(
                                 state_delta={'mock_a2a_calls': mock_value}
                             )
                         )
                         if adk_session:
                             session_service.append_event(adk_session, state_update_event)
                             adk_session = session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id) # type: ignore
                             logger.info(f"[{session_id}] State 'mock_a2a_calls' updated via event to: {adk_session.state.get('mock_a2a_calls')}") # type: ignore
                         else:
                             logger.warning(f"[{session_id}] adk_session is None, cannot update mock_a2a_calls state.")
                    else:
                         logger.warning(f"[{session_id}] Received unknown message type from client: {message_type}")

            except WebSocketDisconnect:
                logger.info(f"[{session_id}] Client WebSocket disconnected (detected in listener).")
                if live_request_queue: live_request_queue.close()
            except asyncio.CancelledError:
                 logger.info(f"[{session_id}] Client -> ADK task cancelled.")
            except Exception as e_inner:
                logger.error(f"[{session_id}] Error in client listener task: {e_inner}", exc_info=True)
                if live_request_queue: live_request_queue.close()
            finally:
                logger.debug(f"[{session_id}] Client -> ADK task finished.")

        logger.info(f"[{session_id}] Starting ADK <-> WebSocket bridge tasks.")
        adk_run_task = asyncio.create_task(adk_to_client())
        client_listener_task = asyncio.create_task(client_to_adk())
        done, pending = await asyncio.wait(
            [adk_run_task, client_listener_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        logger.info(f"[{session_id}] One of the bridge tasks completed ({len(done)} done, {len(pending)} pending).")

        for task in pending:
            if not task.done():
                logger.debug(f"[{session_id}] Cancelling pending bridge task...")
                task.cancel()
                with suppress(asyncio.CancelledError): await task

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected gracefully for session: {session_id}")
    except asyncio.CancelledError:
         logger.info(f"WebSocket endpoint task cancelled for session: {session_id}")
    except Exception as e_outer:
        logger.error(f"Unhandled error in WebSocket endpoint for session {session_id}: {e_outer}", exc_info=True)
        with suppress(Exception):
            await websocket.close(code=1011, reason=f"Internal Server Error: {type(e_outer).__name__}")
    finally:
        logger.info(f"Performing final cleanup for session: {session_id}")
        if live_request_queue:
             logger.debug(f"[{session_id}] Closing live request queue in final cleanup.")
             try:
                 live_request_queue.close()
             except Exception as q_close_err:
                 logger.warning(f"[{session_id}] Error closing LiveRequestQueue: {q_close_err}")
        if adk_run_task and not adk_run_task.done(): adk_run_task.cancel()
        if client_listener_task and not client_listener_task.done(): client_listener_task.cancel()
        if adk_session:
            try:
                session_service.delete_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
                logger.info(f"Session removed from service: {session_id}")
            except Exception as del_err:
                 logger.warning(f"[{session_id}] Error deleting session: {del_err}")
        logger.info(f"WebSocket connection fully closed for session: {session_id}")


# --- Static File Serving ---
if not STATIC_DIR.is_dir():
    logger.error(f"Static directory not found at {STATIC_DIR}. UI will not be served.")
else:
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    logger.info(f"Serving static files from {STATIC_DIR}")

    @app.get("/")
    async def read_index():
        index_path = STATIC_DIR / "index.html"
        if not index_path.is_file():
            logger.error("index.html not found in static directory.")
            return {"error": "index.html not found"}, 404
        logger.debug("Serving index.html")
        return FileResponse(str(index_path))

# --- Server Startup (for running directly) ---
# The `main_async_startup` is now only needed if you run this file directly.
# Uvicorn will handle app creation and startup events when run as `uvicorn app.live_server:app`
if __name__ == "__main__":
    project_root_path = Path(__file__).parent.parent
    if str(project_root_path) not in sys.path:
        sys.path.insert(0, str(project_root_path))
        logger.debug(f"Added project root to sys.path for direct execution: {project_root_path}")

    LIVE_SERVER_MODEL_ID = os.getenv("LIVE_SERVER_MODEL")
    if not LIVE_SERVER_MODEL_ID:
        logger.warning("LIVE_SERVER_MODEL env var not set. Defaulting to 'gemini-2.0-flash-exp'.")
        LIVE_SERVER_MODEL_ID = "gemini-2.0-flash-exp"

    LIVE_SERVER_HOST = os.getenv("LIVE_SERVER_HOST")
    if LIVE_SERVER_HOST is None:
        logger.critical("CRITICAL: LIVE_SERVER_HOST env var not set.")
        sys.exit(1)

    raw_port = os.getenv("LIVE_SERVER_PORT")
    if raw_port is None:
        logger.critical("LIVE_SERVER_PORT env var not set.")
        sys.exit(1)
    
    try:
        LIVE_SERVER_PORT = int(raw_port)
    except ValueError:
        logger.critical(f"Invalid LIVE_SERVER_PORT: '{raw_port}'. Must be integer.")
        sys.exit(1)

    # When running directly, we need to manually trigger the startup event logic
    # (or Uvicorn does it if 'app' object is passed)
    # Uvicorn will call startup events registered on the app object.
    # No need to call initialize_adk_system() directly here if Uvicorn is used.
    logger.info(f"Starting Project Horizon Live Server (direct run) on http://{LIVE_SERVER_HOST}:{LIVE_SERVER_PORT}")
    try:
        import uvicorn
        uvicorn.run(app, host=LIVE_SERVER_HOST, port=LIVE_SERVER_PORT, log_level="info")
    except ImportError:
         logger.critical("Uvicorn not installed. Run 'pip install uvicorn[standard]'.")
    except Exception as startup_error:
         logger.critical(f"Failed to start Uvicorn server: {startup_error}", exc_info=True)