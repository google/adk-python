# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FastAPI application demonstrating ADK Gemini Live API Toolkit with WebSocket."""

import asyncio
import base64
import json
import logging
from pathlib import Path
import warnings

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Load environment variables from .env file BEFORE importing agent
load_dotenv(Path(__file__).parent / ".env")

# Import agent after loading environment variables
# pylint: disable=wrong-import-position
from google_search_agent.agent import agent  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("google_adk." + __name__)

# Suppress Pydantic serialization warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Application name constant
APP_NAME = "bidi-demo"

# ========================================
# Phase 1: Application Initialization (once at startup)
# ========================================

app = FastAPI()

# Mount static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Define your session service
session_service = InMemorySessionService()

# Define your runner
runner = Runner(app_name=APP_NAME, agent=agent, session_service=session_service)

# ========================================
# HTTP Endpoints
# ========================================


@app.get("/")
async def root():
  """Serve the index.html page."""
  return FileResponse(Path(__file__).parent / "static" / "index.html")


# ========================================
# WebSocket Endpoint
# ========================================


@app.websocket("/ws/{user_id}/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str,
    session_id: str,
    proactivity: bool = False,
    affective_dialog: bool = False,
) -> None:
  """WebSocket endpoint for bidirectional streaming with ADK.

  Args:
      websocket: The WebSocket connection
      user_id: User identifier
      session_id: Session identifier
      proactivity: Enable proactive audio (native audio models only)
      affective_dialog: Enable affective dialog (native audio models only)
  """
  logger.debug(
      f"WebSocket connection request: user_id={user_id},"
      f" session_id={session_id}, proactivity={proactivity},"
      f" affective_dialog={affective_dialog}"
  )
  await websocket.accept()
  logger.debug("WebSocket connection accepted")

  # ========================================
  # Phase 2: Session Initialization (once per streaming session)
  # ========================================

  # Native audio models ONLY support the AUDIO response modality, so the
  # model speaks its replies and transcription is enabled in both
  # directions to keep a readable transcript of the conversation.
  # Proactivity and affective dialog are native-audio-only features.
  run_config = RunConfig(
      streaming_mode=StreamingMode.BIDI,
      response_modalities=["AUDIO"],
      input_audio_transcription=types.AudioTranscriptionConfig(),
      output_audio_transcription=types.AudioTranscriptionConfig(),
      session_resumption=types.SessionResumptionConfig(),
      proactivity=(
          types.ProactivityConfig(proactive_audio=True) if proactivity else None
      ),
      enable_affective_dialog=affective_dialog if affective_dialog else None,
  )
  logger.debug(
      f"Model: {agent.model}, using AUDIO response modality, "
      f"proactivity={proactivity}, affective_dialog={affective_dialog}"
  )
  logger.debug(f"RunConfig created: {run_config}")

  # Get or create session (handles both new sessions and reconnections)
  session = await session_service.get_session(
      app_name=APP_NAME, user_id=user_id, session_id=session_id
  )
  if not session:
    await session_service.create_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )

  live_request_queue = LiveRequestQueue()

  # ========================================
  # Phase 3: Active Session (concurrent bidirectional communication)
  # ========================================

  async def upstream_task() -> None:
    """Receives messages from WebSocket and sends to LiveRequestQueue."""
    logger.debug("upstream_task started")
    while True:
      # Receive message from WebSocket (text or binary)
      message = await websocket.receive()

      # The low-level receive() reports a client disconnect as a
      # message rather than raising, so stop the loop explicitly.
      # Calling receive() again after this would raise a RuntimeError.
      if message["type"] == "websocket.disconnect":
        logger.debug("Client disconnected, stopping upstream_task")
        raise WebSocketDisconnect(message.get("code", 1000))

      # Handle binary frames (audio data)
      if "bytes" in message:
        audio_data = message["bytes"]
        logger.debug(f"Received binary audio chunk: {len(audio_data)} bytes")

        audio_blob = types.Blob(
            mime_type="audio/pcm;rate=16000", data=audio_data
        )
        live_request_queue.send_realtime(audio_blob)

      # Handle text frames (JSON messages)
      elif "text" in message:
        text_data = message["text"]
        logger.debug(f"Received text message: {text_data[:100]}...")

        json_message = json.loads(text_data)

        # Extract text from JSON and send to LiveRequestQueue
        if json_message.get("type") == "text":
          logger.debug(f"Sending text content: {json_message['text']}")
          content = types.Content(parts=[types.Part(text=json_message["text"])])
          live_request_queue.send_content(content)

        # Handle image data
        elif json_message.get("type") == "image":
          logger.debug("Received image data")

          # Decode base64 image data
          image_data = base64.b64decode(json_message["data"])
          mime_type = json_message.get("mimeType", "image/jpeg")

          logger.debug(
              f"Sending image: {len(image_data)} bytes, type: {mime_type}"
          )

          # Send image as blob
          image_blob = types.Blob(mime_type=mime_type, data=image_data)
          live_request_queue.send_realtime(image_blob)

  async def downstream_task() -> None:
    """Receives Events from run_live() and sends to WebSocket."""
    logger.debug("downstream_task started, calling runner.run_live()")
    logger.debug(
        f"Starting run_live with user_id={user_id}, session_id={session_id}"
    )
    async for event in runner.run_live(
        user_id=user_id,
        session_id=session_id,
        live_request_queue=live_request_queue,
        run_config=run_config,
    ):
      event_json = event.model_dump_json(exclude_none=True, by_alias=True)
      logger.debug(f"[SERVER] Event: {event_json}")
      await websocket.send_text(event_json)
    logger.debug("run_live() generator completed")

  # Run both tasks concurrently
  # Exceptions from either task will propagate and cancel the other task
  try:
    logger.debug("Starting asyncio.gather for upstream and downstream tasks")
    await asyncio.gather(upstream_task(), downstream_task())
    logger.debug("asyncio.gather completed normally")
  except WebSocketDisconnect:
    logger.debug("Client disconnected normally")
  except Exception as e:
    logger.error(f"Unexpected error in streaming tasks: {e}", exc_info=True)
  finally:
    # ========================================
    # Phase 4: Session Termination
    # ========================================

    # Always close the queue, even if exceptions occurred
    logger.debug("Closing live_request_queue")
    live_request_queue.close()
