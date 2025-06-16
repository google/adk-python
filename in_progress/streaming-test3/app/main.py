# main.py

import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse

# This is the LlmAgent that can stream
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig

from .audio_agent import AudioAgent
from .camera_agent import CameraAgent
from .web_interface import generate_frames, get_camera_instance

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- API and WebSocket Routes ---
@app.get("/video_feed", include_in_schema=False)
def video_feed():
    camera = get_camera_instance()
    return StreamingResponse(
        generate_frames(camera),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established.")

    # Create separate agent instances
    audio_agent = AudioAgent()
    camera_agent = CameraAgent()

    # Define the single callback for both agents
    async def response_handler(response: str, agent_name: str):
        if response:
            await websocket.send_json({"agent": agent_name, "response": response})

    # Create RunConfigs that pass the agent's name to the callback
    audio_run_config = RunConfig(
        response_handler=lambda res: response_handler(res, "Audio_Agent")
    )
    camera_run_config = RunConfig(
        response_handler=lambda res: response_handler(res, "Camera_Agent")
    )

    try:
        # Start both agents. The `run_live` method in the working example
        # is a blocking call that we run in the background.
        audio_task = asyncio.create_task(audio_agent.run_live(audio_run_config))
        camera_task = asyncio.create_task(camera_agent.run_live(camera_run_config))

        logger.info("Agent live runs initiated in background tasks.")

        # Loop to process incoming messages
        while True:
            data = await websocket.receive_json()
            agent_name = data["agent"]
            payload = data["payload"]

            if agent_name == "Audio_Agent":
                await audio_agent.send_live_request(payload, audio_run_config)
            elif agent_name == "Camera_Agent":
                await camera_agent.send_live_request(payload, camera_run_config)

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"EXCEPTION in websocket_endpoint: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up resources...")
        # Cancel the background agent tasks
        if 'audio_task' in locals() and not audio_task.done():
            audio_task.cancel()
        if 'camera_task' in locals() and not camera_task.done():
            camera_task.cancel()
        logger.info("Agent tasks cancelled.")


# --- Static File and Root HTML Routes ---
static_dir = Path(__file__).parent / "static"
app.mount(
    "/static",
    StaticFiles(directory=static_dir),
    name="static",
)

@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    return FileResponse(static_dir / "index.html")