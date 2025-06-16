# Import necessary Python modules
import os  # For interacting with the operating system (e.g., environment variables, file paths)
import asyncio  # For asynchronous programming, core to FastAPI's async features
import uvicorn  # The ASGI server that will run the FastAPI application; it's used here
                #  if the script is run directly, but typically you run Uvicorn from
                #  the command line.

from contextlib import asynccontextmanager  # For creating async context managers (for lifespan)

# Third-party libraries for FastAPI
from fastapi import FastAPI, Request, HTTPException  # Core FastAPI classes, Request object for incoming requests, & HTTPException for errors.
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse  # Different types of HTTP responses.
from fastapi.staticfiles import StaticFiles  # For serving static files (that is, files that don't change, like HTML, CSS, JS).
from fastapi.middleware.cors import CORSMiddleware  # For handling Cross-Origin Resource Sharing (commonly needed for web apps).

# Python standard libraries
from pathlib import Path  # For object-oriented manipulation of file system paths.
import json  # For parsing JSON data.

# Application-specific imports from this project
# The `.` in `.camera_agent` means that the module is in the same directory ("app") as this file.
from .camera_agent import CameraAgent  # Import the CameraAgent class.
from .audio_agent import AudioAgent  # Import the AudioAgent class.

# Import for loading environment variables from a `.env` file.
from dotenv import load_dotenv

# --- Configuration and Environment Setup ---
# Define the path to the `.env` file.
# __file___ represents the path to the current file (i.e., `streaming-test2/app/main.py`).
# Path(__file__).resolve() ensures an absolute path.
  # .parent takes us from `app/main.py` to `app/`.
  # .parent.parent takes us from `app/` to `streaming-test2/`.
  # Then we append '.env' to form the full path.
dotenv_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)  # Load the environment variables from the .env file.

# Read environment variables after loading the .env file.
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")

# Global variables to hold the singleton instances of the agents.
camera_agent_instance = None
audio_agent_instance = None

# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global camera_agent_instance, audio_agent_instance  # Declare that we will modify the global variables.
    print("MAIN_APP: FastAPI server starting up (with lifespan)...")

    # Critical check for environment variables.
    if not PROJECT_ID or not LOCATION:
        print(f"MAIN_APP: FATAL - GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_LOCATION not set in .env (path: {dotenv_path}). Agents will not initialize.")
        camera_agent_instance = None
        audio_agent_instance = None
        yield # This is where the application runs. Startup has completed.
        # Code after yield runs on shutdown.
        print("MAIN_APP: FastAPI server shutting down (with lifespan)...")
        if camera_agent_instance: # Should be None if critical env vars were missing
            print("MAIN_APP: Stopping CameraAgent...")
            camera_agent_instance.stop()
        return #Use return instead of exception for normal startup logic

    # Initialize agents here
    try:
        print("MAIN_APP: Initializing CameraAgent...")
        camera_agent_instance = CameraAgent(project_id=PROJECT_ID, location=LOCATION)
        camera_agent_instance.start()
        print("MAIN_APP: CameraAgent initialized and started.")
    except Exception as e:
        print(f"MAIN_APP: Error initializing CameraAgent: {e}")
        camera_agent_instance = None

    try:
        print("MAIN_APP: Initializing AudioAgent...")
        audio_agent_instance = AudioAgent(project_id=PROJECT_ID, location=LOCATION)
        print("MAIN_APP: AudioAgent initialized.")
    except Exception as e:
        print(f"MAIN_APP: Error initializing AudioAgent: {e}")
        audio_agent_instance = None

    yield # This is the point where the application is considered running.
    # After the application has finished, code here will be executed as part of shutdown.
    print("MAIN_APP: FastAPI server shutting down (after yield in lifespan)...")
    if camera_agent_instance:
        print("MAIN_APP: Stopping CameraAgent...")
        camera_agent_instance.stop()

    # Audio agent sessions are managed by the AudioAgent itself when SSE connections close.

# --- FastAPI Application Initialization with Lifespan ---
app = FastAPI(lifespan=lifespan)  # Initialize FastAPI with the lifespan handler.

# CORS (Cross-Origin Resource Sharing) Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define paths for static files and log files.
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

# --- HTML Page Endpoint ---
@app.get("/", response_class=HTMLResponse)
async def get_root():
    return FileResponse(STATIC_DIR / "index.html")

# --- Audio Agent Endpoints ---
@app.get("/events/{user_id}")
async def audio_sse_endpoint(user_id: str, request: Request, is_audio: str = "true"):
    if not audio_agent_instance:
        print("MAIN_APP SSE: AudioAgent not initialized.")
        raise HTTPException(status_code=503, detail="AudioAgent not available.")

    print(f"MAIN_APP SSE request for user: {user_id}, audio: {is_audio}")
    try:
        live_events, _ = await audio_agent_instance.start_adk_session(user_id, is_audio == "true")

        async def event_generator():
            """
Async generator that yields data chunks from the AudioAgent's SSE processor.
It also monitors if the client disconnects, and performs cleanup.
            """
            try:
                async for data_chunk in audio_agent_instance.agent_to_client_sse_handler(user_id):
                    if await request.is_disconnected():
                        break
                    yield data_chunk
            except asyncio.CancelledError:
                print(f"MAIN_APP: SSE for {user_id} cancelled.")
            except Exception as e:
                print(f"MAIN_APP: SSE error for {user_id}: {e}")
            finally:
                print(f"MAIN_APP: Cleaning ADK session for {user_id} (SSE ended).")
                await audio_agent_instance.stop_adk_session(user_id)

        # The event_generator must be returned by the endpoint function for StreamingResponse
        # The StreamingResponse will then iterate over the async generator.
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except Exception as e:
        print(f"MAIN_APP: Error starting SSE for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f'Failed to start session: {str(e)}')

@app.post("/send/{user_id}")
async def audio_send_message_endpoint(user_id: str, request: Request):
    if not audio_agent_instance:
        print("MAIN_APP SEND ERROR: AudioAgent not initialized.")
        raise HTTPException(status_code=503, detail="AudioAgent not available.")

    try:
        message_data = await request.json()
    except json.JSONDecodeError:
        print("MAIN_APP SEND ERROR: Invalid JSON received.")
        raise HTTPException(status_code=400, detail="Invalid JSON.")

    response = await audio_agent_instance.client_to_agent_handler(user_id, message_data)

    if response.get("error"):
        print(f"MAIN_APP SEND ERROR: Agent returned error for user {user_id}: {response['error']}")
        raise HTTPException(status_code=400, detail=response["error"])
    return response

# --- Log File Fetching Endpoints ---
@app.get("/logs/{log_type}")
async def get_log_content(log_type: str):
    if log_type not in ["camera", "audio"]:
        print(f"MAIN_APP LOG ERROR: Invalid log type requested: {log_type}")
        raise HTTPException(status_code=404, detail="Invalid log type. Use 'camera' or 'audio'.")

    log_file_name = f"{log_type}_log.json"
    log_file_path = LOGS_DIR / log_file_name

    if not log_file_path.exists() or log_file_path.stat().st_size == 0:
        return JSONResponse(content=[], status_code=200)

    try:
        with open(log_file_path, 'r') as f:
            log_content = json.load(f)
        return JSONResponse(content=log_content)
    except Exception as e:
        print(f'MAIN_APP LOG ERROR: Error reading log file {log_file_path}: {e}')
        return JSONResponse(content=[{"error": f"Failed to parse {log_file_name}"}], status_code=200)


@app.get("/agent/camera/status")
async def get_camera_agent_status():
    if camera_agent_instance and camera_agent_instance.running:
        return {"status": "running", "monitoring": (camera_agent_instance.video_monitor.cap is not None) and camera_agent_instance.video_monitor.cap.isOpened()}
    return {"status": "stopped"}

# --- Run the Application (for local development) ---
if __name__ == "__main__":
    print("MAIN_APP: To run this app, navigate to the `streaming-test2` directory and use Uvicorn. Example:") # Corrected comment
    print("python -m uvicorn app.main:app --reload --port 8000 --host 0.0.0.0  (from inside streaming-test2)") # Corrected command
