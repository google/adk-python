# Streaming Test 2 - Multimodal Streaming Agents

This project (`streaming-test2`) demonstrates a multimodal agent system using Python, FastAPI, and Google Agent Development Kit (ADK) for live bi-directional audio, and a separate camera monitoring agent using Gemini.

## Features

- __CameraAgent__: Monitors a live video feed, detects significant visual changes, sends observations to a Gemini model ("gemini-2.0-flash"), and logs interpretations to `/logs/camera_log.json`.
- __AudioAgent__: Handles live bi-directional audio interaction using Google ADK and the "gemini-2.0-flash-live-preview-04-09" model. Logs transcriptions to `/logs/audio_log.json`.
- __Agent Communication__: The AudioAgent reads from the CameraAgent's log to gain contextual awareness for its responses.
- __FastAPI Backend__: Serves the application, handling SSE (Server-Sent Events) for audio streaming, and provides HTTP endpoints for client interaction.
- __Web Interface__: A basic HTML/CSS/JS frontend to interact with the agents and view logs.

## Prerequisites

- Python 3.8 + (3.9/3.10/3.11 recommended)
- `pip` for installing packages.
- Access to a camera (for the CameraAgent).
-  A microphone (for the AudioAgent).
- A Google Cloud Project with Vertex AI enabled.


## Setup Instructions

1. *__Create a Virtual Environment (Recommended)__*
   ```sh
   python -m venv .venv
   ```
   Activate the virtual environment:
   - On macOS/Linux:
     ```sh
     source .venv/bin/activate
     ```
   - On Windows:
     ```sh
     .venv\Scripts\activate
     ```

2. *__Install Dependencies__*
   With your virtual environment active, navigate to the `streaming-test2` directory and run the following command to install the required packages:
   ```sh
   pip install -r requirements.txt
   ```
   (Ensure you are in the `streaming-test2` directory when running this command, as requirements.txt is located there.)

3. *__Create the `.env` File__*
   Create a file named `.env` within the `streaming-test2` directory.
   Edit this `.env` file and add the following content, replacing the placeholders with your actual Google Cloud project details:
   ```        GOOGLE_GENAI_USE_VERTEXAI=True
        GOOGLE_CLOUD_PROJECT="analog-delight-459210-v0"
        GOOGLE_CLOUD_LOCATION="us-central1"
   ```
   Note: The Python scripts in the `app/` directory are configured to load this particular `.env` file from the `streaming-test2` directory.

4. *__Run the Application__*
   Navigate into the `streaming-test2` directory in your terminal.
   ```sh
   cd path/to/streaming-test2
   ```
   Then, run the following command to start the FastAPI application using Uvicorn:
   ```sh
   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
   - `app.main:app` tells Uvicorn to look for the `app` folder, the `main` module (main.py), and the `app` object within that module. This assumes you are running from the `streaming-test2` directory.
   - `--reload` enables automatic reloading when code changes are detected.
   - `--host 0.0.0.0` makes the server accessible on your network.
   - `--port 8000` sets the port the server is running on.

Once the server is running, open your web browser and navigate to `http://localhost:8000`.


## Usage

1. The UI will display statuses for the Camera Agent and the Audio Agent.
2. The Camera Agent will automatically start monitoring.
   - Its rager will appear in the 'Camera Log' section.
3. To interact with the AudioAgent:
   - Click the `Start Audio Interaction` button.
   - Your server permission to use the microphone.
   - Speak, or type a message in the input field and click `Send Text`.
   - The AudioAgent's response (audio and text transcription) will appear.
   - Click `Stop Audio Interaction` to end the audio session.
- Audio transcriptions will appear in the 'Audio Log' section.

## Project Structure

```
/streaming-test2
    .env                                     # (You create this) Google Cloud config
    README.md                                # This file
    requirements.txt                         # Python dependencies
    app/
        __init__.py                          # Makes 'app' a Python package
        audio_agent.py                       # Logic for the AudioAgent
        camera_agent.py                      # Logic for the CameraAgent
        config.py                            # Configuration (e.g., VIDEO_FPS)
        main.py                              # FastAPI application

        static/                              # Static files for the web UI
            css/
                style.css
            js/
                app.js
                audio-player.js
                audio-recorder.js
                pcm-player-processor.js
                pcm-recorder-processor.js
            index.html
        video_utils.py                       # Utilities for video processing
    logs/
        audio_log.json                       # Log of audio transcriptions
        camera_log.json                      # Log of camera observations
```

## Notes
- The JavaScript audio processor files (`audio-player.js`, `audio-recorder.js`, `pcm-player-processor.js`, `pcm-recorder-processor.js`) are adapted from Google ADK examples and are crucial for the audio functionality.
-  The application is designed for development and may require further configuration and security hardening for production use.
