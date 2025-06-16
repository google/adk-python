# Live Multimodal Streaming Agent with Gemini

This project implements a Python-based multimodal agent that interacts using live audio and video streams, leveraging Google Cloud's Vertex AI (with a Gemini model), Speech-to-Text, and Text-to-Speech services. It features a simple web interface for real-time video monitoring and chat interaction.

## Features

*   **Live Audio Input/Output**: Captures audio from a microphone, transcribes it in real-time, and can synthesize text responses back to audio.
*   **Live Video Input & Monitoring**: Captures video from a camera and monitors the feed for significant changes (e.g., motion).
*   **AI Model Interaction**: Uses a specified Gemini model via Vertex AI to understand and comment on text, audio transcriptions, and video frame changes.
*   **Web Interface**: Provides a browser-based UI to:
    *   View the live camera feed.
    *   Engage in text-based chat with the agent.
    *   Receive real-time updates on detected video changes and the agent's comments.
    *   Start and stop the agent's interaction.
*   **Environment Configuration**: Uses a `.env` file for easy setup of Google Cloud project details.
*   **Unit Tests**: Includes a suite of unit tests for core components.

## Prerequisites

1.  **Python**: Python 3.8 or newer.
2.  **Google Cloud Project**:
    *   A Google Cloud Platform project.
    *   Vertex AI API enabled.
    *   Cloud Speech-to-Text API enabled.
    *   Cloud Text-to-Speech API enabled.
    *   Billing enabled for the project.
3.  **Authentication**:
    *   `gcloud` CLI installed and configured.
    *   Application Default Credentials (ADC) set up. The easiest way is to run:
        ```bash
        gcloud auth application-default login
        ```
    *   Alternatively, you can use a service account key by setting the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of your JSON key file. This is often used in production or automated environments.
4.  **Hardware**:
    *   A working microphone connected to the system where the application will run.
    *   A working camera (e.g., webcam) connected to the system.
5.  **System Dependencies**:
    *   **For `sounddevice` (audio)**: PortAudio library.
        *   On Debian/Ubuntu: `sudo apt-get update && sudo apt-get install libportaudio2`
        *   On Fedora: `sudo dnf install portaudio-devel`
        *   On macOS (using Homebrew): `brew install portaudio`
        *   On Windows: PortAudio is often bundled with Python audio libraries, but ensure your microphone drivers are up to date.
    *   **For `opencv-python` (video)**: While `opencv-python-headless` bundles many dependencies, ensure your system allows access to camera hardware.

## Setup Instructions

1.  **Clone the Repository** (if applicable):
    ```bash
    # git clone <repository-url>
    # cd <repository-directory>/streaming-test
    # For this project, you are already in the streaming-test directory.
    ```

2.  **Create and Activate Virtual Environment**:
    It is highly recommended to use a virtual environment.
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**:
    Create a `.env` file in the `streaming-test` directory (root of this project). Add the following lines, replacing placeholders with your actual Google Cloud project details:
    ```env
    GOOGLE_GENAI_USE_VERTEXAI=True
    GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
    GOOGLE_CLOUD_LOCATION="your-gcp-project-location" # e.g., us-central1
    ```
    The application is currently hardcoded to use `analog-delight-459210-v0` and `us-central1` as per the initial issue, so ensure these are correct for your setup or update the `.env` file accordingly.

5.  **Verify Google Cloud Authentication**:
    Ensure your environment is authenticated to Google Cloud as mentioned in the Prerequisites (e.g., by running `gcloud auth application-default login`).

## Running the Application

1.  **Start the Application**:
    From the `streaming-test` directory, run:
    ```bash
    python run.py
    ```
    You should see log messages indicating the agent and web server are starting.

2.  **Access the Web Interface**:
    Open your web browser and navigate to:
    [http://localhost:8080](http://localhost:8080) (or the host/port specified in `app/config.py` if changed).

    *   You should see the live video feed.
    *   You can type messages in the chat input to interact with the agent.
    *   Use the "Start Agent" and "Stop Agent" buttons to control the main interaction logic (though some features like video feed might start automatically).
    *   Watch for messages about detected video changes and the agent's comments.

## Running Tests

To run the unit tests:
```bash
python run_tests.py
```
This command will discover and execute all tests in the `tests` directory. You should see output indicating the number of tests run and their status.

## Project Structure

```
streaming-test/
├── .venv/                  # Python virtual environment
├── .env                    # Environment variables (GITIGNORED)
├── .gitignore              # Specifies intentionally untracked files
├── app/                    # Main application package
│   ├── __init__.py
│   ├── agent.py            # Core MultimodalAgent class
│   ├── audio_utils.py      # Audio recording, transcription, synthesis
│   ├── config.py           # Application configuration
│   ├── main.py             # Main application logic (used by run.py)
│   ├── video_utils.py      # Video capture and change detection
│   ├── web_interface.py    # Flask web server and UI logic
│   └── templates/
│       └── index.html      # HTML template for the web UI
├── requirements.txt        # Python dependencies
├── run.py                  # Main script to start the application (web server & agent tasks)
├── run_tests.py            # Script to execute unit tests
└── tests/                  # Unit tests
    ├── __init__.py
    ├── test_agent.py
    ├── test_audio_utils.py
    └── test_video_utils.py
```

## Troubleshooting

*   **Camera/Microphone Not Working**:
    *   Ensure they are properly connected and enabled in your system settings.
    *   Check if other applications can access them.
    *   Permissions: Your OS might require explicit permission for Python or your terminal to access the camera/microphone.
    *   OpenCV/Sounddevice: If `VideoMonitor` or `record_audio` report errors, ensure system dependencies (like PortAudio) are installed correctly. Try a different camera index in `app/agent.py` if you have multiple cameras (e.g., `camera_index=1`).
*   **Google Cloud API Errors**:
    *   `PermissionDenied`: Check your ADC (`gcloud auth list`) or service account key permissions. Ensure the necessary APIs (Vertex AI, Speech, Text-to-Speech) are enabled in your GCP project.
    *   `Billing not enabled`: Ensure billing is active for your GCP project.
    *   `QuotaExceeded`: You might have hit API usage quotas. Check the GCP console.
*   **Python Errors**:
    *   `ModuleNotFoundError`: Ensure your virtual environment is activated and all dependencies in `requirements.txt` are installed.
    *   Errors from `sounddevice` or `cv2`: Often related to missing system libraries or hardware access issues.
*   **Web Interface Issues**:
    *   Video feed not showing: Check browser console for errors. Ensure the Flask server is running and there are no errors in the terminal output related to camera capture.
    *   SSE events not updating: Check browser console and Flask server logs. The `/agent_events` endpoint might have issues, or the background thread in `run.py` might not be running correctly.

This README provides a starting point. Feel free to expand it as the project evolves.
