# ADK Integration Guide: A Decoupled Client-Server Architecture

This document explains the decoupled client-server architecture used in the security agent sample. This design leverages a "thin client" frontend and an "intelligent" backend to create a responsive, secure, and powerful user experience.

## Architectural Overview

- **Frontend (Thin Client):** A Streamlit application that uses the Google GenAI SDK directly. It handles the user interface and the real-time, asynchronous streaming of the agent's conversation. It is lightweight and does not contain complex logic or sensitive credentials.

- **Backend (Intelligent Server):** A FastAPI application that serves as the brain of the system. It provides "tools-as-a-service" to the frontend agent, manages all security credentials, performs complex data aggregation and analysis, and maintains persistent conversation history.

---

## Frontend Integration

The frontend, located in `frontend/`, is responsible for the user-facing experience.

### Key Components:
- **`components/chat/chat_view.py`**: This is the core of the user interaction.
  - It uses `google.genai.adk.Agent` and `Runner` to manage the conversation.
  - It leverages `async/await` and `st.write_stream` to render the agent's responses in real-time without blocking the UI.
  - **Crucially, the agent's "tools" are simple wrappers that make API calls to the backend.**

- **`api_client_consolidated.py`**: This has been deprecated for chat but is still used for non-chat API calls (e.g., initial data loading). For agent interactions, the frontend now uses the GenAI SDK directly.

### Communication Flow:
1. The user enters a prompt.
2. The Streamlit app passes the prompt to the GenAI `Runner`.
3. The agent, running in the frontend's Python process, decides to use a tool (e.g., `discover_gcp_resources`).
4. The tool's wrapper function makes an `httpx` call to the corresponding backend API endpoint (e.g., `POST /api/v1/assets/discover`).
5. The frontend streams the agent's final response to the user as it's generated.

---

## Backend Integration

The backend, located in `backend/`, provides the core logic and data processing capabilities.

### Key Responsibilities:
- **Tools-as-a-Service:** The backend exposes API endpoints that serve as the concrete implementations of the agent's tools. For example, the `/api/v1/assets/discover` endpoint contains the actual logic for querying the GCP Asset Inventory.
- **Security:** It securely manages all GCP credentials and API keys. The frontend never has access to them.
- **Data Processing:** It performs complex operations like querying multiple GCP services, aggregating the data, and formatting it into clean, consistent Pydantic models.
- **Session & Context Management:** It manages persistent conversation history, allowing the agent to have contextually aware conversations.

### Key Files:
- **`main.py`**: The FastAPI application entry point, defining the tool-serving API endpoints.
- **`services/`**: Contains the heavy-lifting logic for interacting with GCP and other services.
- **`models/`**: Defines the Pydantic data models that ensure a clean data contract between the backend and frontend.

---

## Running the Application

This decoupled architecture requires both services to be running:

1.  **Start the Backend (The Engine):**
    ```bash
    python contributing/samples/security_agent/run_backend.py
    ```

2.  **Start the Frontend (The Cockpit):**
    ```bash
    streamlit run contributing/samples/security_agent/frontend/main_app.py
    ```
