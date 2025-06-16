import os # For interacting with the OS
import json # For JSON data handling
import base64 # For encoding/decoding binary data for JSON
import datetime # For timestamping log entries
import warnings # For handling warnings
import asyncio # For asynchronous programming, crucial for ADK streaming
from pathlib import Path # For object-oriented file path handling
from dotenv import load_dotenv # For loading environment variables from a “.env” file

# Imports from the Google Agent Development Kit (ADK)
from google.genai.types import Part, Content, Blob # Data structures for content in ADK
from google.adk.runners import InMemoryRunner # Runner for executing ADK agents in memory
from google.adk.agents import LiveRequestQueue, Agent # Core ADK Agent class and live request handling
from google.adk.agents.run_config import RunConfig # Configuration for how an agent runs

# --- Environment Variable Loading ---
# Set the path to the .env file, expecting it in the `streaming-test2` directory.
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=dotenv_path)

# Warnings filtering - suppress specific UserWarnings, often from pydantic used by FastAPI
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

APP_NAME = "ADK_Streaming_Audio_Agent" # A name for this application/module, used by ADK

class AudioAgent:
    """
    The AudioAgent handles live bi-directional audio streaming using the Google ADK.
    It integrates with a FastAPI backend to communicate with a client (e.g., a web browser).
    It manages ADK sessions, processes incoming and outgoing audio/text, logs
    transcriptions, and potentially incorporates context from other agents.
    """

    def __init__(self, project_id: str, location: str, model_name: str = "gemini-2.0-flash-live-preview-04-09"):
        """
        Initializes the AudioAgent.

        Args:
            project_id (str): The Google Cloud Project ID.
            location (str): The Google Cloud region for Vertex AI services.
            model_name (str, optional): The name of the Gemini model to use for live audio.
                                        Defaults to "gemini-2.0-flash-live-preview-04-09".
        """
        # Validate project ID and location, they are essential.
        if not project_id:
            raise ValueError("CAUDIO_AGENT: Google Cloud Project ID is required.")
        if not location:
            raise ValueError("CAUDIO_AGENT: Google Cloud Location is required.")

        self.project_id = project_id
        self.location = location

        # Ensure that the application is configured to use Vertex AI, as the ADK for large
        # language models typically relies on it.
        if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "False").lower() != "true":
            raise EnvironmentError("AUDIO_AGENT: GOOGLE_GENAI_USE_VERTEXAI must be set to True in .env")

        # Define the `self.root_agent_adk`. This is the core ADK Agent object.
        #   - `name`: A descriptive name for the agent.
        #   - `model`: The specific Gemini model that supports live audio streaming.
        #   - `description`: A brief description of the agent's purpose.
        #   - `instruction`: The system-prompt or instructions that guide the agent's behavior.
        #     It includes a placeholder to be aware of visual context, which will
        #     be populated from the camera agent's logs.
        self.root_agent_adk = Agent(
           name="streaming_audio_root_agent",
           model=model_name,
           description="Live audio conversational agent.",
           instruction="You are a helpful voice assistant. Respond to the user's speech. Be concise and natural. When appropriate, consider the following visual context from your surroundings.",
        )
        print(f"AUDIO_AGENT: Initialized with ADK root agent using model: {model_name}")

        # Define paths for log files.
        # Audio log file stores transcriptions of user and agent speech.
        self.audio_log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'audio_log.json')
        # Camera log file is used to read context from the CameraAgent.
        self.camera_log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'camera_log.json')

        # Ensure the 'logs' directory exists.
        os.makedirs(os.path.dirname(self.audio_log_file), exist_ok=True)
        # Initialize the audio log file with an empty JSON list.
        with open(self.audio_log_file, 'w') as f:
            json.dump([], f)
        print(f"AUDIO_AGENT: Logging audio transcriptions to: {self.audio_log_file}")

        # Ensure the camera log file exists for reading, even if it's just initialized as empty.
        if not os.path.exists(self.camera_log_file):
            with open(self.camera_log_file, 'w') as f:
                json.dump([],f)
            print(f"AUDIO_AGENT: Initialized empty camera log file at: {self.camera_log_file}")

        # `active_adk_sessions`: A dictionary to store active ADK session data per user.
        # Each user's session will have its own ADK runner, session, queue, and event stream.
        self.active_adk_sessions = {}

        # `camera_context`: Stores the latest information received from the CameraAgent.
        # This is used to prime the AudioAgent's conversations.
        self.camera_context = "No camera updates yet."

    def _log_transcription(self,speaker: str, text: str):
        """
        Internal helper method to log audio transcriptions to a JSON file.
        Each log entry includes a timestamp, the speaker ('user' or 'agent'), and the transcribed text.

        Args:
            speaker (str): The source of the speech (either "user" or "agent").
            text (str): The transcribed text of the speech.
        """
        timestamp = datetime.datetime.now().isoformat()
        log_entry = {"timestamp": timestamp, "speaker": speaker, "text": text}
        try:
            current_logs = []
            # CRobustly read, append, and write the JSON log file.
            if os.path.exists(self.audio_log_file) and os.path.getsize(self.audio_log_file) > 0:
                with open(self.audio_log_file, 'r') as f:
                    try:
                        current_logs = json.load(f)
                        if not isinstance(current_logs, list): current_logs = []
                    except json.JSONDecodeError: current_logs = [] # If file is corrupt, start fresh.
            current_logs.append(log_entry)
            with open(self.audio_log_file, 'w') as f:
                json.dump(current_logs, f, indent=4)
        except IOError as e: print(f'AUDIO_AGENT: Error writing to audio log: {e}')
        except Exception as e: print(f"AUDIO_AGENT: Unexpected error during audio logging: {e}")

    def update_camera_context(self, context: str):
        """
        Allows for external updating of the camera context.
        This is used by the main FastAPI application to pass information
        from the CameraAgent to this AudioAgent.
        """
        self.camera_context = context
        print(f"AUDIO_AGENT: Camera context manually updated: {context[:100]}...")

    def refresh_camera_context_from_log(self, max_entries=3):
        """
        Reads the `camera_log.json`, poentries should contain
        a comment from the LLM ("comment_by_llm").
        If relevant comments are found, `the current camera context to be used in future
        audio prompts.
        """
        try:
            # Check if camera log file exists and is not empty.
            if not os.path.exists(self.camera_log_file) or os.path.getsize(self.camera_log_file) == 0:
                # print(f'AUDIO_AGENT: Camera log missing or empty at {self.camera_log_file}.') # Can be noisy
                return
            with open(self.camera_log_file, 'r') as f:
                try:
                    all_camera_logs = json.load(f)
                    if not isinstance(all_camera_logs, list): all_camera_logs = []
                except json.JSONDecodeError:
                    # print(f'AUDIO_AGENT: Could not decode camera log: {self.camera_log_file}.') # Can be noisy
                    return
            if not all_camera_logs: return

            # Get the last `max_entries` from the log.
            recent_logs = all_camera_logs[-max_entries:]
            # Extract LLM comments. Get empty string as default if not found. Strip whitespace.
            relevant_comments = [log.get("comment_by_llm", "").strip() for log in recent_logs if isinstance(log.get("comment_by_llm"), str)]
            # Filter out any empty comments that resulted after stripping.
            relevant_comments = [comment for comment in relevant_comments if comment]

            if relevant_comments:
                new_context = " ".join(relevant_comments)
                if new_context != self.camera_context:
                    self.camera_context = new_context
                    print(f"AUDIO_AGENT: Camera context refreshed: {self.camera_context[:200]}...")
        except IOError as e: print(f"AUDIO_AGENT: Error reading camera log: {e}")
        except Exception as e: print(f"AUDIO_AGENT: Unexpected error refreshing camera context: {e}")

    async def start_adk_session(self, user_id: str, is_audio: bool = True):
        """
        Starts a new ADK session for a given user ID.
        If a requested session for the user.
        This also refreshes the `camera_context` from the log file before starting.
        Creates and stores an ADK runner, session, live request queue, and live events stream.
        Injects the current camera context as an initial message into the ADK agent.
        """
        if user_id in self.active_adk_sessions:
            print(f'AUDIO_AGENT: Session for user {user_id} already exists. Closing old one.')
            await self.stop_adk_session(user_id)

        self.refresh_camera_context_from_log()

        print(f"AUDIO_AGENT: Starting ADK session for {user_id}, audio: {is_audio}")
        try:
            # Create an InMemoryRunner. The `app_name` should be unique per session or user if
            # multiple runners are used, but for multie users served by one AudioAgent,
            # using user_id here is a good way to distinguish ADK's internal app needs.
            runner = InMemoryRunner(app_name=f'{APP_NAME}_{user_id}', agent=self.root_agent_adk)

            # Create an ADK session for the user.
            session = await runner.session_service.create_session(
                app_name=f'{APP_NAME}_{user_id}', # Matches the runner's app_name
                user_id=user_id
            )

            # Determine the response modality (audio or text).
            modality = "AUDIO" if is_audio else "TEXT"
            run_config = RunConfig(response_modalities=[modality])

            # Create a LiveRequestQueue for sending data to the agent.
            live_request_queue = LiveRequestQueue()

            # Construct an initial message to send to the agent to prime it with
            # the current camera context. This is sent as the first `user` turn.
            # The agent's instruction should guide it not to directly reply to this system message.
            initial_msg = (f"System Info: Visual context: '{self.camera_context}'. Please wait for user.")
            initial_content = Content(role="user", parts=[Part.from_text(initial_msg)])

            # Start the live ADK session, passing the initial context message.
            live_events = runner.run_live(
                session=session,
                live_request_queue=live_request_queue,
                run_config=run_config,
                request=initial_content  # Sends the priming message at session start.
            )

            # Store the active session components.
            self.active_adk_sessions[user_id] = {
                "runner": runner,
                "session": session,
                "live_request_queue": live_request_queue,
                "live_events": live_events,
                "is_audio": is_audio
            }
            print(f"AUDIO_AGENT: ADK session started for {user_id} with context: {self.camera_context[:100]}...")
            return live_events, live_request_queue
        except Exception as e:
            print(f'AUDIO_AGENT: Error starting ADK session for {user_id}: {e}')
            if user_id in self.active_adk_sessions: del self.active_adk_sessions[user_id]
            raise

    async def stop_adk_session(self, user_id: str):
        """
        Stops and cleans up an active ADK session for the specified user ID.
        """
        if user_id in self.active_adk_sessions:
            print(f"AUDIO_AGENT: Stopping ADK session for {user_id}...")
            # Get the live request queue to close it. Closing the queue is the
            # primary means of signaling the background ADK process to terminate.
            self.active_adk_sessions[user_id]["live_request_queue"].close()
            # Remove the session data from the active sessions dictionary.
            del self.active_adk_sessions[user_id]
            print(f"AUDIO_AGENT: ADK session stopped for {user_id}.")
        # Else clause isn't very necessary here, as it's not an error if a
        # it would just make the logs a bit noisier.
        # else: print(f"AUDIO_AGENT: No active ADK session for user {user_id} to stop.") # Can be noisy

    async def agent_to_client_sse_handler(self, user_id: str):
        """
        A FastAPI Server-Sent Events (SSE) handler that streams events from the ADK agent
        to the client browser. It acts as an asynchronous generator.
        This method is called by the FastAPI's SSE endpoint.
        It handles different types of events from ADK:
            - Turn completion or interruption.
            - Audio data (e.g., PCM).
            - Text data (partial or full transcriptions/responses).
        It logs agent's speech transcriptions.

        Args:
            user_id (str): The unique identifier of the user who se session is being handled.
        """
        if user_id not in self.active_adk_sessions:
            print(f"AUDIO_AGENT SSE: Session not found for user {user_id}.")
            yield f"data: {json.dumps({'error': 'Session not found.'})}\n\n"; return

        # Retrieve the live events stream and audio mode flag for the user's session.
        live_events = self.active_adk_sessions[user_id]["live_events"]
        is_audio = self.active_adk_sessions[user_id]["is_audio"]

        # `text_turn` accumulates all partial text responses for a single agent turn.
        text_turn = ""

        print(f"AUDIO_AGENT SSE: Starting event stream for user {user_id}.")
        async for event in live_events:
            # Handle events signaling the end of an agent turn.
            if event.turn_complete or event.interrupted:
                message = {"turn_complete": event.turn_complete,"interrupted": event.interrupted}
                yield f"data: {json.dumps(message)}\n\n"
                # If there was accumulated text, log it.
                if text_turn and "System Info:" not in text_turn:
                    self._log_transcription("agent", text_turn)
                text_turn = "" # Reset for the next turn.
                continue

            # Extract the first part of the content, if it exists.
            part = event.content and event.content.parts and event.content.parts[0]
            if not part: continue

            # Handle audio data if the session is in audio mode.
            if is_audio and part.inline_data and part.inline_data.mime_type.startswith("audio/"):
                if part.inline_data.data: # Ensure audio data exists
                    yield f"data: {json.dumps({'mime_type': part.inline_data.mime_type, 'data': base64.b64encode(part.inline_data.data).decode('ascii')})}\n\n"

            # Handle text data.
            if part.text:
                # Filter out the echo of the initial system information message.
                if "System Info:" in part.text and event.content.role == "model":
                    continue

                yield f"data: {json.dumps({'mime_type': 'text/plain', 'data': part.text})}\n\n"
                # Accumulate text for logging at the end of the turn.
                text_turn += part.text

        # print(f"AUDIO_AGENT SSE: Stream ended for {user_id}.") # Noisy for normal client disconnect

    async def client_to_agent_handler(self, user_id: str, client_msg: dict):
        """
        Handles messages sent from the client to the ADK agent.
        This is typically called from a FastAPI POST endpoint.
        It parses the client's message (audio or text), logs the user's transcription,
        and sends the data to the ADK agent via the live request queue.

        Args:
            user_id (str): The user's unique identifier.
            client_msg (dict): A dictionary containing 'mime_type' and 'data' from the client.

        Returns:
            dict: A status dictionary, or an error dictionary.
        """
        if user_id not in self.active_adk_sessions:
            print(f"AUDIO_AGENT MSG: Session not found for user {user_id} in client_to_agent_handler.")
            return {"error": "Session not found."}

        # The camera context is now primarily set at the start of the ADK session.
        # If it needs to be updated more frequently during a session, that
        # would require a different mechanism for injecting into an ongoing ADK run.
        # self.refresh_camera_context_from_log() # Context is now primarily set at session start

        queue = self.active_adk_sessions[user_id]["live_request_queue"]
        is_audio = self.active_adk_sessions[user_id]["is_audio"]
        mime, data = client_msg.get("mime_type"), client_msg.get("data")

        if not mime or data is None: return {"error": "Invalid message."}

        log_text = ""
        if mime == "text/plain":
            log_text = data
            # Send the text data directly. Context is primarily injected at session start.
            queue.send_content(Content(role="user", parts=[Part.from_text(data)]))
            print(f"AUDIO_AGENT MSG [CLIENT->AGENT] ({user_id}): Text: '{data[:50]}...'")
        elif mime.startswith('audio/') and is_audio:
            try:
                decoded = base64.b64decode(data)
                queue.send_realtime(Blob(data=decoded, mime_type=mime))
                print(f"AUDIO_AGENT MSG [CLIENT->AGENT] ({user_id}): {mime} {len(decoded)} bytes.")
            except Exception as e: return {"error": f"Audio error: {e}"}
        else: return {"error": "Unsupported mime/session."}

        if log_text: # Log user's text.
            self._log_transcription("user", log_text)

        return {"status": "sent"}

# This block allows for conceptual testing of the AudioAgent when the file is run directly.
if __name__ == '__main__':
    print("CAUDIO_AGENT: Conceptual test.")
    project, loc = os.getenv('GOOGLE_CLOUD_PROJECT'), os.getenv('GOOGLE_CLOUD_LOCATION')
    if os.getenv('GOOGLE_GENAI_USE_VERTEXAI','F').lower()=='t' and project and loc:
        agent = AudioAgent(project, loc)
        # agent.camera_log_file # This line seems incomplete, was it meant to be used?
        # Create a dummy camera log for testing context refresh
        with open(agent.camera_log_file, 'w') as f: json.dump([{"comment_by_llm": "Red car passed."},{"comment_by_llm": "Dog barked."}], f)
        print(f"Initial ctx: '{agent.camera_context}'.")
        agent.refresh_camera_context_from_log()
        print(f"Refreshed ctx: '{agent.camera_context}'") # Should be "Red car... Dog barked."
        async def main_test():
            uid = "test_uid_004"
            try:
                await agent.start_adk_session(uid, is_audio=False) # Test with text session
                await agent.client_to_agent_handler(uid, {"mime_type": "text/plain", "data": "What's new?"})
                # Add more test interactions here if needed
            finally: await agent.stop_adk_session(uid)
        asyncio.run(main_test())
        if os.path.exists(agent.camera_log_file): os.remove(agent.camera_log_file) # Clean up dummy log
        print("AUDIO_AGENT: Test done.")
    else: print("CAUDIO_AGENT: Skip test, Vertex AI env not set.")
