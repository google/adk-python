# Import necessary standard Python libraries
import os  # For interacting with the operating system (e.g., file paths)
import time  # For time-related functions (e.g., sleep, timestamps)
import json  # For working with JSON data (saving and loading logs)
import datetime  # For getting current date and time
import threading  # For running tasks in separate threads (non-blocking operations)

# Import libraries for Google Cloud Vertex AI
import vertexai  # Main Vertex AI SDK
import vertexai.generative_models as genai_models  # For using generative AI models and creating content parts
from vertexai.generative_models import GenerativeModel, Part #shorter alias

# Import custom utility for video monitoring
from .video_utils import VideoMonitor  # Assumes video_utils.py is in the same 'app' directory

# Import library for loading environment variables from a .env file
from dotenv import load_dotenv

# --- Environment Variable Loading ---
# Construct the absolute path to the .env file, assuming it's in the 'streaming-test2/' directory (parent of 'app/')
# __file__ is the path to the current script (camera_agent.py)
# os.path.dirname(__file__) gives the directory of the current script ('app/')
# os.path.dirname(os.path.dirname(__file__)) gives the parent directory ('streaming-test2/')
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
# Load the .env file, making its variables available as environment variables
load_dotenv(dotenv_path=dotenv_path)

class CameraAgent:
    """
    The CameraAgent is responsible for monitoring a video feed from a camera,
    detecting significant visual changes, sending these changes (as images)
    to a generative AI model for interpretation, and logging the outcomes.
    It runs its monitoring process in a separate background thread.
    """

    def __init__(self,
                 project_id: str,
                 location: str,
                 model_name: str = "gemini-2.0-flash",
                 camera_index: int = 0,
                 fps_limit: int = 1):
        """
        Initializes the CameraAgent.

        Args:
            project_id (str): The Google Cloud Project ID.
            location (str): The Google Cloud region for Vertex AI services.
            model_name (str, optional): The name of the Gemini model to use for image interpretation.
                                        Defaults to "gemini-2.0-flash".
            camera_index (int, optional): The index of the camera to use (e.g., 0 for the default system camera).
                                          Defaults to 0.
            fps_limit (int, optional): The maximum frames per second to process from the camera.
                                       Defaults to 1.
        """
        # --- Validate Essential Configuration ---
        if not project_id:
            raise ValueError("CAMERA_AGENT: Google Cloud Project ID is required for CameraAgent.")
        if not location:
            raise ValueError("CAMERA_AGENT: Google Cloud Location is required for CameraAgent.")

        # --- Store Configuration ---
        self.project_id = project_id
        self.location = location
        self.model_name = model_name

        # Define the path for the camera log file
        # It will be stored in 'streaming-test2/logs/camera_log.json'
        self.log_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),  # 'streaming-test2/'
            'logs',  # 'logs/'
            'camera_log.json'  # 'camera_log.json'
        )

        # --- Initialize Logging ---
        # Ensure the 'logs' directory exists; create it if it doesn't.
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        # Initialize the log file with an empty JSON list '[]'.
        # This ensures the file is always a valid JSON, even if empty.
        # 'w' mode overwrites the file if it exists, effectively clearing old logs on startup.
        with open(self.log_file, 'w') as f:
            json.dump([], f)

        # --- Initialize Vertex AI ---
        # This print statement helps confirm that the agent is starting up and using the correct project/location.
        print(f"CAMERA_AGENT: Initializing Vertex AI with project: {self.project_id}, location: {self.location}")
        # Initialize the Vertex AI SDK with the provided project and location.
        # This needs to be done before using any Vertex AI services.
        vertexai.init(project=self.project_id, location=self.location)

        print(f"CAMERA_AGENT: Loading generative model: {self.model_name}")
        # Create an instance of the GenerativeModel class using the specified model name.
        self.model = GenerativeModel(self.model_name)
        # `self.chat` will store the chat session object with the model.
        # It's initialized to None and will be created when the first message needs to be sent.
        self.chat = None

        # --- Initialize Video Monitoring ---
        # Create an instance of VideoMonitor from our video_utils.py.
        # This object will handle the actual camera interaction and frame processing.
        self.video_monitor = VideoMonitor(camera_index=camera_index, fps_limit=fps_limit)

        # --- Threading Control ---
        # `self.running` is a boolean flag that controls the main monitoring loop in the background thread.
        # Setting it to False will signal the thread to stop.
        self.running = False
        # `self.thread` will hold the Thread object once it's created.
        self.thread = None

        print(f"CAMERA_AGENT: Initialized. Logging to: {self.log_file}")
        print(f"CAMERA_AGENT: Video monitor configured for camera index {camera_index}, FPS limit {fps_limit}.")

    def _start_chat_session(self):
        """
        Internal helper method to initialize or re-initialize a chat session with the Vertex AI model.
        This is called if a chat session is not active when needed.
        Achat session allows for conversational context to be maintained with the model.
        """
        # If a chat session doesn't exist or has been closed (e.g., due to an error)
        if not self.chat:
            print("CAMERA_AGENT: Starting new chat session with Vertex AI model...")
            try:
                # Start a new chat with the loaded model.
                # `response_validation=False` can sometimes be useful for models that might return non-standard responses,
                # though for typical text/image inputs, it's often not strictly necessary.
                self.chat = self.model.start_chat(response_validation=False)
                print("CAMERA_AGENT: Chat session started successfully.")
            except Exception as e:
                # If starting the chat session fails, print an error and ensure `self.chat` remains None.
                print(f"CAMERA_AGENT: Critical error starting chat session: {e}")
                self.chat = None
                # Re-raise the exception because failing to start a chat session is a critical issue for this agent.
                raise

    def _log_change(self, description: str, model_comment: str):
        """
        Internal helper method to log detected changes and the AI model's comments to the JSON log file.
        Each log entry is a JSON object containing a timestamp, type of event, the CV's description,
        and the language model's comment.

        Args:
            description (str): The description of the change detected by the computer vision algorithm.
            model_comment (str): The interpretation or comment provided by the AI model.
        """
        # Get the current time in ISO 8601 format (e.g., "2023-10-27T10:30:00.123456")
        timestamp = datetime.datetime.now().isoformat()
        # Prepare the data to be logged as a Python dictionary.
        log_entry = {
            "timestamp": timestamp,
            "type": "video_change",  # Helps categorize log entries if other types are added later.
            "description_by_cv": description,  # Description from the local CV processing.
            "comment_by_llm": model_comment      # Comment generated by the language model.
        }

        try:
            # --- Read existing logs, append new entry, and write back ---
            # This read-modify-write approach is simple. For very high-frequency logging,
            #  more advanced techniques like a dedicated logging queue or database might be better
            #  to avoid I/O bottlenecks or race conditions. For this project's expected rate of
            #  significant visual changes, this method is generally acceptable.

            current_logs = []
            # Check if the log file already exists and is not empty.
            if os.path.exists(self.log_file) and os.path.getsize(self.log_file) > 0:
                # Open the log file in read mode ('r').
                with open(self.log_file, 'r') as f:
                    try:
                        # Attempt to load the existing log entries from the JSON file.
                        current_logs = json.load(f)
                        # Ensure the loaded content is a list, as expected.
                        # If the file was corrupted and contained something else (e.g., a single dictionary),
                        #  reset to an empty list to prevent errors when appending.
                        if not isinstance(current_logs, list):
                            print(f"CAMERA_AGENT: Warning - camera_log.json content was not a list. Resetting log.")
                            current_logs = []
                    except json.JSONDecodeError:
                        # If the JSON is invalid (e.g., file corrupted or improperly formatted),
                        #  print a warning and start with an empty list. This prevents the agent from crashing.
                        print(f"CAMERA_AGENT: Warning - camera_log.json was corrupted. Starting with a new log list.")
                        current_logs = []

            # Add the new log entry to the list of logs (either newly created or loaded).
            current_logs.append(log_entry)

            # Open the log file in write mode ('w'). This will overwrite the existing file.
            with open(self.log_file, 'w') as f:
                # Write the entire updated list of logs back to the file.
                # `indent=4` formats the JSON output to be human-readable with pretty printing.
                json.dump(current_logs, f, indent=4)

            # Print a confirmation that the change was logged, showing the first 50 characters of the comment.
            print(f"CAMERA_AGENT: Logged change. LLM comment: '{model_comment[:50]}...'")
        except IOError as e:
            # Handle potential errors during file input/output operations (e.g., disk full, permissions).
            print(f"CAMERA_AGENT: Error writing to log file {self.log_file}: {e}")
        except Exception as e:
            # Catch any other unexpected errors during the logging process.
            print(f'CAMERA_AGENT: Unexpected error during logging: {e}')

    def _monitor_loop(self):
        """
        The main loop for monitoring the video feed. This method is intended to be run in a separate thread.
        It continuously captures frames using `VideoMonitor`, processes them for significant changes,
        sends relevant frames to the AI model for interpretation, and logs the results.
        The loop continues as long as the `self.running` flag is True.
        """
        print("CAMERA_AGENT: Video monitoring loop started.")

        # Attempt to start video capture using the VideoMonitor instance.
        # This initializes the camera.
        if not self.video_monitor.start_capture():
            # If video capture cannot be started (e.g., camera not found, permissions error),
            #  print a critical error message and exit the loop.
            print("CAMERA_AGENT: CRITICAL - Failed to start video capture. Monitoring loop cannot run.")
            self.running = False  # Set running flag to False to ensure the loop doesn't try to run.
            return  # Exit the method.

        # This is the main operational loop of the CameraAgent.
        while self.running:
            try:
                # `process_frame_for_changes` is a method in `VideoMonitor` (from `video_utils.py`).
                # It handles capturing a frame, comparing it to previous frames (as or a background model)
                #  to detect significant changes, and applying any FPS limits.
                # - `changed` (bool): True if a significant change was detected, False otherwise.
                # - `description` (str): A textual description of the change from the CV algorithm (e.g., "Motion detected in region X").
                # - `frame_bytes` (bytes | None): The raw image data (e.g., JPEG encoded) if a change was detected and a frame is available, otherwise None.
                changed, description, frame_bytes = self.video_monitor.process_frame_for_changes()

                # If a change was detected AND animage data (frame_bytes) is available:
                if changed and frame_bytes:
                    print(f"CAMERA_AGENT: Video change detected by CV: {description}. Preparing image for model.")

                    # Ensure an active chat session with the AI model exists.
                    # If not, try to start one.
                    if not self.chat:
                        try:
                            self._start_chat_session()
                        except Exception as e:
                            # If starting the chat session fails, log the error and skip processing this change.
                            # This prevents the agent from getting stuck if the model is temporarily unavailable.
                            print(f"CAMERA_AGENT: Could not start chat session for processing change: {e}. Skipping this event.")
                            time.sleep(5)  # Wait for a few seconds before trying again in the next loop iteration.
                            continue  # Skip the rest of this iteration and go to the next.

                    # Prepare the image data for the AI model.
                    # The Gemini model expects multimodal inputs (like images) to be wrapped in a `Part` object.
                    image_part = Part.from_data(frame_bytes, mime_type="image/jpeg")  # Assuming JPEG format from VideoMonitor

                    # Construct a detailed and specific prompt for the AI model.
                    # A good prompt helps the model understand the context and provide relevant interpretations.
                    prompt_text = (
                        "You are an AI assistant observing a live video feed from a security camera. "
                        "A computer vision algorithm has detected a significant visual change in the scene. "
                        f"The algorithm's initial description of this change is: '{description}'. "
                        "Please analyze the provided image carefully and give a concise, human-readable description "
                        "of what you observe in the image that is relevant to this detected change. "
                        "Focus on new objects, significant movements, changes in object states, or anything unusual. "
                        "Avoid generic statements like 'I see an image.' Be specific."
                    )

                    print(f"CAMERA_AGENT: Sending video change event and image to model. Prompt: '{prompt_text[:100]}...'")

                    try:
                        # Send the prompt (text) and the image part to the AI model via the active chat session.
                        # The model will process both and generate a response.
                        # For this agent, we are expecting a dsingle, non-streamed response.
                        response = self.chat.send_message([prompt_text, image_part])

                        model_comment = "" # Initialize to an empty string.
                        # Safely extract the text response from the model.
                        # The response object can be complex; we need to navigate to the actual text part.
                        # `response.candidates[0]` usually contains the primary response.
                        # `content.parts[0]` usually contains the main content block.
                        # `.text` gives the textual content.
                        if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                            model_comment = response.candidates[0].content.parts[0].text or ""  # Use .text, default to "" if None or empty

                        if model_comment.strip():  # Check if the comment is not empty or just whitespace.
                            print(f"CAMERA_AGENT: Model provided comment: '{model_comment}'")
                            # Log the original CV description and the model's interpretation.
                            self._log_change(description, model_comment)
                        else:
                            print("CAMERA_AGENT: Model did not provide a text comment for the video change.")
                            # Optionally, log that no comment was provided if that's important information.
                            # self._log_change(description, "LLM provided no comment or an empty comment.")

                    except Exception as e:
                        # Handle errors that might occur during communication with the AI model (e.g., network issues, API errors).
                        print(f"CAMERA_AGENT: Error sending message to model or processing its response: {e}")
                        # It's often a good practice to reset the chat session if an error occurs,
                        #  as the session state might have become invalid.
                        self.chat = None
                        print("CAMERA_AGENT: Chat session reset due to error.")
                        time.sleep(2)  # Brief pause bnext loop iteration to prevent rapid error cycling.

                elif changed:
                    # A change was detected by the CV algorithm, but `frame_bytes` is None (e.g., an error occurred during frame encoding).
                    # We can't send anything to the model without the image.
                    print(f"CAMERA_AGENT: Video change detected ('{description}'), but frame data is unavailable. Cannot send to model.")
                    # Optionally, log this situation.
                    # self._log_change(description, "CV detected change, but frame data was not available for LLM.")

                # --- Loop Pacing ---
                # The `video_monitor.process_frame_for_changes()` method in `video_utils.py`
                #  should already incorporate logic (like `time.sleep`) to respect the `fps_limit`.
                #  However, adding a very small sleep here ensures that this `_monitor_loop`
                #  yields execution control, preventing it from becoming a "tight loop" that consumes
                #   100%CPU if `process_frame_for_changes` returns very quickly (e.g., if fps_limit is high or None).
                time.sleep(0.1)  # 100ms. Adjust as needed based on system performance and desired responsiveness.

            except Exception as e:
                # Catch any other unexpected errors that might occur within the main monitoring loop.
                print(f'CAMERA_AGENT: Unexpected error in monitor loop: {e}')
                # Depending on the severity or type of error, one might implement more sophisticated error handling,
                #  such as an exponential backoff for retries, or stopping the agent if errors persist.
                time.sleep(5) # Wait a bit to avoid overwhelming logs or services.

        # --- Loop End & Cleanup ---
        # This section is reached when `self.running` becomes `False`.
        # It's important to clean up sresources, especially the camera.
        if self.video_monitor.cap:  # Check if the video capture object exists and might be open.
            self.video_monitor.stop_capture()  # Call the method in VideoMonitor to release the camera.
        print("CAMERA_AGENT: Video monitoring loop stopped.")

    def start(self):
        """
        Starts the camera agent's monitoring process.
        It creates and starts a new background thread that runs the `_monitor_loop` method.
        If the agent is already running, this method prints a message and does nothing more.
        """
        if not self.running:
            self.running = True  # Set the flag to True to allow the `_monitor_loop` to execute.
            # Create a new Thread object.
            # - `target=self._monitor_loop`: Specifies the function to be executed by the thread.
            # - `daemon=True`: Makes the thread a daemon thread. Daemon threads automatically exit
            #   when the main program (the one that started them) exits. This is useful for background tasks.
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()  # Start the execution of the thread.
            print("CAMERA_AGENT: Agent started monitoring in a background thread.")
        else:
            print("CAMERA_AGENT: Agent is already running.")

    def stop(self):
        """
        Stops the camera agent's monitoring process.
        It sets the `self.running` flag to False, which signals the `_monitor_loop` in the
        background thread to terminate its current iteration and exit.
        It then waits for the thread to finish using `thread.join()`.
        """
        if self.running:
            print("CAMERA_AGENT: Attempting to stop agent...")
            self.running = False  # Signal the `_monitor_loop` to stop.

            if self.thread and self.thread.is_alive():
                # `self.thread.join(timeout=10)` will wait for the background thread to complete.
                # The timeout (e.g., 10 seconds) is a safeguard in case the thread gets stuck
                #  and doesn't terminate cleanly.
                self.thread.join(timeout=10)
                if self.thread.is_alive():
                    # If the thread is still alive after the timeout, it indicates a problem.
                    print("CAMERA_AGENT: Warning - background thread did not stop in the allocated time (10 seconds).")
                else:
                    print("CAMERA_AGENT: Background thread stopped successfully.")
            else:
                # This case handles if the thread was not active or already stopped.
                print("CAMERA_AGENT: Background thread was not active or already stopped.")

            self.thread = None # Clear the reference to the thread object.
            print("CAMERA_AGENT: Agent stopped.")
        else:
            print("CAMERA_AGENT: Agent is not currently running.")

# --- Main execution block for direct testing (optional) ---
# This `if __name__ == '__main__':` block allows the script to be run directly
# (e.g., by executing `python camera_agent.py` from the terminal).
# It's a common Python idiom for providing test code or a simple command-line interface for a module.
if __name__ == '__main__':
    print("CAMERA_AGENT: Running camera_agent.py for direct testing...")

    # Load required environment variables (PROJECT_ID, LOCATION) from the .env file.
    # `load_dotenv()` should have been called at the top of the script.
    project_id_env = os.getenv('GOOGLE_CLOUD_PROJECT')
    location_env = os.getenv('GOOGLE_CLOUD_LOCATION')
    # Check if Vertex AI is configured to be used.
    use_vertex_env = os.getenv('GOOGLE_GENAI_USE_VERTEXAI', 'False').lower() == 'true'

    # Proceed only if all necessary environment variables are set.
    if use_vertex_env and project_id_env and location_env:
        print(f"CAMERA_AGENT MAIN: Initializing agent with project='{project_id_env}', location='{location_env}', model='gemini-2.0-flash'")
        # Create an instance of the CameraAgent for testing purposes.
        test_agent = CameraAgent(
            project_id=project_id_env,
            location=location_env,
            model_name="gemini-2.0-flash",  # Can be overridden here if a different model is needed for testing.
            camera_index=0,  # Use the default system camera (usually index 0).
            fps_limit=0.2  # Set a low FPS for testing (1 frame every 5 seconds) to easily observe behavior.
        )

        print("CAMERA_AGENT MAIN: Starting agent for testing...")
        test_agent.start()  # Start the agent's monitoring thread.

        try:
            print("CAMERA_AGENT MAIN: Agent running. Monitoring for video changes for approximately 20 seconds.")
            print(f"CAMERA_AGENT MAIN: Check the log file '{test_agent.log_file}' for output.")
            # Keep the main thread (this script) alive for a period to allow the background agent thread to run.
            time.sleep(20)
        except KeyboardInterrupt:
            # Allow the user to stop the test prematurely by pressing Ctrl+C in the terminal.
            print("CAMERA_AGENT MAIN: Test interrupted by user (KeyboardInterrupt).")
        finally:
            # This `finally` block ensures that the agent is stopped cleanly,
            #  regardless of whether the test completes normally or is interrupted.
            print("CAMERA_AGENT MAIN: Stopping agent...")
            test_agent.stop()  # Stop the agent's monitoring thread and release resources.
            print("CAMERA_AGENT MAIN: Direct test finished.")
    else:
        # If required environment variables not set, print an error message and skip the test.
        print("CAMERA_AGENT MAIN: CRITICAL - Environment variables for Vertex AI "
              "(GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION) or GOOGLE_GENAI_USE_VERTEXAI=True "
              "not set correctly in .env file.")
        print(f"  Loaded Project ID: {project_id_env}")
        print(f"  Loaded Location: {location_env}")
        print(f"  Use Vertex AI: {use_vertex_env}")
        print(f"  The .env file was expected at: {dotenv_path}")
        print("CAMERA_AGENT MAIN: Skipping direct agent test.")
