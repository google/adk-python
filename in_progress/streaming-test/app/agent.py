import os
import time # Required for sleep in __main__
import vertexai
from vertexai.generative_models import GenerativeModel, Part

from .audio_utils import transcribe_audio_bytes, synthesize_text_to_audio_bytes, play_audio_bytes, record_audio
from .video_utils import VideoMonitor
from dotenv import load_dotenv # Added for direct run in __main__

class MultimodalAgent:
    def __init__(self, project_id: str, location: str, model_name: str, camera_index: int = 0):
        if not project_id:
            raise ValueError("Google Cloud Project ID is required.")
        if not location:
            raise ValueError("Google Cloud Location is required.")
        if not model_name:
            raise ValueError("Model name is required.")

        self.project_id = project_id
        self.location = location
        self.model_name = model_name

        vertexai.init(project=self.project_id, location=self.location)
        self.model = GenerativeModel(self.model_name)
        self.chat = None

        self.video_monitor = VideoMonitor(camera_index=camera_index, fps_limit=1)

        print(f"AGENT: MultimodalAgent initialized with project: {self.project_id}, location: {self.location}, model: {self.model_name}")
        print(f"AGENT: Video monitor configured for camera index {camera_index}.")

    def start_chat(self):
        print("AGENT: Attempting to start chat session...")
        self.chat = self.model.start_chat(response_validation=False)
        print("AGENT: New chat session started with Vertex AI.")
        if not self.video_monitor.cap:
             if self.video_monitor.start_capture():
                 print("AGENT: Video capture started successfully.")
             else:
                 print("AGENT: Failed to start video capture.")
        return self.chat

    def stop_chat(self):
        print("AGENT: Attempting to stop chat session...")
        if self.video_monitor.cap:
            self.video_monitor.stop_capture()
            # print("AGENT: Video capture stopped.") # Covered by video_monitor's own print
        self.chat = None
        print("AGENT: Chat session ended.")

    def send_text_message(self, text_prompt: str) -> str:
        if not self.chat:
            print("AGENT: Chat not started in send_text_message. Starting chat...")
            self.start_chat()

        print(f"AGENT: Sending text to model: '{text_prompt}'")
        try:
            response = self.chat.send_message(text_prompt)
            # Ensure response.text is accessed safely
            response_text = ""
            if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                 response_text = response.candidates[0].content.parts[0].text if response.candidates[0].content.parts[0].text else ""

            print(f"AGENT: Received text response from model: '{response_text[:100]}...'")
            return response_text
        except Exception as e:
            print(f"AGENT: Error sending text message to model: {e}")
            raise

    def handle_voice_interaction(self, record_duration_seconds: int = 5, event_queue=None):
        print(f"AGENT: --- Starting Voice Interaction Cycle (Duration: {record_duration_seconds}s) ---")
        if not self.chat:
            print("AGENT: Chat not started in handle_voice_interaction. Starting chat...")
            self.start_chat()

        # 1. Record audio
        print("AGENT: Calling record_audio...")
        user_audio_bytes = record_audio(duration_seconds=record_duration_seconds)
        if not user_audio_bytes:
            print("AGENT: Audio recording failed or returned no data.")
            if event_queue: event_queue.put({"type": "status", "message": "Audio recording failed."})
            return
        print(f"AGENT: record_audio successful, {len(user_audio_bytes)} bytes recorded.")

        # 2. Transcribe audio
        print("AGENT: Calling transcribe_audio_bytes...")
        user_text_prompt = transcribe_audio_bytes(user_audio_bytes)
        if user_text_prompt is None:
            print("AGENT: Audio transcription failed (returned None).")
            if event_queue: event_queue.put({"type": "status", "message": "Transcription failed."})
            return
        if not user_text_prompt: # Empty string if no speech detected
            print("AGENT: No speech detected in the recording (transcription returned empty string).")
            if event_queue: event_queue.put({"type": "status", "message": "No speech detected."})
            return
        print(f"AGENT: User (transcribed): '{user_text_prompt}'")
        if event_queue:
            print(f"AGENT: Putting transcription to event_queue: '{user_text_prompt}'")
            event_queue.put({"type": "transcription", "text": user_text_prompt})

        # 3. Send transcribed text to model
        print(f"AGENT: Calling send_text_message with transcribed text: '{user_text_prompt}'")
        model_response_text = self.send_text_message(user_text_prompt)
        if model_response_text is None : # Explicitly check for None, empty string is a valid response
            print("AGENT: Model did not return a text response (send_text_message returned None).")
            if event_queue: event_queue.put({"type": "status", "message": "Model did not respond."})
            return

        print(f"AGENT: Model (text response): '{model_response_text}'")
        if event_queue:
            print(f"AGENT: Putting model text response to event_queue: '{model_response_text}'")
            event_queue.put({"type": "model_response_audio_text", "text": model_response_text})

        # 4. Synthesize model's text response to audio
        if not model_response_text.strip(): # Don't synthesize if empty or only whitespace
            print("AGENT: Model response was empty, skipping audio synthesis.")
        else:
            print("AGENT: Calling synthesize_text_to_audio_bytes...")
            model_audio_bytes = synthesize_text_to_audio_bytes(model_response_text)
            if not model_audio_bytes:
                print("AGENT: Model audio synthesis failed.")
            else:
                print(f"AGENT: synthesize_text_to_audio_bytes successful, {len(model_audio_bytes)} bytes synthesized.")
                # 5. Play back synthesized audio (plays on server where Python runs)
                print("AGENT: Calling play_audio_bytes (server-side playback)...")
                play_audio_bytes(model_audio_bytes, output_format="mp3")

        print("AGENT: --- Voice Interaction Cycle Ended ---")


    def check_for_video_changes_and_comment(self, event_queue=None):
        if not self.video_monitor.cap or not self.video_monitor.cap.isOpened():
            if self.chat:
                print("AGENT: Video monitor not active, attempting to restart for change detection...")
                self.video_monitor.start_capture()
                if not self.video_monitor.cap or not self.video_monitor.cap.isOpened():
                    # print("AGENT: Video capture is not active. Cannot check for changes.") # Can be noisy
                    return
            else:
                return

        changed, description, frame_bytes = self.video_monitor.process_frame_for_changes()

        if changed and frame_bytes:
            print(f"AGENT: Video change detected: {description}. Preparing to send to model.")

            if not self.chat:
                print("AGENT: Chat not started in check_for_video_changes. Starting chat...")
                self.start_chat()

            image_part = Part.from_data(frame_bytes, mime_type="image/jpeg")
            prompt_text = f"A change was detected in the video feed: {description}. Concisely describe what you observe in this image related to the change."

            print(f"AGENT: Sending video change event to model: '{prompt_text}'")
            model_response = self.send_message(text_prompt=prompt_text, image_parts=[image_part])

            if model_response: # Includes empty string
                print(f"AGENT: Model commented on video change: '{model_response}'")
                if event_queue:
                    print(f"AGENT: Putting video change comment to event_queue: '{model_response}'")
                    event_queue.put({"type": "video_change_comment", "comment": model_response})
            else: # Only if model_response is None
                print("AGENT: Model did not provide a comment for the video change (send_message returned None).")
        elif changed:
            print(f"AGENT: Video change detected ({description}), but frame data is unavailable.")

    def send_message(self, text_prompt: str = None, image_parts: list = None, audio_parts: list = None, video_parts: list = None) -> str | None:
        if not self.chat:
            print("AGENT: Chat not started in send_message. Starting chat...")
            self.start_chat()

        if not text_prompt and not image_parts and not audio_parts and not video_parts:
            print("AGENT: send_message called with no content.")
            # raise ValueError("At least one input type (text, image, audio, video) must be provided.")
            return None # Return None instead of raising error to prevent crash

        prompt_parts = []
        if text_prompt:
            prompt_parts.append(Part.from_text(text_prompt))

        if image_parts: prompt_parts.extend(image_parts)
        if audio_parts: prompt_parts.extend(audio_parts)
        if video_parts: prompt_parts.extend(video_parts)

        part_types_summary = []
        for p in prompt_parts:
            part_dict = p.to_dict() if hasattr(p, 'to_dict') else {}
            if part_dict.get('text'): # Check if it's a text part
                 part_types_summary.append('text')
            elif part_dict.get('inline_data'): # Check if it's an inline_data part
                 part_types_summary.append(f"inline_data:{part_dict['inline_data'].get('mime_type', 'unknown')}")
            else:
                 part_types_summary.append('unknown_part_type') # Should ideally not happen with current Part creation
        print(f"AGENT: Sending parts to model: {part_types_summary}")

        try:
            response = self.chat.send_message(prompt_parts, stream=False)
            # Safe access to response text
            response_text = ""
            if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                 response_text = response.candidates[0].content.parts[0].text if response.candidates[0].content.parts[0].text else ""

            print(f"AGENT: Received raw response from model. Text content: '{response_text[:100]}...'")
            return response_text
        except Exception as e:
            print(f"AGENT: Error sending multimodal message to model: {e}")
            # Potentially return None or re-raise specific error types
            return None


if __name__ == '__main__':
    print("Attempting to run agent.py for direct testing...")
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    print(f"AGENT MAIN: Loading .env from {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)

    use_vertex_env = os.getenv('GOOGLE_GENAI_USE_VERTEXAI', 'False').lower() == 'true'
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    location = os.getenv('GOOGLE_CLOUD_LOCATION')
    model_name_config = "gemini-2.0-flash"

    if use_vertex_env and project and location:
        print(f"AGENT MAIN: Initializing agent with project='{project}', location='{location}', model='{model_name_config}'")
        agent = MultimodalAgent(
            project_id=project,
            location=location,
            model_name=model_name_config,
            camera_index=0
        )
        print("AGENT MAIN: Starting chat...")
        agent.start_chat()

        if not agent.video_monitor.cap or not agent.video_monitor.cap.isOpened():
            print("AGENT MAIN: Video capture failed to initialize. Video tests will be skipped.")
        else:
            print("AGENT MAIN: Video capture active. Monitoring for video changes for ~5 seconds for test.")
            start_time = time.time()
            try:
                while time.time() - start_time < 5:
                    agent.check_for_video_changes_and_comment()
                    time.sleep(1)
            except KeyboardInterrupt:
                print("AGENT MAIN: Test interrupted.")

        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            print("\nAGENT MAIN: Attempting a voice interaction cycle (will try to record for 3s).")
            print("AGENT MAIN: Please speak into your microphone when 'Recording audio...' appears.")
            agent.handle_voice_interaction(record_duration_seconds=3)
        else:
            print("\nAGENT MAIN: GOOGLE_APPLICATION_CREDENTIALS not set. Skipping direct voice interaction test.")
            print("AGENT MAIN: Testing text-only interaction instead.")
            response_text = agent.send_text_message("Hello! This is a text-only direct test from agent.py.")
            print(f"AGENT MAIN: Model Response: {response_text}")

        print("AGENT MAIN: Stopping chat...")
        agent.stop_chat()
        print("AGENT MAIN: Direct test finished.")
    else:
        print("AGENT MAIN: Environment variables for Vertex AI not set. Skipping agent direct test.")
