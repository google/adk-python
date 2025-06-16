import sys
import os
import time # Added time for agent_background_tasks
import threading # For agent's background tasks

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.web_interface import web_app, get_agent, agent_event_queue # Import Flask app and agent instance
from app.config import WEB_HOST, WEB_PORT
# We might need to import agent parts if we run its loop here

# --- Agent Background Processing Loop ---
# This is a conceptual loop. The actual implementation of how the agent
# processes audio/video continuously and interacts with the model needs refinement.
# It should place results/events onto the agent_event_queue for the web UI.

agent_stop_event = threading.Event()

def agent_background_tasks():
    print("Agent background task thread started.")
    agent = get_agent() # Initialize/get the agent instance
    if not agent:
        print("Agent not initialized, background tasks cannot run.")
        return

    # Ensure agent is "active" (chat started, video capture on)
    # This might be better controlled via web UI (start/stop buttons)
    # For now, let's assume if background tasks run, agent should be active.
    # However, start_chat() initializes the Vertex AI chat, which costs.
    # So, let's make it start only on web request.
    # The video feed itself will start capture if needed.
    # This background task will primarily focus on non-web-request-driven events.

    # if not agent.chat:
    #    agent.start_chat() # This also starts video capture

    last_audio_check_time = time.time()
    last_video_check_time = time.time()

    while not agent_stop_event.is_set():
        current_time = time.time()

        # 1. Check for video changes (if video is active)
        if agent.video_monitor and agent.video_monitor.cap and agent.video_monitor.cap.isOpened():
            if current_time - last_video_check_time > 2: # Check every 2 seconds (adjust as needed)
                # print("Background: Checking for video changes...") # Can be noisy
                # The check_for_video_changes_and_comment method itself might be slow
                # if it involves a model call.
                # It needs to be adapted to put its findings on agent_event_queue.

                # Simplified: Assume agent.check_for_video_changes_and_comment() is modified
                # to return the comment, which we then put on the queue.
                # Or, it directly puts on the queue. Let's assume the latter for cleanliness.

                # --- Modification for agent.py's check_for_video_changes_and_comment ---
                # It should look like:
                # if changed and frame_bytes:
                #    ... model_response = self.send_message(...)
                #    if model_response:
                #        agent_event_queue.put({"type": "video_change_comment", "comment": model_response})
                # -------------------------------------------------------------------------
                try:
                    # This call will now internally try to put to agent_event_queue if a change is commented on by model
                    agent.check_for_video_changes_and_comment(event_queue=agent_event_queue)
                except Exception as e:
                    print(f"Error in background video check: {e}")
                last_video_check_time = current_time

        # 2. Placeholder for continuous audio processing (if we were doing live mic streaming)
        # For example, if agent had a method like `process_live_audio_chunk()`
        # that transcribed and potentially sent to model, and put results on queue.
        # if agent.is_listening_for_voice and (current_time - last_audio_check_time > 0.1): # Process frequently
        #     transcription, model_response_text = agent.process_live_audio_chunk()
        #     if transcription:
        #         agent_event_queue.put({"type": "transcription", "text": transcription})
        #     if model_response_text:
        #         agent_event_queue.put({"type": "model_response_audio_text", "text": model_response_text})
        #         # Synthesize and play audio would happen on client or if agent has speakers
        #     last_audio_check_time = current_time


        time.sleep(0.5) # Loop frequency for background checks

    print("Agent background task thread stopped.")


if __name__ == "__main__":
    print("Starting application via run.py...")

    # Initialize agent (important to do this before starting Flask or background thread)
    print("Initializing agent instance...")
    agent = get_agent()
    if not agent:
        print("CRITICAL: Agent failed to initialize. Web interface may not work correctly.")
        # sys.exit(1) # Optionally exit if agent is critical for any operation

    # Start agent's background processing tasks in a separate thread
    # These tasks would handle things like continuous video monitoring for changes,
    # or live audio stream processing if implemented.
    print("Starting agent background tasks thread...")
    background_thread = threading.Thread(target=agent_background_tasks, daemon=True)
    background_thread.start()

    print(f"Starting Flask web server on {WEB_HOST}:{WEB_PORT}...")
    # Use threaded=True for Flask dev server to handle concurrent requests (e.g. video feed + SSE)
    # For production, use a proper WSGI server like Gunicorn or Waitress.
    web_app.run(host=WEB_HOST, port=WEB_PORT, debug=False, threaded=True, use_reloader=False)

    # Signal the background thread to stop when Flask server exits
    print("Flask server stopped. Signaling agent background tasks to stop...")
    agent_stop_event.set()
    background_thread.join(timeout=5) # Wait for thread to finish
    print("Application shut down.")
