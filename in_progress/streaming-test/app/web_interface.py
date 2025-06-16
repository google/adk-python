from flask import Flask, render_template, Response, request, jsonify, stream_with_context
import time
import json # For SSE
from app.agent import MultimodalAgent # Assuming agent is in app.agent
from app.config import WEB_HOST, WEB_PORT, MODEL_NAME # Assuming these are in config
import os
from dotenv import load_dotenv
import queue # For SSE messages

# --- Global Agent Initialization ---
# Load .env file from the parent directory of 'app'
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=dotenv_path)

project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
location = os.getenv("GOOGLE_CLOUD_LOCATION")
use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "False").lower() == "true"

agent_instance = None
agent_event_queue = queue.Queue() # Queue for SSE messages from agent

def get_agent():
    global agent_instance
    if agent_instance is None:
        if use_vertex and project_id and location:
            print(f"WEB_INTERFACE: Initializing Agent with project: {project_id}, loc: {location}, model: {MODEL_NAME}")
            agent_instance = MultimodalAgent(
                project_id=project_id,
                location=location,
                model_name=MODEL_NAME,
                camera_index=0
            )
        else:
            print("WEB_INTERFACE: Agent cannot be initialized. Check .env settings.")
    return agent_instance

# --- Flask App Setup ---
web_app = Flask(__name__)

def generate_video_frames():
    local_agent = get_agent()
    if not local_agent or not local_agent.video_monitor:
        print("WEB_INTERFACE: Video frames - Agent or video monitor not available.")
        return

    if not local_agent.video_monitor.cap or not local_agent.video_monitor.cap.isOpened():
        print("WEB_INTERFACE: Video frames - Camera not started. Attempting to start.")
        if not local_agent.video_monitor.start_capture():
            print("WEB_INTERFACE: Video frames - Failed to start camera for web feed.")
            return

    print("WEB_INTERFACE: Starting video frame generation for feed.")
    while True:
        if not local_agent.video_monitor.cap or not local_agent.video_monitor.cap.isOpened():
            print("WEB_INTERFACE: Video stream - camera became unavailable.")
            break

        frame = local_agent.video_monitor.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue

        ret, buffer = local_agent.video_monitor.cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(1.0 / local_agent.video_monitor.fps_limit if local_agent.video_monitor.fps_limit and local_agent.video_monitor.fps_limit > 0 else 0.03)


@web_app.route('/')
def index():
    return render_template('index.html')

@web_app.route('/video_feed')
def video_feed():
    return Response(generate_video_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@web_app.route('/start_interaction', methods=['GET'])
def start_interaction():
    print("WEB_INTERFACE: /start_interaction called.")
    local_agent = get_agent()
    if not local_agent:
        return jsonify({"error": "Agent not initialized. Check server logs and .env settings."}), 500
    try:
        local_agent.start_chat()
        agent_event_queue.put({"type": "status", "message": "Agent interaction started. Video capture active."})
        print("WEB_INTERFACE: Agent interaction started successfully.")
        return jsonify({"status": "Agent interaction started. Video capture active."})
    except Exception as e:
        print(f"WEB_INTERFACE: Error starting agent interaction: {e}")
        return jsonify({"error": str(e)}), 500

@web_app.route('/stop_interaction', methods=['GET'])
def stop_interaction():
    print("WEB_INTERFACE: /stop_interaction called.")
    local_agent = get_agent()
    if not local_agent:
        # This case might be okay if user stops an agent that couldn't init
        return jsonify({"status": "Agent was not initialized or already stopped."})
    try:
        local_agent.stop_chat()
        agent_event_queue.put({"type": "status", "message": "Agent stopped. Video capture released."})
        print("WEB_INTERFACE: Agent interaction stopped successfully.")
        return jsonify({"status": "Agent interaction stopped."})
    except Exception as e:
        print(f"WEB_INTERFACE: Error stopping agent interaction: {e}")
        return jsonify({"error": str(e)}), 500

@web_app.route('/send_chat_message', methods=['POST'])
def send_chat_message():
    print("WEB_INTERFACE: /send_chat_message called.")
    data = request.get_json()
    user_message = data.get('message')
    if not user_message:
        return jsonify({"error": "No message provided."}), 400

    local_agent = get_agent()
    if not local_agent:
        return jsonify({"error": "Agent not available."}), 503

    if not local_agent.chat:
        print("WEB_INTERFACE: Chat not active in send_chat_message, starting chat...")
        local_agent.start_chat()

    try:
        model_reply = local_agent.send_text_message(user_message)
        return jsonify({"reply": model_reply})
    except Exception as e:
        print(f"WEB_INTERFACE: Error processing chat message: {e}")
        return jsonify({"error": str(e)}), 500

@web_app.route('/send_voice_message', methods=['POST'])
def send_voice_message():
    print("WEB_INTERFACE: /send_voice_message called.")
    local_agent = get_agent()
    if not local_agent:
        return jsonify({"error": "Agent not available."}), 503

    if not local_agent.chat:
        print("WEB_INTERFACE: Chat not active in send_voice_message, starting chat...")
        local_agent.start_chat()

    print("WEB_INTERFACE: Calling agent.handle_voice_interaction...")
    try:
        # The agent's handle_voice_interaction method is blocking.
        # Results (transcription, model text) are pushed to SSE queue from within the agent method.
        local_agent.handle_voice_interaction(record_duration_seconds=5, event_queue=agent_event_queue)
        print("WEB_INTERFACE: agent.handle_voice_interaction completed.")
        return jsonify({"status": "Voice interaction processed. Check chat for updates via SSE."})
    except Exception as e:
        print(f"WEB_INTERFACE: Error during voice interaction: {e}")
        return jsonify({"error": str(e)}), 500

def generate_agent_events():
    print("WEB_INTERFACE: SSE client connected. Starting event stream.")
    try:
        while True:
            message = agent_event_queue.get()
            # print(f"WEB_INTERFACE SSE: Sending event: {message}") # Can be noisy
            yield f"data: {json.dumps(message)}\n\n"
            agent_event_queue.task_done()
            time.sleep(0.01) # Prevent busy loop if queue is rapidly filled
    except GeneratorExit:
        print("WEB_INTERFACE: SSE client disconnected.")
    except Exception as e:
        print(f"WEB_INTERFACE: Error in SSE event generator: {e}")


@web_app.route('/agent_events')
def agent_events():
    return Response(stream_with_context(generate_agent_events()), mimetype="text/event-stream")

if __name__ == '__main__':
    print("Starting Flask web server directly from web_interface.py (for testing)...")
    print(f"Agent use_vertex: {use_vertex}, project: {project_id}, location: {location}")
    if not (use_vertex and project_id and location):
        print("WARNING: Agent will not be fully functional due to missing .env vars.")

    web_app.run(host=WEB_HOST, port=WEB_PORT, debug=True, threaded=True, use_reloader=False)
