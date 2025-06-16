# web_interface.py
# This module contains the logic for handling the camera and generating
# a video stream for the web interface.

import cv2
import threading

# --- Singleton Camera Instance ---
# We use a singleton pattern here to ensure that only one camera object
# is created and used throughout the application. This prevents conflicts
# where both the video feed and the video interpretation agent try to
# access the camera simultaneously.

_camera = None
_camera_lock = threading.Lock()

def get_camera_instance():
    """
    Initializes and returns a single instance of the camera.
    Uses a lock to make it thread-safe.
    """
    global _camera
    with _camera_lock:
        if _camera is None:
            # Initialize the camera. 0 is typically the default built-in webcam.
            _camera = cv2.VideoCapture(0)
            if not _camera.isOpened():
                raise RuntimeError("Could not start camera.")
    return _camera

# --- Video Frame Generation ---

def generate_frames(camera):
    """
    A generator function that continuously captures frames from the camera,
    encodes them as JPEGs, and yields them for streaming.
    This is what creates the live video feed.
    """
    while True:
        # Read a frame from the camera.
        success, frame = camera.read()
        if not success:
            # If we fail to get a frame, break the loop.
            print("Failed to grab frame")
            break
        else:
            # Encode the frame into JPEG format.
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Failed to encode frame")
                continue
            
            # Convert the buffer to bytes.
            frame_bytes = buffer.tobytes()
            
            # Yield the frame in the format required for a multipart response.
            # The browser will recognize this and display the stream of images.
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')